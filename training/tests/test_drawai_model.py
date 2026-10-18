"""D2 model contracts and untouched Swin initialization (no ONNX import)."""
import json
import unittest
from pathlib import Path

import torch

from training.drawai.models import build_model, swin_config


class DrawAIModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_config_file_matches_factory(self):
        config = json.loads((Path(__file__).parents[1]/'drawai/configs/swin.json').read_text())
        self.assertEqual(config['model_kwargs'],swin_config(64))
        with self.assertRaises(ValueError):
            build_model('unknown')
        with self.assertRaises(ValueError):
            build_model('swin',63)

    def test_eval_is_deterministic_and_batch_independent(self):
        for length in (64,128):
            with self.subTest(length=length):
                torch.manual_seed(42)
                model=build_model('swin',length).eval()
                inputs=torch.randn(4,3,length)
                with torch.inference_mode():
                    together=model(inputs)
                    singles=torch.cat([model(x[None]) for x in inputs])
                    repeated=model(inputs)
                self.assertEqual(together.shape,(4,10))
                self.assertTrue(torch.isfinite(together).all())
                torch.testing.assert_close(together,singles,atol=1e-6,rtol=1e-5)
                torch.testing.assert_close(together,repeated,atol=0,rtol=0)

    def test_wrong_input_shape_is_rejected(self):
        model=build_model('swin',64).eval()
        with self.assertRaises(RuntimeError):
            model(torch.zeros(1,2,64))
        with self.assertRaises(AssertionError):
            model(torch.zeros(1,3,128))

    def test_original_initialization_learns_attention_gradients(self):
        for length in (64,128):
            with self.subTest(length=length):
                torch.manual_seed(42)
                model=build_model('swin',length)
                block=model.layers[0].blocks[1]
                # Backbone zero-initializes residual post-norm: the first step
                # need not have attention gradients. Do not alter these weights.
                self.assertEqual(float(block.norm1.weight.abs().sum()),0)
                self.assertGreater(block.shift_size,0)
                optimizer=torch.optim.AdamW(model.parameters(),lr=0.003)
                inputs=torch.randn(4,3,length); labels=torch.tensor([0,3,7,9])
                for _ in range(4):
                    optimizer.zero_grad()
                    loss=torch.nn.functional.cross_entropy(model(inputs),labels)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward(); optimizer.step()
                for parameter in (model.patch_embed.proj.weight,block.attn.qkv.weight,
                                  block.mlp.fc1.weight,model.head.weight):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(float(parameter.grad.abs().sum()),0)


if __name__=='__main__':
    unittest.main()
