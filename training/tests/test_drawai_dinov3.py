"""Run these optional tests in .venv-dinov3; baseline env skips cleanly."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

AVAILABLE=importlib.util.find_spec('transformers') is not None


@unittest.skipUnless(AVAILABLE,'Optional P2 environment required')
class DinoTests(unittest.TestCase):
    def test_random_initialization_requires_explicit_smoke_choice(self):
        from training.drawai.models import build_model
        from training.drawai.dinov3_model import WeightsUnavailableError
        with self.assertRaises(WeightsUnavailableError):build_model('dinov3_vits16')
        with self.assertRaises(ValueError):build_model('dinov3_vits16',weights_dir='missing',smoke_only=True)

    def test_missing_or_wrong_weights_never_fall_back(self):
        from training.drawai.dinov3_model import DinoDigitClassifier,WeightsUnavailableError
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            with self.assertRaises(WeightsUnavailableError):DinoDigitClassifier.from_pretrained(p)
            for name in ['config.json','preprocessor_config.json','model.safetensors']:(p/name).write_text('{}')
            with patch('transformers.DINOv3ViTModel.from_pretrained',side_effect=AssertionError('No unverified weights may load')):
                with self.assertRaisesRegex(ValueError,'hash differs'):DinoDigitClassifier.from_pretrained(p)

    def test_exact_full_architecture_freeze_gradient_and_register_pooling(self):
        import torch
        from training.drawai.models import build_model
        torch.set_num_threads(4)
        model=build_model('dinov3_vits16',smoke_only=True)
        with self.assertRaisesRegex(ValueError,'NCHW'):
            model(torch.zeros(1,3,128))
        self.assertEqual(sum(p.numel() for p in model.backbone.parameters()),21596544)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),7690)
        model.train();self.assertFalse(model.backbone.training);self.assertTrue(model.head.training)
        pixels=torch.randn(2,3,224,224)
        with torch.no_grad():
            sequence=model.backbone(pixel_values=pixels).last_hidden_state
            expected=torch.cat([sequence[:,0],sequence[:,5:].mean(1)],dim=1)
            torch.testing.assert_close(model.forward_features(pixels),expected)
        logits=model(pixels);self.assertEqual(tuple(logits.shape),(2,10))
        logits.square().mean().backward()
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertFalse(model.provenance['pretrained_loaded']);self.assertTrue(model.provenance['smoke_only'])

    def test_config_tamper_is_rejected_before_loading_weights(self):
        import hashlib
        from training.drawai.dinov3_model import DinoDigitClassifier
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            for name in ['config.json','preprocessor_config.json','model.safetensors']:(p/name).write_bytes(b'{}')
            with patch('training.drawai.dinov3_model.OFFICIAL_WEIGHTS_SHA256',hashlib.sha256(b'{}').hexdigest()),patch('transformers.DINOv3ViTModel.from_pretrained',side_effect=AssertionError('No altered config may load')):
                with self.assertRaisesRegex(ValueError,'config blob mismatch'):
                    DinoDigitClassifier.from_pretrained(p)

    def test_public_pretrained_native_state_pooling_and_gradients(self):
        import torch
        from safetensors.torch import load_file
        from training.drawai.models import build_model
        from training.drawai.dinov3_model import TIMM_WEIGHTS_SHA256
        root=Path(__file__).resolve().parents[2]
        folder=root/'models/pretrained/dinov3-vits16-timm-qkvb'
        if not (folder/'model.safetensors').exists():self.skipTest('Run the pinned public downloader for this integration test')
        torch.set_num_threads(4)
        model=build_model('dinov3_vits16',weights_dir=folder)
        self.assertTrue(model.provenance['pretrained_loaded'])
        self.assertEqual(model.provenance['weights_sha256'],TIMM_WEIGHTS_SHA256)
        self.assertEqual(sum(p.numel() for p in model.parameters()),21603850)
        state=load_file(str(folder/'model.safetensors'))
        self.assertTrue(all(torch.equal(model.backbone.model.state_dict()[key],value) for key,value in state.items()))
        model.train();self.assertFalse(model.backbone.training)
        inputs=torch.randn(2,3,224,224)
        with torch.no_grad():
            native=model.backbone.model.forward_features(inputs)
            self.assertEqual(tuple(native.shape),(2,201,384))
            expected=torch.cat((native[:,0],native[:,5:].mean(1)),dim=1)
            torch.testing.assert_close(model.forward_features(inputs),expected,atol=0,rtol=0)
        output=model(inputs);self.assertEqual(tuple(output.shape),(2,10))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))

    def test_public_pretrained_config_tamper(self):
        import json,shutil
        from training.drawai.dinov3_model import DinoDigitClassifier
        root=Path(__file__).resolve().parents[2];source=root/'models/pretrained/dinov3-vits16-timm-qkvb'
        if not (source/'model.safetensors').exists():self.skipTest('Public pinned checkpoint required')
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);shutil.copy2(source/'model.safetensors',folder/'model.safetensors')
            config=json.loads((source/'config.json').read_text());config['pretrained_cfg']['mean']=[0,0,0]
            (folder/'config.json').write_text(json.dumps(config))
            with patch('timm.create_model',side_effect=AssertionError('No altered configuration may instantiate a model')):
                with self.assertRaisesRegex(ValueError,'config blob mismatch'):DinoDigitClassifier.from_pretrained(folder)


if __name__=='__main__':unittest.main()
