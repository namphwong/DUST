import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from har_metrics import classification_metrics
from har_models import build_model
from train_har import train, validate_data
import train_har
from evaluate_har import freeze_selection, seed_statistics
from prepare_uci_har import content_hash, write_json


class Phase3Tests(unittest.TestCase):
    def test_seed_statistics(self):
        values = [dict(model='swin', seed=seed, test=dict(accuracy=score, macro_f1=score))
                  for seed, score in [(42, 0.8), (43, 0.9), (44, 1.0)]]
        result = seed_statistics(values)['swin']
        self.assertEqual(result['count'], 3)
        self.assertAlmostEqual(result['macro_f1']['mean'], .9)
        self.assertAlmostEqual(result['macro_f1']['std'], .1)
        self.assertIsNone(seed_statistics(values[:1])['swin']['macro_f1']['std'])

    def test_metrics(self):
        result = classification_metrics([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
        self.assertEqual(result['macro_f1'], 1)
        self.assertEqual(result['accuracy'], 1)
        # One missed class and one false positive: F1 = [2/3, 0, 1, 1, 1, 1].
        result = classification_metrics([0, 1, 2, 3, 4, 5], [0, 0, 2, 3, 4, 5])
        self.assertAlmostEqual(result['macro_f1'], (2/3+4)/6)
        self.assertEqual(result['confusion_matrix'][1][0], 1)
        self.assertEqual(result['per_class'][1]['recall'], 0)
        for truth, pred in [([], []), ([6], [0]), ([0], [0, 1]), ([float('nan')], [0])]:
            with self.assertRaises(ValueError):
                classification_metrics(truth, pred)

    def test_model_forward_backward(self):
        root = Path(__file__).resolve().parents[1]
        torch.set_num_threads(2)
        for family in ['cnn', 'swin']:
            config = json.loads((root / f'configs/train-{family}.json').read_text())
            model = build_model(config)
            logits = model(torch.randn(2, 3, 128))
            self.assertEqual(tuple(logits.shape), (2, 6))
            torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.assertTrue(any(p.grad.abs().sum() > 0 for p in model.parameters() if p.grad is not None))

    def make_data(self, directory):
        rng = np.random.default_rng(42)
        manifest = {'splits': {}}
        for split, subject in [('train', 1), ('val', 2)]:
            n = 12
            np.savez(directory / f'{split}.npz', X=rng.normal(size=(n, 3, 128)).astype('float32'),
                     y=np.tile(np.arange(6, dtype='int64'), 2), subject_id=np.full(n, subject, dtype='int64'),
                     source_index=np.arange(n, dtype='int64'))
            manifest['splits'][split] = dict(subject_ids=[subject], source_indices_zero_based=list(range(n)))
        write_json(directory / 'split-manifest.json', manifest)
        write_json(directory / 'preprocessing.json', {'mean': [0, 0, 0], 'std': [1, 1, 1]})
        write_json(directory / 'labels.json', {'classes': list(range(6))})

    def test_train_no_test_and_resume(self):
        # No test.npz exists: training and selection must still work.
        with tempfile.TemporaryDirectory(prefix='dust-phase3-test-') as tmp:
            root = Path(tmp)
            self.make_data(root)
            config = dict(experiment='unit', model='cnn', seed=42, epochs=2, patience=12,
                          batch_size=6, lr=0.001, weight_decay=0.01, warmup_epochs=1, clip_grad_norm=1,
                          model_kwargs={'channels': [4, 8], 'dropout': 0.1})
            output = root / 'cnn-seed42'
            result = train(config, root, output, 'cpu', 2)
            self.assertFalse(result['test_used_for_training'])
            self.assertEqual(result['epochs_completed'], 2)
            before = torch.load(output / 'last.pth', weights_only=False)
            resumed = train(config, root, output, 'cpu', 2, resume=True)
            self.assertEqual(result['checkpoint_sha256'], resumed['checkpoint_sha256'])
            for key, value in before['model_state_dict'].items():
                torch.testing.assert_close(value, torch.load(output / 'last.pth', weights_only=False)['model_state_dict'][key], rtol=0, atol=0)
            with self.assertRaises(ValueError):
                train(dict(config, lr=0.002), root, output, 'cpu', 2, resume=True)
            interrupted_output = root / 'interrupted'
            def interrupt_after_checkpoint(path, value):
                write_json(path, value)
                if path.name == 'history.json' and len(value) == 1:
                    raise RuntimeError('Simulated interruption after epoch 1 checkpoint')
            with patch.object(train_har, 'write_json', side_effect=interrupt_after_checkpoint):
                with self.assertRaisesRegex(RuntimeError, 'Simulated interruption'):
                    train(config, root, interrupted_output, 'cpu', 2)
            resumed = train(config, root, interrupted_output, 'cpu', 2, resume=True)
            after = torch.load(interrupted_output / 'last.pth', weights_only=False)
            for key, value in before['model_state_dict'].items():
                torch.testing.assert_close(value, after['model_state_dict'][key], rtol=0, atol=0)
            self.assertEqual(resumed['best_validation_macro_f1'], result['best_validation_macro_f1'])

    def test_subject_leakage_rejected(self):
        with tempfile.TemporaryDirectory(prefix='dust-phase3-test-') as tmp:
            root = Path(tmp)
            self.make_data(root)
            with np.load(root / 'val.npz') as source:
                data = dict(source)
            data['subject_id'][:] = 1
            np.savez(root / 'val.npz', **data)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                validate_data(root)

    def test_selection_frozen_without_test(self):
        with tempfile.TemporaryDirectory(prefix='dust-phase3-test-') as tmp:
            root = Path(tmp)
            self.make_data(root)
            data = validate_data(root)
            runs = []
            for name, score in [('cnn-seed42', .8), ('swin-seed42', .9), ('swin-seed43', .85)]:
                run = root / name
                run.mkdir()
                # Dummy checkpoint content: selection uses completed summary/hash, not deserialization.
                (run / 'best.pth').write_bytes(b'unit test')
                import hashlib
                write_json(run / 'summary.json', dict(config=dict(model=name.split('-')[0], seed=int(name[-2:])),
                           best_validation_macro_f1=score, best_epoch=1,
                           checkpoint_sha256=hashlib.sha256(b'unit test').hexdigest(),
                           data_hash=content_hash(*data[0], *data[1]),
                           preprocessing=json.loads((root / 'preprocessing.json').read_text()),
                           labels=json.loads((root / 'labels.json').read_text())))
                runs.append(run)
            output = root / 'evaluation'
            output.mkdir()
            selected = freeze_selection(runs, output, root)
            self.assertEqual(selected['deployment_run'], 'swin-seed42')
            self.assertFalse(selected['test_used_for_selection'])
            self.assertEqual(selected, freeze_selection(runs, output, root))
            with self.assertRaisesRegex(ValueError, 'frozen'):
                freeze_selection(runs[:2], output, root)


if __name__ == '__main__':
    unittest.main()
