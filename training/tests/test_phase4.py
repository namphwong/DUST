"""Release contract checks against the generated, shipped Phase 4 artifact."""
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / 'web/public/models/har-swin-kaggle-v1'


def read(name):
    return json.loads((TARGET / name).read_text(encoding='utf-8'))


class Phase4Tests(unittest.TestCase):
    def test_release_hashes_and_trained_contract(self):
        manifest, report = read('release.json'), read('report.json')
        self.assertTrue(manifest['trained'])
        self.assertTrue(report['trained'])
        self.assertEqual(manifest['version'], report['version'])
        self.assertEqual(report['input_shape'], [1, 3, 128])
        self.assertEqual(report['output_shape'], [1, 6])
        self.assertFalse(report['quantized'])
        self.assertFalse(report['weights_modified'])
        for name, entry in manifest['files'].items():
            content = (TARGET / name).read_bytes()
            self.assertEqual(len(content), entry['bytes'])
            self.assertEqual(hashlib.sha256(content).hexdigest(), entry['sha256'])

    def test_full_test_parity_and_accuracy(self):
        report = read('report.json')
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['evaluation_samples'], 2947)
        self.assertEqual(report['pytorch_onnx_prediction_agreement'], 2947)
        self.assertEqual(report['kaggle_onnx_prediction_agreement'], 2947)
        self.assertLessEqual(report['macro_f1_degradation'], 0.005)
        self.assertAlmostEqual(report['onnx_test']['macro_f1'], report['kaggle_test_macro_f1'], places=12)

    def test_fixture_normalization_and_label_order(self):
        fixtures, prep, labels = read('fixtures.json'), read('preprocessing.json'), read('labels.json')
        self.assertEqual(len(fixtures['cases']), 14)
        self.assertEqual([c['id'] for c in labels['classes']], list(range(6)))
        self.assertEqual([c['code'] for c in labels['classes']],
                         ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING'])
        mean = np.asarray(prep['mean'], dtype=np.float32)[:, None]
        std = np.asarray(prep['std'], dtype=np.float32)[:, None]
        for case in fixtures['cases']:
            raw = np.asarray(case['raw'], dtype=np.float32).reshape(3, 128)
            normalized = np.asarray(case['input'], dtype=np.float32).reshape(3, 128)
            np.testing.assert_array_equal((raw - mean) / std, normalized)
            self.assertTrue(np.isfinite(case['expected']).all())
        self.assertEqual([c['reference_label'] for c in fixtures['cases'] if c['kind'] == 'uci-reference'], list(range(6)))


if __name__ == '__main__':
    unittest.main()
