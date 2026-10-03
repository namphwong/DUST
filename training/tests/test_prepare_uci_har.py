import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training'))
from prepare_uci_har import fit_preprocessing, load_split, normalize, split_subjects

CONFIG = json.loads((ROOT / 'training/configs/data.json').read_text())


class PreparationTests(unittest.TestCase):
    def test_split_is_subject_disjoint_and_repeatable(self):
        subjects = np.repeat(np.arange(1, 22), 6)
        first = split_subjects(subjects, 42, 5)
        second = split_subjects(subjects, 42, 5)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        train, val, selected = first
        self.assertFalse(set(subjects[train]) & set(subjects[val]))
        self.assertEqual(len(selected), 5)
        np.testing.assert_array_equal(np.sort(np.concatenate([train, val])), np.arange(len(subjects)))

    def test_statistics_fit_only_training_and_keep_layout(self):
        data = np.random.default_rng(5).normal(size=(126, 3, 128)).astype(np.float32)
        subjects = np.repeat(np.arange(1, 22), 6)
        train, val, _ = split_subjects(subjects, 42, 5)
        mean, std = fit_preprocessing(data[train], CONFIG)
        data[val] = 1e6
        again_mean, again_std = fit_preprocessing(data[train], CONFIG)
        np.testing.assert_array_equal(mean, again_mean)
        np.testing.assert_array_equal(std, again_std)
        transformed = normalize(data[train], mean, std)
        self.assertEqual(transformed.shape, (96, 3, 128))
        self.assertEqual(transformed.dtype, np.float32)
        np.testing.assert_allclose(transformed.mean(axis=(0, 2)), 0, atol=1e-6)
        np.testing.assert_allclose(transformed.std(axis=(0, 2)), 1, atol=1e-6)
        np.testing.assert_allclose(transformed[0, 1], (data[train[0], 1]-mean[1])/std[1])

    def test_constant_channel_and_nonfinite_input(self):
        data = np.ones((2, 3, 128), dtype=np.float32)
        mean, std = fit_preprocessing(data, CONFIG)
        self.assertTrue(np.isfinite(normalize(data, mean, std)).all())
        data[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_preprocessing(data, CONFIG)

    def test_source_rejects_nonintegral_ids_and_mismatched_rows(self):
        class FakeSource:
            labels = b'1\n2\n'
            subjects = b'1\n2\n'
            def read(self, path):
                if path.endswith('/y_train.txt'):
                    return self.labels
                if path.endswith('/subject_train.txt'):
                    return self.subjects
                return ((' '.join(['1'] * 128) + '\n') * 2).encode()
        source = FakeSource()
        self.assertEqual(load_split(source, 'train', CONFIG)[0].shape, (2, 3, 128))
        source.labels = b'1.5\n2\n'
        with self.assertRaises(ValueError):
            load_split(source, 'train', CONFIG)
        source.labels = b'1\n'
        with self.assertRaises(ValueError):
            load_split(source, 'train', CONFIG)


if __name__ == '__main__':
    unittest.main()
