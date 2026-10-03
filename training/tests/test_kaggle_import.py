"""Security checks do not require executing or unpickling remote checkpoints."""
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from import_kaggle_results import verify_and_import


class KaggleImportTests(unittest.TestCase):
    def test_unsafe_archive_rejected_before_write(self):
        payload = (np.zeros((1, 3, 128), dtype=np.float32), np.zeros(1, dtype=np.int64),
                   np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64))
        for member in ['../escape.json', '/escape.json', 'artifacts/evil.exe', 'artifacts/C:/bad.json', 'artifacts\\bad.json']:
            with self.subTest(member=member), tempfile.TemporaryDirectory(prefix='dust-import-test-') as directory:
                root = Path(directory)
                archive = root / 'bad.zip'
                with zipfile.ZipFile(archive, 'w') as file:
                    file.writestr(member, '{}')
                with patch('import_kaggle_results.validate_data', return_value=(payload, payload)), \
                     patch('import_kaggle_results.load_data', return_value=payload):
                    with self.assertRaises(ValueError):
                        verify_and_import(archive, root / 'output')
                self.assertFalse((root / 'output').exists())


if __name__ == '__main__':
    unittest.main()
