"""Synthetic temporary fixtures test the ingestion gate; they are never human evidence."""
import json
import tempfile
import unittest
from pathlib import Path

from training.drawai.evaluate_canvas import validate_manifest


class CanvasGateTests(unittest.TestCase):
    def fixture(self,root):
        people=[dict(id=f'P{i:03}',consent=True,input_device='mouse',device_description='synthetic test fixture',browser='synthetic test fixture') for i in range(1,6)]
        samples=[]
        for i,p in enumerate(people):
            for label in range(10):
                for repetition in range(1,6):
                    name=f"{p['id']}-{label}-{repetition}.json"
                    drawing=dict(schema_version='drawai-strokes-v1',coordinate_system='canvas-y-down',canvas=dict(width=400,height=400),
                                 strokes=[[dict(x=i*10+label,y=repetition),dict(x=100+label,y=150+repetition)]],reference_label=label,source=dict(kind='user'))
                    (root/name).write_text(json.dumps(drawing))
                    samples.append(dict(participant_id=p['id'],label=label,repetition=repetition,file=name))
        manifest=dict(protocol_version='drawai-canvas-eval-v1',release_version='drawai-swin-v1',human_source_attested=True,participants=people,samples=samples)
        path=root/'manifest.json';path.write_text(json.dumps(manifest));return path,manifest

    def test_empty_or_unattested_data_cannot_be_evaluated(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'manifest.json';path.write_text(json.dumps(dict(protocol_version='drawai-canvas-eval-v1',release_version='drawai-swin-v1',human_source_attested=False,participants=[],samples=[])))
            with self.assertRaisesRegex(ValueError,'attestation'):validate_manifest(path)
            path.write_text(json.dumps(dict(protocol_version='drawai-canvas-eval-v1',release_version='drawai-swin-v1',human_source_attested=True,participants=[],samples=[])))
            with self.assertRaisesRegex(ValueError,'five'):validate_manifest(path)
            path,manifest=self.fixture(Path(folder));manifest['participants'][0]['consent']=False;path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'consent'):validate_manifest(path)

    def test_full_matrix_gate_and_missing_or_duplicate_repetition(self):
        with tempfile.TemporaryDirectory() as folder:
            path,manifest=self.fixture(Path(folder));_,rows=validate_manifest(path);self.assertEqual(len(rows),250)
            manifest['samples'].pop();path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'five times'):validate_manifest(path)
            manifest['samples'].append(manifest['samples'][0]);path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'Duplicate participant'):validate_manifest(path)

    def test_dataset_source_and_bad_timestamps_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path,manifest=self.fixture(Path(folder));sample=path.parent/manifest['samples'][0]['file'];drawing=json.loads(sample.read_text())
            drawing['source']['kind']='dataset';sample.write_text(json.dumps(drawing))
            with self.assertRaisesRegex(ValueError,'human user'):validate_manifest(path)
            drawing['source']['kind']='user';drawing['strokes'][0][0]['t']=2;drawing['strokes'][0][1]['t']=1;sample.write_text(json.dumps(drawing))
            with self.assertRaisesRegex(ValueError,'timestamps'):validate_manifest(path)

    def test_paths_outside_root_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path,manifest=self.fixture(Path(folder));manifest['samples'][0]['file']='../outside.json';path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'escapes'):validate_manifest(path)


if __name__=='__main__':unittest.main()
