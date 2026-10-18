"""P4 corruption/leakage guards; full-size positive evidence is the P4 audit."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from training.drawai import paint_dataset as p, paint_data as data
from training.drawai.private_data import sha, dump, canonical
from training.drawai.raster import normalize_rgb


class PaintLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        pixels = np.full((64,64),255,dtype=np.uint8); pixels[10:50,25:30] = 0
        sid = 'paint35_p006_d1_r01'; name = f'images64/{sid}.png'
        (self.root/'images64').mkdir(); Image.fromarray(pixels).save(self.root/name)
        rgb = data.dino_rgb(pixels)
        self.row = dict(sample_id=sid,source_index='1000',person_id='6',label='1',repetition='1',split='train',png64=name,
            png_sha256=sha((self.root/name).read_bytes()),image_sha256=sha(pixels.tobytes()),
            swin_rows_sha256=sha(data.swin_rows(pixels).tobytes()),dinov3_rgb_sha256=sha(rgb.tobytes()),
            dinov3_tensor_sha256=sha(normalize_rgb(rgb).tobytes()))
        self.pixels = pixels

    def test_valid_image_both_representations(self):
        self.assertTrue(np.array_equal(p.check_image(self.root,self.row),self.pixels))
        self.assertEqual(data.swin_rows(self.pixels).shape,(64,64))
        self.assertEqual(normalize_rgb(data.dino_rgb(self.pixels)).shape,(3,224,224))

    def test_png_tampering(self):
        (self.root/self.row['png64']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'PNG hash'):p.check_image(self.root,self.row)

    def test_tensor_tampering(self):
        self.row['swin_rows_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError,'preprocessing'):p.check_image(self.root,self.row)

    def test_path_traversal(self):
        self.row['png64'] = '../outside.png'
        with self.assertRaisesRegex(ValueError,'path'):p.check_image(self.root,self.row)

    def test_writer_split_drift(self):
        self.row['split'] = 'test'
        with self.assertRaisesRegex(ValueError,'split'):p.check_image(self.root,self.row)

    def test_partial_production_quota_rejected(self):
        with self.assertRaisesRegex(ValueError,'quota'):p.quotas([self.row],('train','validation'))

    def test_test_and_pilot_loader_rejected_before_read(self):
        for split in ('test','pilot-qc','pilot-reserved'):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError,'train/validation'):
                next(p.iter_development(self.root,split,'0'*64))

    def test_unknown_representation_rejected(self):
        with self.assertRaisesRegex(ValueError,'representation'):
            next(p.iter_development(self.root,'train','0'*64,'stroke-l128'))

    def test_untrusted_lock_rejected(self):
        dump(self.root/'dataset-lock.json',{'lock_sha256':'a'*64})
        with self.assertRaisesRegex(ValueError,'trusted'):p.verify_lock(self.root,'b'*64)

    def test_metadata_tamper_even_with_matching_claim(self):
        record = {'version':'tampered','lock_sha256':'a'*64}; dump(self.root/'dataset-lock.json',record)
        with self.assertRaisesRegex(ValueError,'trusted'):p.verify_lock(self.root,'a'*64)

    def review_fixture(self):
        group = dict(group_id='near_x',source_indices=[1,2],cross_split=True)
        value = dict(version='drawai-paint-review-v1',bindings={'source':'hash'},reviewed_at='2026-10-17T00:00:00+00:00',
                     exclusions=[],model_based_selection=False,similarity_independence_verified=False,source_already_public=True,
                     near_groups=[dict(**group,decision=p.DECISION,min_pair_different_pixels=1,reason='Retention does not establish independent drawings; preserve fixed split.')])
        return value, {'source':'hash'}, {'near_duplicate_groups':[group]}

    def test_review_retains_disclosed_uncertainty(self):
        p.validate_review(*self.review_fixture())

    def test_stale_review_rejected(self):
        v,b,r = self.review_fixture(); b['source'] = 'new'
        with self.assertRaisesRegex(ValueError,'bindings'):p.validate_review(v,b,r)

    def test_missing_group_rejected(self):
        v,b,r = self.review_fixture(); v['near_groups'] = []
        with self.assertRaisesRegex(ValueError,'decisions'):p.validate_review(v,b,r)

    def test_false_independence_rejected(self):
        v,b,r = self.review_fixture(); v['similarity_independence_verified'] = True
        with self.assertRaisesRegex(ValueError,'unresolved'):p.validate_review(v,b,r)

    def test_model_based_exclusion_rejected(self):
        v,b,r = self.review_fixture(); v['exclusions'] = ['low_accuracy']
        with self.assertRaisesRegex(ValueError,'exclusions'):p.validate_review(v,b,r)

    def test_output_overlap_and_public_rejected(self):
        for target in (self.root/'nested', p.ROOT/'web/public/private-data'):
            with self.subTest(target=target), self.assertRaises(ValueError):p.separate_output(target,self.root)

    def test_exact_copies_cannot_be_marked_retained(self):
        v,b,r = self.review_fixture(); v['near_groups'][0]['min_pair_different_pixels'] = 0
        with self.assertRaisesRegex(ValueError,'retention'):p.validate_review(v,b,r)


if __name__ == '__main__':unittest.main()
