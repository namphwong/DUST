"""Technical image fixtures; distinct from actual user-attested Paint35 source."""
from copy import deepcopy
import csv
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from training.drawai import paint_data as paint
from training.drawai.private_data import dump,load,sha


def arrays(n=2):
    images=np.full((n,64,64),255,dtype=np.uint8)
    for i in range(n):images[i,10:50,20+i:25+i]=0
    return dict(images=images,labels=np.arange(n,dtype=np.int8)%10,person_ids=np.ones(n,dtype=np.int8),
        sample_ids=np.ones(n,dtype=np.int8),description=np.asarray('Synthetic technical test fixture'))

def provenance():
    return dict(version='drawai-paint-provenance-v1',source_sha256=paint.SOURCE_SHA256,source_kind='human-user-attested',capture_tool='Paint',
        person_ids_are_distinct_people=True,permissions=dict(training=True,evaluation=True,kaggle_publication=True),metadata_description_is_incorrect=True,
        original_strokes_available=False,attestation_source='Explicit human user confirmation in this project chat',confirmed_date='2026-10-18')

class PaintDataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def source(self,value=None):
        path=self.root/'source.npz';np.savez_compressed(path,**(arrays() if value is None else value));return path
    def test_axis_and_normalization_are_spatial_rows_not_strokes(self):
        image=np.full((64,64),255,dtype=np.uint8);image[7,12]=0
        values=paint.swin_rows(image)
        self.assertEqual(values.shape,(64,64));self.assertEqual(values.dtype,np.float32)
        self.assertEqual(values[12,7],1);self.assertEqual(values[7,12],0)
        self.assertFalse(paint.contract()['swin']['existing_stroke_checkpoint_compatible'])
        with self.assertRaises(ValueError):paint.swin_rows(image.astype(np.float32))
    def test_dino_rgb_resize_and_normalization(self):
        from training.drawai.raster import normalize_rgb
        for pixel in (0,255):
            rgb=paint.dino_rgb(np.full((64,64),pixel,dtype=np.uint8));self.assertEqual(rgb.shape,(224,224,3))
            self.assertTrue(np.array_equal(rgb[:,:,0],rgb[:,:,2]))
            values=normalize_rgb(rgb)
            np.testing.assert_allclose(values[:,0,0],(pixel/255-np.array([.485,.456,.406]))/np.array([.229,.224,.225]),atol=1e-6)
    def test_npz_strict_shape_labels_and_pickle_free_load(self):
        value,meta=paint.read_source(self.source(),expected_sha=None);self.assertEqual(value['images'].shape,(2,64,64))
        for mutate in [lambda v:v.update(images=np.zeros((2,32,32),dtype=np.uint8)),lambda v:v.update(labels=np.array([0,10])),
                       lambda v:v.update(labels=np.array(['0','1'])),lambda v:v.update(description=np.array(['not scalar'])),
                       lambda v:v.update(images=np.array([{},{}],dtype=object))]:
            v=arrays();mutate(v)
            with self.subTest(mutate=mutate),self.assertRaises((ValueError,TypeError)):paint.read_source(self.source(v),expected_sha=None)
    def test_source_fingerprint_and_unexpected_member_rejected(self):
        path=self.source()
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):paint.read_source(path)
        value=arrays();value['unexpected']=np.array([1])
        with self.assertRaisesRegex(ValueError,'members'):paint.read_source(self.source(value),expected_sha=None)
    def test_provenance_requires_exact_confirmed_scope(self):
        paint.check_provenance(provenance(),paint.SOURCE_SHA256)
        for change in [lambda v:v['permissions'].update(kaggle_publication=False),lambda v:v.update(metadata_description_is_incorrect=False),
                       lambda v:v.update(person_ids_are_distinct_people=False),lambda v:v.update(source_sha256='0'*64),lambda v:v.update(email='PII')]:
            v=provenance();change(v)
            with self.assertRaises(ValueError):paint.check_provenance(v,paint.SOURCE_SHA256)
    def test_new_plan_preserves_writer_disjoint_groups_and_quota(self):
        p=paint.policy();self.assertEqual([p[s]['samples'] for s in ('pilot_qc','pilot_reserved','train','validation','test')],[250,750,4000,1000,1000])
        sets=[set(p[s]['person_ids']) for s in ('pilot_qc','train','validation','test')]
        for i,a in enumerate(sets):
            for b in sets[i+1:]:self.assertFalse(a&b)
        counts={s:0 for s in ['pilot-qc','pilot-reserved','train','validation','test']}
        for person in range(1,36):
            for rep in range(1,21):counts[paint.split_for(person,rep)]+=10
        self.assertEqual(counts,{'pilot-qc':250,'pilot-reserved':750,'train':4000,'validation':1000,'test':1000})
    def test_audit_detects_composite_duplicates_blank_and_preserves_metadata(self):
        v=arrays();v['labels'][:]=0;v['images'][1]=v['images'][0]
        report=paint.audit_arrays(v,dict(sha256=paint.SOURCE_SHA256,original_description='Synthetic fixture'))
        self.assertTrue(report['duplicate_composite_keys']);self.assertTrue(report['exact_duplicates'])
        self.assertEqual(report['near_duplicate_pairs'],1)
        v['labels'][1]=1
        crossed=paint.audit_arrays(v,dict(sha256=paint.SOURCE_SHA256,original_description='Synthetic fixture'))
        self.assertTrue(crossed['exact_duplicates'][0]['cross_label']);self.assertFalse(crossed['near_duplicate_groups'])
        self.assertEqual(report['source_kind'],'unverified');self.assertTrue(report['metadata_conflict']['original_mentions_synthetic'])
        v['images'][:]=255;report=paint.audit_arrays(v,dict(sha256=paint.SOURCE_SHA256,original_description='Synthetic fixture'),provenance())
        self.assertEqual(report['blank_or_low_contrast_indices'],[0,1]);self.assertFalse(report['independent_human_verification'])
    def prepared(self):
        (self.root/'images64').mkdir();image=arrays(1)['images'][0];Image.fromarray(image).save(self.root/'images64/test.png')
        from training.drawai.raster import normalize_rgb
        row={field:'' for field in paint.MANIFEST_FIELDS};row.update(sample_id='test',source_index=0,person_id=1,label=3,repetition=1,split='pilot-qc',
            png64='images64/test.png',png_sha256=sha((self.root/'images64/test.png').read_bytes()),swin_rows_sha256=sha(paint.swin_rows(image).tobytes()),
            dinov3_tensor_sha256=sha(normalize_rgb(paint.dino_rgb(image)).tobytes()))
        with (self.root/'manifest.csv').open('w',newline='',encoding='utf-8') as handle:
            w=csv.DictWriter(handle,fieldnames=paint.MANIFEST_FIELDS);w.writeheader();w.writerow(row)
        dump(self.root/'dataset.json',dict(manifest_sha256=sha((self.root/'manifest.csv').read_bytes())))
    def test_both_image_inputs_keep_same_id_label_and_guard_test(self):
        self.prepared();a=list(paint.iter_prepared(self.root,'pilot-qc','swin-rows64'));b=list(paint.iter_prepared(self.root,'pilot-qc','dinov3-rgb224'))
        self.assertEqual(a[0][1:],b[0][1:]);self.assertEqual(a[0][0].shape,(64,64));self.assertEqual(b[0][0].shape,(3,224,224))
        with self.assertRaisesRegex(ValueError,'held-out'):list(paint.iter_prepared(self.root,'test','swin-rows64'))
    def test_prepared_manifest_and_png_tamper_rejected(self):
        self.prepared();path=self.root/'images64/test.png';path.write_bytes(path.read_bytes()+b'modified')
        with self.assertRaisesRegex(ValueError,'PNG hash'):list(paint.iter_prepared(self.root,'pilot-qc','swin-rows64'))
        path=self.root/'manifest.csv';path.write_text(path.read_text()+'modified')
        with self.assertRaisesRegex(ValueError,'Manifest hash'):list(paint.iter_prepared(self.root,'pilot-qc','swin-rows64'))
    def test_swin1d_image_row_input_can_forward_and_backpropagate(self):
        import torch
        from models.SwinTransformer_cls import SwinTransformer_1D
        torch.set_num_threads(4);torch.manual_seed(42)
        model=SwinTransformer_1D(seq_len=64,in_chans=64,num_classes=10,patch_size=2,window_size=8,
            embed_dim=32,depths=[2,2,2],num_heads=[2,4,8],drop_rate=.1,drop_path_rate=.1)
        x=torch.from_numpy(np.stack([paint.swin_rows(image) for image in arrays()['images']]))
        y=model(x);self.assertEqual(tuple(y.shape),(2,10));y.square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()));self.assertTrue(torch.isfinite(y).all())

if __name__=='__main__':unittest.main()
