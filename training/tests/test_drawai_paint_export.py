"""P6 gates and PNG runtime contract; actual trained parity uses1000 test images."""
import base64
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from training.drawai.native_runtime import configure_windows_dlls
configure_windows_dlls()
import numpy as np
import torch
from PIL import Image
from training.drawai import paint_export as p,paint_runtime as runtime
from training.drawai.private_data import dump


class PaintExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.image=np.full((64,64),255,dtype=np.uint8);self.image[10:50,30:34]=0

    def test_model_names_and_shapes_separate_from_legacy(self):
        self.assertEqual(p.spec('swin1d-image')['input_shape'],[1,64,64]);self.assertEqual(p.spec('dinov3-vits16')['input_shape'],[1,3,224,224])
        self.assertIn('rows64',p.spec('swin1d-image')['model_filename']);self.assertNotEqual(p.spec('swin1d-image')['model_filename'],'model.onnx')

    def test_unknown_family_rejected(self):
        with self.assertRaises(ValueError):p.spec('cnn')

    def test_exact_parity_and_metrics(self):
        x=np.eye(10,dtype=np.float32)
        value=p.check_parity(x,x,np.arange(10),'swin1d-image')
        self.assertEqual(value['argmax_disagreements'],0);self.assertEqual(value['onnx_metrics']['accuracy'],1)

    def test_fixed_p2_tolerances(self):
        self.assertEqual(p.TOLERANCES['dinov3-vits16'],{'atol':1e-4,'rtol':1e-3})

    def test_bad_logits_rejected(self):
        x=np.zeros((2,10),dtype=np.float32)
        for actual in (np.zeros((1,10)),np.full((2,10),np.nan),np.full((2,10),1.)):
            with self.subTest(shape=actual.shape),self.assertRaises(ValueError):p.check_parity(x,actual,None,'swin1d-image')

    def test_argmax_change_rejected_inside_numeric_tolerance(self):
        x=np.zeros((1,10),dtype=np.float32);y=x.copy();y[0,1]=1e-8
        with self.assertRaisesRegex(ValueError,'Argmax'):p.check_parity(x,y,None,'swin1d-image')

    def test_png_roundtrip(self):
        self.assertTrue(np.array_equal(runtime.decode_png(p.encode_png(self.image)),self.image))

    def test_invalid_base64_and_magic_rejected(self):
        for value in ('!!',None,'',base64.b64encode(b'not PNG').decode(),'a'*24577):
            with self.subTest(value=str(value)[:10]),self.assertRaises(ValueError):runtime.decode_png(value)

    def test_wrong_png_mode_and_size_rejected(self):
        for image in (Image.new('RGB',(64,64)),Image.new('L',(65,64))):
            b=io.BytesIO();image.save(b,format='PNG')
            with self.assertRaises(ValueError):runtime.decode_png(base64.b64encode(b.getvalue()).decode())

    def test_corrupt_png_rejected(self):
        value=base64.b64encode(b'\x89PNG\r\n\x1a\n'+b'x'*40).decode()
        with self.assertRaises(ValueError):runtime.decode_png(value)

    def test_runtime_uses_correct_preprocessing_both_families(self):
        obj=object.__new__(runtime.Runtime);obj.record={'bundle_sha256':'a'*64};obj.sessions={};obj.torch=torch
        for family in ('swin1d-image','dinov3-vits16'):
            def run(_,inputs):
                x=next(iter(inputs.values()));self.assertEqual(list(x.shape),p.spec(family)['input_shape']);self.assertEqual(x.dtype,np.float32)
                return [np.arange(10,dtype=np.float32)[None]]
            if family=='swin1d-image':
                session=Mock();session.run.side_effect=run
            else:
                def session(x):
                    self.assertEqual(list(x.shape),p.spec(family)['input_shape']);return torch.arange(10,dtype=torch.float32)[None]
            obj.sessions[family]=session
            result=obj.infer(dict(model_id=family,png64_base64=p.encode_png(self.image)))
            self.assertEqual(result['predicted_id'],9);self.assertFalse(result['data_persisted'])

    def test_runtime_rejects_schema_and_invalid_output(self):
        obj=object.__new__(runtime.Runtime);obj.sessions={'swin1d-image':Mock()};obj.record={'bundle_sha256':'a'*64}
        for value in ({},{'model_id':'cnn','png64_base64':'x'},{'model_id':[],'png64_base64':'x'}):
            with self.assertRaises(ValueError):obj.infer(value)
        obj.sessions['swin1d-image'].run.return_value=[np.full((1,10),np.nan)]
        with self.assertRaisesRegex(ValueError,'ONNX output'):obj.infer(dict(model_id='swin1d-image',png64_base64=p.encode_png(self.image)))

    def test_missing_external_anchor_rejected(self):
        dump(self.root/'bundle-manifest.json',{})
        with self.assertRaisesRegex(ValueError,'trusted bundle'):p.verify_bundle(self.root,None)

    def test_changed_manifest_rejected(self):
        dump(self.root/'bundle-manifest.json',{'bundle_sha256':'a'*64})
        with self.assertRaisesRegex(ValueError,'anchor'):p.verify_bundle(self.root,'a'*64)

    def test_fixtures_are_procedural_and_reproducible(self):
        a=p.fixtures();b=p.fixtures();self.assertEqual(len(a),9)
        self.assertTrue(all(x[0]==y[0] and np.array_equal(x[1],y[1]) for x,y in zip(a,b)))


if __name__=='__main__':unittest.main()
