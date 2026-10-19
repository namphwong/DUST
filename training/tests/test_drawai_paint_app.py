import base64
import unittest
import tempfile
import os
from pathlib import Path
from training.drawai.paint_app_runtime import adapt_request,ADAPTER,PROTOCOL,Runtime,BUNDLE_SHA256,ROOT
from training.drawai.paint_canvas import pixels64


class PaintAppTest(unittest.TestCase):
    def request(self):
        pixels=pixels64([[{'x':80,'y':80},{'x':320,'y':320}]])
        return dict(protocol=PROTOCOL,request_id=1,drawing_revision=0,model_id='dinov3-vits16',canvas_adapter=ADAPTER,
            pixels64_base64=base64.b64encode(pixels.tobytes()).decode())

    def test_lossless_adapter(self):
        from training.drawai.paint_runtime import decode_png
        request=self.request();value,digest=adapt_request(request)
        self.assertEqual(decode_png(value['png64_base64']).tobytes(),base64.b64decode(request['pixels64_base64']))
        self.assertEqual(len(digest),64)

    def test_rejects_wrong_contract(self):
        for key,value in [('request_id',True),('drawing_revision',-1),('model_id',[]),('model_id','legacy'),('canvas_adapter','old'),
                          ('protocol','old'),('pixels64_base64','!'),('pixels64_base64',base64.b64encode(bytes([255])*4096).decode())]:
            request=self.request();request[key]=value
            with self.subTest(key=key,value=str(value)[:20]),self.assertRaises((ValueError,TypeError)):adapt_request(request)
        request=self.request();request['extra']=1
        with self.assertRaises(ValueError):adapt_request(request)

    def test_round_cap_and_no_join_between_strokes(self):
        image=pixels64([[{'x':50,'y':50}],[{'x':350,'y':350}]])
        self.assertLess(int(image.min()),255)
        self.assertEqual(int(image[32,32]),255)
        self.assertEqual(image.shape,(64,64))

    def damaged_bundle(self,corrupt):
        source=ROOT/'models/exported/paint35-v1'
        with tempfile.TemporaryDirectory(prefix='drawai-p7-',dir=source.parent) as temporary:
            directory=Path(temporary)
            for path in source.rglob('*'):
                if path.is_file():
                    target=directory/path.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);os.link(path,target)
            target=directory/'swin1d-image/drawai_swin1d-image_paint35-main-v1_rows64_fp32_v1.onnx'
            # Unlink temporary hardlink BEFORE writing, preserving original model bytes.
            target.unlink()
            if corrupt:target.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Unexpected/missing bundle file|Bundle file hash mismatch'):Runtime(directory,BUNDLE_SHA256)

    def test_missing_model_blocks_startup(self):self.damaged_bundle(False)
    def test_corrupt_model_blocks_startup(self):self.damaged_bundle(True)


if __name__=='__main__':unittest.main()
