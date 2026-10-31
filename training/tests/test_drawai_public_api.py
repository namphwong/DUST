import base64
import json
import unittest
from fastapi.testclient import TestClient
from training.drawai.public_api import create_app,decode_request,PROTOCOL,ADAPTER,MODEL_ID


def payload():
    pixels=bytearray([255]*4096);pixels[2000:2020]=bytes(20)
    return dict(protocol=PROTOCOL,request_id=1,drawing_revision=7,model_id=MODEL_ID,canvas_adapter=ADAPTER,pixels64_base64=base64.b64encode(pixels).decode())


class FakeRuntime:
    def health(self):return dict(status='ready',models=[MODEL_ID])
    def predict(self,value,pixels):return dict(model_id=MODEL_ID,request_id=value['request_id'],drawing_revision=value['drawing_revision'],logits=[float(i) for i in range(10)],predicted_id=9)


class PublicApiTests(unittest.TestCase):
    def setUp(self):
        self.client=TestClient(create_app(runtime_factory=FakeRuntime,allowed_origins=['https://drawai.example.org']))
        self.client.__enter__();self.addCleanup(self.client.__exit__,None,None,None)

    def test_health_and_inference_actual_http_contract(self):
        self.assertEqual(self.client.get('/health').json()['models'],[MODEL_ID])
        r=self.client.post('/predict',json=payload());self.assertEqual(r.status_code,200);self.assertEqual(r.json()['drawing_revision'],7)

    def test_correct_and_forbidden_cors(self):
        r=self.client.options('/predict',headers={'Origin':'https://drawai.example.org','Access-Control-Request-Method':'POST'})
        self.assertEqual(r.status_code,200);self.assertEqual(r.headers['access-control-allow-origin'],'https://drawai.example.org')
        self.assertEqual(self.client.post('/predict',json=payload(),headers={'Origin':'https://evil.example'}).status_code,403)

    def test_unknown_models_and_bad_ids_rejected(self):
        for patch in [dict(model_id='swin1d-image'),dict(request_id=True),dict(drawing_revision=-1),dict(extra=1)]:
            with self.subTest(patch=patch):self.assertEqual(self.client.post('/predict',json=dict(payload(),**patch)).status_code,400)

    def test_duplicate_json_and_nonfinite_rejected(self):
        for body in ['{"x":0,"x":1}','{"x":NaN}']:
            self.assertEqual(self.client.post('/predict',content=body,headers={'Content-Type':'application/json'}).status_code,400)

    def test_body_limit_and_content_type(self):
        self.assertEqual(self.client.post('/predict',content=' '*8193,headers={'Content-Type':'application/json'}).status_code,413)
        self.assertEqual(self.client.post('/predict',content='{}',headers={'Content-Type':'text/plain'}).status_code,415)

    def test_bad_base64_and_blank_rejected(self):
        for pixels in ['!'*5464,base64.b64encode(bytes([255])*4096).decode()]:
            self.assertEqual(self.client.post('/predict',json=dict(payload(),pixels64_base64=pixels)).status_code,400)

    def test_pixel_transport_lossless(self):
        value=payload();_,pixels=decode_request(json.dumps(value));self.assertEqual(pixels.tobytes(),base64.b64decode(value['pixels64_base64']))

    def test_wildcard_origins_rejected(self):
        with self.assertRaises(ValueError):create_app(runtime_factory=FakeRuntime,allowed_origins=['https://*.vercel.app'])


if __name__=='__main__':unittest.main()
