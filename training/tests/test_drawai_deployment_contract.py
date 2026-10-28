"""VR0 locks: reject integrity changes and unsafe re-signed deployment policies."""
import copy
import unittest
from tools.verify_vr0 import read,CONTRACT,ANCHOR,validate_contract,local_path
from training.drawai.private_data import canonical,sha


class DeploymentContractTests(unittest.TestCase):
    def setUp(self):
        self.contract=read(CONTRACT);self.anchor=read(ANCHOR)['contract_sha256']

    def resign(self,value):
        value['contract_sha256']=sha(canonical({k:v for k,v in value.items() if k!='contract_sha256'}))
        return value['contract_sha256']

    def test_current_contract_passes(self):
        self.assertEqual(validate_contract(self.contract,self.anchor)['phase'],'VR0')

    def test_modified_content_fails_external_anchor(self):
        value=copy.deepcopy(self.contract);value['models']['dinov3-vits16']['seed']=99
        self.resign(value)
        with self.assertRaisesRegex(ValueError,'anchor/hash'):validate_contract(value,self.anchor)

    def test_resigned_default_change_rejected(self):
        value=copy.deepcopy(self.contract);value['architecture']['default_model']='swin1d-image'
        with self.assertRaisesRegex(ValueError,'Default/baseline'):validate_contract(value,self.resign(value))

    def test_resigned_silent_fallback_rejected(self):
        value=copy.deepcopy(self.contract);value['architecture']['silent_fallback']=True
        with self.assertRaisesRegex(ValueError,'fallback'):validate_contract(value,self.resign(value))

    def test_resigned_diagnostic_onnx_as_primary_rejected(self):
        value=copy.deepcopy(self.contract);value['models']['dinov3-vits16']['spec']['runtime_format']='onnx'
        with self.assertRaisesRegex(ValueError,'format/shape'):validate_contract(value,self.resign(value))

    def test_resigned_relaxed_tolerance_rejected(self):
        value=copy.deepcopy(self.contract);value['models']['dinov3-vits16']['tolerance']['atol']=.01
        with self.assertRaisesRegex(ValueError,'tolerance'):validate_contract(value,self.resign(value))

    def test_resigned_collection_route_published_rejected(self):
        value=copy.deepcopy(self.contract);value['profiles']['public_target']['public_routes'].append('/collect')
        with self.assertRaisesRegex(ValueError,'route scope'):validate_contract(value,self.resign(value))

    def test_resigned_memory_over_budget_rejected(self):
        value=copy.deepcopy(self.contract);value['budgets']['backend']['peak_cgroup_memory_max_bytes']=600000000
        with self.assertRaisesRegex(ValueError,'Memory guard'):validate_contract(value,self.resign(value))

    def test_resigned_false_cloud_success_rejected(self):
        value=copy.deepcopy(self.contract);value['claims']['cloud_memory_tested']=True
        with self.assertRaisesRegex(ValueError,'completion claim'):validate_contract(value,self.resign(value))

    def test_local_paths_cannot_escape_project(self):
        with self.assertRaisesRegex(ValueError,'outside project'):local_path('../outside-drawai.bin')


if __name__=='__main__':unittest.main()
