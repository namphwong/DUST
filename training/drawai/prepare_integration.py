"""Generate fixed D6 JSON-to-tensor-to-trained-logits references, not human handwriting data."""
from .legacy_release import LEGACY_FILENAME

import json
import os
import sys

_dll_handle=None
if sys.platform=='win32' and os.environ.get('DUST_DLL_DIRECTORY'):
    _dll_handle=os.add_dll_directory(os.environ['DUST_DLL_DIRECTORY'])
import numpy as np
import torch
import onnxruntime as ort

from .prepare_data import ROOT,digest,write_json
from .preprocessing import preprocess
from .models import build_model
from .verify_feasibility import geometry_cases


def main():
    experiment=ROOT/'artifacts/drawai/d3-local-v1';target=ROOT/'web/public/models/drawai-swin-v1'
    selection=json.loads((experiment/'selection.json').read_text())
    path=experiment/'runs'/selection['deployment_run']/'best.pth'
    assert digest(path)==selection['checkpoint_sha256']
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=build_model('swin',128).eval();model.load_state_dict(saved['model_state_dict'],strict=True);torch.set_num_threads(4)
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(target/LEGACY_FILENAME),sess_options=options,providers=['CPUExecutionProvider'])
    cases=[]
    for fixture in geometry_cases(128):
        strokes=[[dict(x=p['x'],y=-p['y'] if fixture['y_axis']=='up' else p['y']) for p in s] for s in fixture['strokes']]
        drawing=dict(schema_version='drawai-strokes-v1',coordinate_system='canvas-y-down',canvas=dict(width=500,height=500),strokes=strokes,
                     reference_label=fixture.get('label'),source=dict(kind='dataset' if 'sample_id' in fixture else 'synthetic'))
        if 'sample_id' in fixture:drawing['source'].update(sample_id=fixture['sample_id'],split='train')
        values=preprocess(strokes,128).reshape(1,3,128)
        with torch.inference_mode():expected=model(torch.from_numpy(values)).numpy()[0]
        observed=session.run(None,dict(drawing=values))[0][0]
        np.testing.assert_allclose(observed,expected,atol=1e-5,rtol=1e-4)
        cases.append(dict(name=fixture['name'],drawing=drawing,input=values.reshape(-1).tolist(),pytorch_logits=expected.tolist(),onnx_logits=observed.tolist()))
    base=dict(schema_version='drawai-strokes-v1',coordinate_system='canvas-y-down',canvas=dict(width=400,height=400),source=dict(kind='synthetic'))
    invalid=[dict(name='blank',drawing=dict(**base,strokes=[]),code='SCHEMA'),
             dict(name='dot',drawing=dict(**base,strokes=[[dict(x=0,y=0)]]),code='TOO_SHORT'),
             dict(name='empty-stroke',drawing=dict(**base,strokes=[[]]),code='EMPTY_STROKE'),
             dict(name='timing-missing',drawing=dict(**base,strokes=[[dict(x=0,y=0,t=0),dict(x=1,y=1)]]),code='TIMESTAMP'),
             dict(name='timing-backwards',drawing=dict(**base,strokes=[[dict(x=0,y=0,t=2),dict(x=1,y=1,t=1)]]),code='TIMESTAMP'),
             dict(name='string-coordinate',drawing=dict(**base,strokes=[[dict(x='1',y=0),dict(x=2,y=2)]]),code='NUMBER'),
             dict(name='too-many-strokes',drawing=dict(**base,strokes=[[dict(x=0,y=0),dict(x=1,y=1)]]*17),code='SCHEMA')]
    out=ROOT/'web/public/verification/drawai/d6-fixtures.json'
    write_json(out,dict(version='drawai-d6-v1',release_sha256=digest(target/'release.json'),checkpoint_sha256=digest(path),
                        model_sha256=digest(target/LEGACY_FILENAME),atol=1e-5,rtol=1e-4,cases=cases,invalid=invalid))
    write_json(ROOT/'artifacts/drawai/verification/fixture-provenance.json',dict(status='passed',
               fixture_sha256=digest(out),valid_cases=len(cases),invalid_cases=len(invalid),
               human_data=False,checkpoint_sha256=digest(path),model_sha256=digest(target/LEGACY_FILENAME)))
    print(f'Prepared {len(cases)} JSON/tensor/PyTorch/ORT fixtures and {len(invalid)} invalid cases.')


if __name__=='__main__':main()
