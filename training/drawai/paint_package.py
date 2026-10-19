"""Portable ONNX bundle validation: deliberately independent of PyTorch/timm."""
from pathlib import Path
import re
from . import paint_data as paint
from .private_data import canonical,sha,load
from .private_dataset import safe_file

FAMILIES=('swin1d-image','dinov3-vits16')
LOCK_HASH='2c3a86c5be2b2e0f952a9c3390d56785473a82634338cec805903cbb1b3ef47a'
TOLERANCES={'swin1d-image':dict(atol=1e-5,rtol=1e-4),'dinov3-vits16':dict(atol=1e-4,rtol=1e-3)}


def spec(family):
    if family not in FAMILIES:raise ValueError('Unknown image model family')
    stem=f"drawai_{family}_paint35-main-v1_{'rows64' if family=='swin1d-image' else 'rgb224'}_fp32_v1"
    return dict(family=family,model_filename=stem+('.onnx' if family=='swin1d-image' else '.torchscript.pt'),
        runtime_format='onnx' if family=='swin1d-image' else 'torchscript',diagnostic_onnx_filename=stem+'.onnx',
        input_name='row_pixels' if family=='swin1d-image' else 'pixel_values',output_name='logits',
        input_shape=[1,64,64] if family=='swin1d-image' else [1,3,224,224],output_shape=[1,10],dtype='float32')


def verify_bundle(directory,expected_bundle_sha256):
    directory=Path(directory).resolve();record=load(directory/'bundle-manifest.json')
    if not isinstance(expected_bundle_sha256,str) or not re.fullmatch('[0-9a-f]{64}',expected_bundle_sha256):raise ValueError('External trusted bundle hash required')
    unsigned={k:v for k,v in record.items() if k!='bundle_sha256'}
    if record.get('bundle_sha256')!=expected_bundle_sha256 or sha(canonical(unsigned))!=expected_bundle_sha256:raise ValueError('Bundle anchor mismatch')
    if record.get('version')!='drawai-paint-onnx-bundle-v1' or record.get('trained') is not True or record.get('product_default_changed') is not False or record.get('lock_sha256')!=LOCK_HASH:
        raise ValueError('Invalid bundle metadata')
    actual={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    if actual!=set(record['files'])|{'bundle-manifest.json'}:raise ValueError('Unexpected/missing bundle file')
    for name,meta in record['files'].items():
        path=safe_file(directory,name)
        if path.stat().st_size!=meta['bytes'] or sha(path.read_bytes())!=meta['sha256']:raise ValueError('Bundle file hash mismatch: '+name)
    registry=load(directory/'registry.json')
    if set(registry['models'])!=set(FAMILIES):raise ValueError('Incomplete model registry')
    for family,entry in registry['models'].items():
        if entry['release_directory']!=family or entry['spec']!=spec(family):raise ValueError('Invalid registry entry')
        release=load(directory/family/'release.json');unsigned={k:v for k,v in release.items() if k!='release_sha256'}
        if release['release_sha256']!=sha(canonical(unsigned)) or entry['release_sha256']!=release['release_sha256']:raise ValueError('Release hash mismatch')
        if release['spec']!=spec(family) or release['trained'] is not True or release['artifact_verified'] is not True or release['release_ready'] is not False:
            raise ValueError('Invalid release/contract')
        if release['lock_sha256']!=LOCK_HASH or release['selection_sha256']!=record['selection_sha256']:raise ValueError('Release provenance mismatch')
        if load(directory/family/'preprocessing.json')!=paint.contract():raise ValueError('Preprocessing changed')
        if load(directory/family/'labels.json')!={'classes':[dict(id=i,name=str(i)) for i in range(10)]}:raise ValueError('Label order changed')
        expected={p.name for p in (directory/family).iterdir() if p.is_file()}-{'release.json'}
        if set(release['files'])!=expected:raise ValueError('Release inventory mismatch')
        for name,meta in release['files'].items():
            if record['files'][family+'/'+name]!=meta:raise ValueError('Release file inventory changed')
    return record,registry
