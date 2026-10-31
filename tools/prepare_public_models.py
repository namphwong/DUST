"""Package or securely fetch the pinned P6 bundle; no training/raw data is shipped."""
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import stat
import sys
import tempfile
import urllib.request
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.drawai.paint_package import verify_bundle

BUNDLE_HASH='86121e4def30d53f52b4e21126efceca3fff66b4d24a591c915c312a7db5eae2'
MANIFEST=ROOT/'deployment/model-release.json'
MAX_BYTES=210*1024*1024


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def extract_checked(archive,destination,expected):
    archive=Path(archive);destination=Path(destination).resolve()
    if digest(archive)!=expected:raise ValueError('Archive SHA-256 mismatch')
    if destination.exists():
        verify_bundle(destination,BUNDLE_HASH);return destination
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='drawai-model-',dir=destination.parent) as temporary:
        stage=Path(temporary)/'bundle';stage.mkdir()
        with zipfile.ZipFile(archive) as z:
            members=z.infolist();seen=set()
            if sum(i.file_size for i in members)>MAX_BYTES:raise ValueError('Archive expands beyond budget')
            for info in members:
                path=PurePosixPath(info.filename)
                if (path.is_absolute() or '..' in path.parts or '\\' in info.filename or ':' in info.filename or
                    info.is_dir() or stat.S_ISLNK(info.external_attr>>16) or info.filename.casefold() in seen):
                    raise ValueError('Unsafe/duplicate archive member')
                seen.add(info.filename.casefold());target=stage/str(path)
                if not target.resolve().is_relative_to(stage.resolve()):raise ValueError('Archive path traversal')
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(info) as source,target.open('wb') as output:
                    while block:=source.read(1024*1024):output.write(block)
        verify_bundle(stage,BUNDLE_HASH)
        os.replace(stage,destination)
    return destination


def package(source,archive):
    source=Path(source);archive=Path(archive);record,_=verify_bundle(source,BUNDLE_HASH)
    archive.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(list(record['files'])+['bundle-manifest.json']):
            info=zipfile.ZipInfo(name,date_time=(2026,10,19,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16;z.writestr(info,(source/name).read_bytes())
    return dict(filename=archive.name,sha256=digest(archive),bytes=archive.stat().st_size,bundle_sha256=BUNDLE_HASH)


def fetch(destination,archive=None):
    record=json.loads(MANIFEST.read_text(encoding='utf-8'));expected=record['archive']['sha256']
    if record['archive']['bundle_sha256']!=BUNDLE_HASH:raise ValueError('Wrong release anchor')
    if archive:return extract_checked(archive,destination,expected)
    destination=Path(destination).resolve()
    if destination.exists():verify_bundle(destination,BUNDLE_HASH);return destination
    with tempfile.TemporaryDirectory(prefix='drawai-fetch-') as directory:
        payload=Path(directory)/'bundle.zip'
        url=record['archive']['url']
        if not url.startswith('https://github.com/namphwong/DUST/releases/download/'):raise ValueError('Untrusted release URL')
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'DrawAI-model-fetch/1.0'}),timeout=120) as response,payload.open('wb') as output:
            size=0
            while block:=response.read(1024*1024):
                size+=len(block)
                if size>MAX_BYTES:raise ValueError('Download exceeds budget')
                output.write(block)
        if payload.stat().st_size!=record['archive']['bytes']:raise ValueError('Download length mismatch')
        return extract_checked(payload,destination,expected)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('package');p.add_argument('--source',type=Path,default=ROOT/'models/exported/paint35-v1');p.add_argument('--archive',type=Path,required=True)
    p=sub.add_parser('fetch');p.add_argument('--destination',type=Path,default=ROOT/'deployment/docker/model-bundle');p.add_argument('--archive',type=Path)
    args=parser.parse_args()
    if args.command=='package':print(json.dumps(package(args.source,args.archive)))
    else:print('Verified model bundle:',fetch(args.destination,args.archive))


if __name__=='__main__':main()
