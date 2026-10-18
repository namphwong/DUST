"""P4 immutable Paint35 image snapshot; integrity sealing is not access control.

Near-similarity flags are retained and disclosed, never certified as independent
drawings by this tool. Owner provenance is an attestation, not identity proof.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import io
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from . import paint_data as paint
from .private_data import ROOT, canonical, sha, load, dump
from .private_dataset import separate_output, safe_file, zip_write, stamp
from .raster import normalize_rgb

DATASET = 'paint35-main-v1'
DECISION = 'retain_flagged_similarity_unresolved'


def manifest(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != paint.MANIFEST_FIELDS:
            raise ValueError('Manifest schema mismatch')
        return list(reader)


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=paint.MANIFEST_FIELDS)
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode()


def check_image(root, row):
    person, label, rep = (int(row[k]) for k in ('person_id', 'label', 'repetition'))
    sid = f'paint35_p{person:03}_d{label}_r{rep:02}'
    if not 0 <= label <= 9 or row['sample_id'] != sid or row['png64'] != f'images64/{sid}.png':
        raise ValueError('Image ID/path mismatch')
    if row['split'] != paint.split_for(person, rep):
        raise ValueError('Writer/split mismatch')
    path = safe_file(Path(root).resolve(), row['png64'])
    if sha(path.read_bytes()) != row['png_sha256']:
        raise ValueError('PNG hash mismatch')
    with Image.open(path) as image:
        if image.mode != 'L' or image.size != (64, 64):
            raise ValueError('Invalid PNG64')
        pixels = np.asarray(image).copy()
    rgb = paint.dino_rgb(pixels)
    hashes = dict(image_sha256=sha(pixels.tobytes()),
                  swin_rows_sha256=sha(paint.swin_rows(pixels).tobytes()),
                  dinov3_rgb_sha256=sha(rgb.tobytes()),
                  dinov3_tensor_sha256=sha(normalize_rgb(rgb).tobytes()))
    if any(row[k] != v for k, v in hashes.items()):
        raise ValueError('Pixel/preprocessing hash mismatch')
    return pixels


def source_state(prepared, source_path):
    prepared = Path(prepared).resolve()
    arrays, source = paint.read_source(source_path)
    provenance = load(prepared/'provenance.json')
    report = paint.audit_arrays(arrays, source, provenance)
    if report['errors'] or not report['source_quota_complete']:
        raise ValueError('Full valid Paint35 source required')
    if load(prepared/'plan.json') != paint.policy() or load(prepared/'preprocessing.json') != paint.contract():
        raise ValueError('Plan/preprocessing changed')
    if load(prepared/'audit.json') != report:
        raise ValueError('Stale prepared audit')
    rows = manifest(prepared/'manifest.csv')
    meta = load(prepared/'dataset.json')
    if len(rows) != 7000 or meta['manifest_sha256'] != sha((prepared/'manifest.csv').read_bytes()):
        raise ValueError('Prepared manifest changed')
    seen = set()
    for row in rows:
        index = int(row['source_index'])
        if index in seen or not 0 <= index < 7000:
            raise ValueError('Repeated/invalid source index')
        seen.add(index)
        for key, array in [('person_id','person_ids'), ('label','labels'), ('repetition','sample_ids')]:
            if int(row[key]) != int(arrays[array][index]):
                raise ValueError('Source/manifest metadata mismatch')
        if not np.array_equal(check_image(prepared, row), arrays['images'][index]):
            raise ValueError('Source/prepared pixels differ')
    bindings = dict(source_sha256=source['sha256'], plan_sha256=paint.policy()['plan_sha256'],
                    manifest_sha256=sha((prepared/'manifest.csv').read_bytes()),
                    audit_sha256=sha(canonical(report)), provenance_sha256=sha(canonical(provenance)),
                    preprocessing_sha256=sha(canonical(paint.contract())))
    return bindings, report, rows, arrays


def review(prepared, source, output):
    output = separate_output(output, prepared)
    bindings, report, rows, arrays = source_state(prepared, source)
    indexed = {int(r['source_index']):r for r in rows}
    groups = []
    for group in report['near_duplicate_groups']:
        indices = group['source_indices']
        # Full source-pixel comparison, not just descriptor/dHash equality.
        differences = [int(np.count_nonzero(arrays['images'][a] != arrays['images'][b]))
                       for j,a in enumerate(indices) for b in indices[j+1:]]
        groups.append(dict(**group, samples=[indexed[i]['sample_id'] for i in indices],
                           min_pair_different_pixels=min(differences), max_pair_different_pixels=max(differences),
                           decision=DECISION,
                           reason='No exact pixel copies; retain all source samples under fixed writer split. '
                                  'ROI similarity may reflect copying or common shape; independence not verified. '
                                  'No model predictions used, no exclusions, no reassignment.'))
    output.mkdir(parents=True)
    (output/'cross-split-sheets').mkdir()
    cross = [g for g in groups if g['cross_split']]
    sheets = []
    for start in range(0, len(cross), 20):
        chunk = cross[start:start+20]
        board = Image.new('RGB', (1180, len(chunk)*94+25), 'white'); draw = ImageDraw.Draw(board)
        draw.text((5,3), 'Paint35 near groups: source integrity QC ONLY; no model predictions', fill='black')
        for j,g in enumerate(chunk):
            y = 25+j*94
            draw.text((2,y), f"{start+j+1} {g['group_id'][5:15]}", fill='black')
            for n,index in enumerate(g['source_indices']):
                row = indexed[index]; x = 125+n*80
                board.paste(Image.fromarray(arrays['images'][index]).convert('RGB'), (x,y+15))
                draw.text((x,y), f"p{row['person_id']} {row['split'][:3]}", fill='black')
        name = f'cross-split-sheets/page-{start//20+1:02}.png'
        board.save(output/name)
        sheets.append(dict(path=name, sha256=sha((output/name).read_bytes()), group_ids=[g['group_id'] for g in chunk]))
    value = dict(version='drawai-paint-review-v1', bindings=bindings,
                 reviewed_at=datetime.now(timezone.utc).isoformat(),
                 reviewer='Codex automated source-pixel audit; owner provenance confirmed in project chat',
                 review_kind='Integrity audit and explicit retention policy; not a human keep-distinct certification',
                 near_groups=groups, exclusions=[], model_based_selection=False,
                 similarity_independence_verified=False, source_already_public=True,
                 cross_split_contact_sheets=sheets)
    dump(output/'review.json', value)
    return value


def validate_review(value, bindings, report):
    if value.get('version') != 'drawai-paint-review-v1' or value.get('bindings') != bindings:
        raise ValueError('Stale/wrong review bindings')
    stamp(value['reviewed_at'])
    if value.get('exclusions') != [] or value.get('model_based_selection') is not False:
        raise ValueError('No exclusions/model-based selection permitted in this snapshot')
    if value.get('similarity_independence_verified') is not False or value.get('source_already_public') is not True:
        raise ValueError('Review must disclose unresolved similarity and public source')
    expected = {g['group_id']:g for g in report['near_duplicate_groups']}
    actual = value.get('near_groups', [])
    if len(actual) != len(expected) or {g['group_id'] for g in actual} != set(expected):
        raise ValueError('Missing/duplicate near-group decisions')
    for g in actual:
        original = expected[g['group_id']]
        if any(g[k] != original[k] for k in ('source_indices','cross_split')):
            raise ValueError('Changed near-group membership')
        if g['decision'] != DECISION or g['min_pair_different_pixels'] <= 0 or len(g['reason']) < 40:
            raise ValueError('Invalid retention decision')


def quotas(rows, splits):
    counts = Counter(r['split'] for r in rows)
    expected = {s:paint.policy()[s]['samples'] for s in splits}
    if dict(counts) != expected or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Sample/split quota mismatch')
    actual = Counter((r['split'],int(r['person_id']),int(r['label']),int(r['repetition'])) for r in rows)
    desired = Counter((s,p,y,r) for s in splits for p in paint.policy()[s]['person_ids']
                      for y in range(10) for r in range(1,21))
    if actual != desired:
        raise ValueError('Writer/class/repetition quota mismatch')


def lock(prepared, source, review_path, output):
    output = separate_output(output, prepared)
    bindings, report, rows, _ = source_state(prepared, source)
    review_bytes = Path(review_path).read_bytes(); value = load(review_path)
    validate_review(value, bindings, report)
    main = [r for r in rows if r['split'] in ('train','validation','test')]
    quotas(main, ('train','validation','test'))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name+'.staging-', dir=output.parent))
    development = [r for r in main if r['split'] != 'test']
    heldout = [r for r in main if r['split'] == 'test']
    for row in development:
        path = staging/'development'/row['png64']; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(safe_file(Path(prepared).resolve(), row['png64']).read_bytes())
    (staging/'development/manifest.csv').write_bytes(csv_bytes(development))
    with zipfile.ZipFile(staging/'heldout-test.zip','w') as archive:
        for row in heldout:
            zip_write(archive, row['png64'], safe_file(Path(prepared).resolve(),row['png64']).read_bytes())
        zip_write(archive,'manifest.csv',csv_bytes(heldout))
        zip_write(archive,'CUSTODY.txt',b'Final evaluation only. Hash sealed, NOT encrypted. Original source is already public. No test-based tuning.\n')
    dump(staging/'plan.json',paint.policy()); dump(staging/'preprocessing.json',paint.contract())
    dump(staging/'provenance.json',load(Path(prepared)/'provenance.json'))
    # Do not expose held-out per-sample labels/IDs through the root review file.
    dump(staging/'review-summary.json',dict(bindings=bindings,review_sha256=sha(review_bytes),
         near_groups=len(value['near_groups']),cross_split_groups=sum(g['cross_split'] for g in value['near_groups']),
         decision=DECISION,exclusions=0,similarity_independence_verified=False))
    files = {p.relative_to(staging).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes()))
             for p in sorted(staging.rglob('*')) if p.is_file()}
    record = dict(version='drawai-paint-lock-v1',dataset=DATASET,dataset_locked=True,
                  bindings=bindings,review_sha256=sha(review_bytes),split_counts={'train':4000,'validation':1000,'test':1000},
                  sample_count=6000,pilot_samples_excluded=1000,files=files,
                  created_at=datetime.now(timezone.utc).isoformat(),test_used_for_tuning=False,
                  source_already_public=True,similarity_independence_verified=False,
                  test_custody='ZIP integrity only; no encryption/ACL; public original prevents secret holdout',
                  training_started=False,release_ready=False,
                  code_sha256={n:sha((ROOT/n).read_bytes()) for n in ['training/drawai/paint_dataset.py','training/drawai/paint_data.py','training/drawai/raster.py']})
    record['lock_sha256'] = sha(canonical(record)); dump(staging/'dataset-lock.json',record)
    # Detect concurrent edits before promotion; never replace an existing lock.
    again, _, _, _ = source_state(prepared,source)
    if again != bindings or Path(review_path).read_bytes() != review_bytes:
        raise ValueError('Input changed during lock; staging is not final')
    verify_lock(staging, record['lock_sha256'])
    staging.rename(output)
    return record


def verify_lock(root, expected_lock_sha256):
    root = Path(root).resolve(); record = load(root/'dataset-lock.json')
    unsigned = {k:v for k,v in record.items() if k != 'lock_sha256'}
    if not isinstance(expected_lock_sha256,str) or len(expected_lock_sha256) != 64 or record.get('lock_sha256') != expected_lock_sha256 or sha(canonical(unsigned)) != expected_lock_sha256:
        raise ValueError('External trusted lock hash mismatch')
    if record.get('version') != 'drawai-paint-lock-v1' or record.get('dataset') != DATASET or record.get('dataset_locked') is not True:
        raise ValueError('Invalid Paint lock')
    if record['bindings']['source_sha256'] != paint.SOURCE_SHA256 or record['bindings']['plan_sha256'] != paint.policy()['plan_sha256']:
        raise ValueError('Source/plan binding mismatch')
    if record['split_counts'] != {'train':4000,'validation':1000,'test':1000} or record['sample_count'] != 6000 or record['pilot_samples_excluded'] != 1000:
        raise ValueError('Invalid main quota')
    if record['test_used_for_tuning'] is not False or record['similarity_independence_verified'] is not False or record['source_already_public'] is not True:
        raise ValueError('Invalid test/similarity declaration')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(record['files']) | {'dataset-lock.json'}:
        raise ValueError('Unexpected/missing package files')
    for name,meta in record['files'].items():
        path = safe_file(root,name)
        if path.stat().st_size != meta['bytes'] or sha(path.read_bytes()) != meta['sha256']:
            raise ValueError('Package hash mismatch: '+name)
    if load(root/'plan.json') != paint.policy() or load(root/'preprocessing.json') != paint.contract():
        raise ValueError('Packaged protocol changed')
    paint.check_provenance(load(root/'provenance.json'),paint.SOURCE_SHA256)
    rows = manifest(root/'development/manifest.csv'); quotas(rows,('train','validation'))
    for row in rows:
        check_image(root/'development',row)
    # Hash the test ZIP without opening labels or extracting it.
    return record, rows


def iter_development(root, split, expected_lock_sha256, representation='swin-rows64'):
    if split not in ('train','validation'):
        raise ValueError('Development accepts train/validation only; test/pilot rejected')
    if representation not in ('swin-rows64','dinov3-rgb224'):
        raise ValueError('Unsupported image representation')
    _, rows = verify_lock(root,expected_lock_sha256)
    for row in rows:
        if row['split'] != split:continue
        pixels = check_image(Path(root)/'development',row)
        value = paint.swin_rows(pixels) if representation == 'swin-rows64' else normalize_rgb(paint.dino_rgb(pixels))
        yield value,int(row['label']),row


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    for command in ('review','lock'):
        p = sub.add_parser(command)
        p.add_argument('--prepared',type=Path,required=True); p.add_argument('--source',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True)
        if command == 'lock':p.add_argument('--review',type=Path,required=True)
    p = sub.add_parser('verify'); p.add_argument('--dataset',type=Path,required=True); p.add_argument('--expected-lock-sha256',required=True)
    args = parser.parse_args()
    if args.command == 'review':
        result = review(args.prepared,args.source,args.output)
        print(f"Reviewed {len(result['near_groups'])} groups; retained with unresolved similarity disclosure.")
    elif args.command == 'lock':
        result = lock(args.prepared,args.source,args.review,args.output); print(result['lock_sha256'])
    else:
        result,_ = verify_lock(args.dataset,args.expected_lock_sha256); print('Verified '+result['lock_sha256'])


if __name__ == '__main__':main()
