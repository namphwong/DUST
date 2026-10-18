"""python -m training.drawai.prepare_data [--download] [--length 64|128]."""
import argparse
import hashlib
import json
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from unlzw3 import unlzw

from .preprocessing import VERSION, contract, preprocess

ROOT = Path(__file__).resolve().parents[2]
URL = 'https://archive.ics.uci.edu/static/public/81/pen%2Bbased%2Brecognition%2Bof%2Bhandwritten%2Bdigits.zip'
SOURCE_PAGE = 'https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_unipen(text, source):
    """Strict parser for the actual PenDigits subset of UNIPEN, not generic UNIPEN.

    .INCLUDE is preserved as metadata only: never resolve or execute a file.
    Segment indexes refer to global strokes, not digit/sample indexes.
    .COMMENT fields are retained without asserting undocumented writer identity.
    """
    samples, current, active = [], None, None
    stroke_index = 0
    allowed_header = ('.INCLUDE', '.LEXICON', '.HIERARCHY')

    def fail(line, message):
        raise ValueError(f'{source}:{line}: {message}')

    def finish(line):
        if current is None:
            return
        if active is not None:
            fail(line, 'unclosed PEN_DOWN')
        if not current['strokes'] or current['comment'] is None:
            fail(line, 'segment missing strokes/comment')
        start, end = current['segment_range']
        if end - start + 1 != len(current['strokes']):
            fail(line, 'segment stroke range disagrees with actual strokes')
        samples.append(current)

    lines = text.splitlines()
    for line_no, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith('.SEGMENT'):
            finish(line_no)
            match = re.fullmatch(r'\.SEGMENT\s+DIGIT\s+(\d+)(?:-(\d+))?\s+\?\s+"([0-9])"', line)
            if not match:
                fail(line_no, 'invalid digit segment')
            start, end = int(match[1]), int(match[2] or match[1])
            if start != stroke_index or end < start:
                fail(line_no, 'non-contiguous or reversed global stroke range')
            current = dict(id=f'{source}:{len(samples):05d}', source=source,
                           source_line=line_no, source_index=len(samples),
                           segment_range=[start, end], label=int(match[3]),
                           comment=None, writer_id=None, strokes=[])
        elif line.split()[0] in allowed_header:
            if current is not None:
                fail(line_no, 'header inside sample')
        elif line.startswith('.COMMENT'):
            if current is None or current['comment'] is not None or active is not None:
                fail(line_no, 'unexpected comment')
            tokens = line.split()[1:]
            if len(tokens) != 3 or not all(token.isdigit() for token in tokens):
                fail(line_no, 'unexpected PenDigits comment format')
            current['comment'] = list(map(int, tokens))
            if current['comment'][0] != current['label']:
                fail(line_no, 'comment/segment labels disagree')
        elif line == '.PEN_DOWN':
            if current is None or active is not None:
                fail(line_no, 'unexpected PEN_DOWN')
            active = []
        elif line == '.PEN_UP':
            if active is None or not active:
                fail(line_no, 'PEN_UP without nonempty stroke')
            current['strokes'].append(active)
            active = None
            stroke_index += 1
        elif line.startswith('.DT'):
            if current is None or active is not None or line != '.DT 100':
                fail(line_no, 'unexpected DT; this parser only supports PenDigits DT 100')
        elif line.startswith('.'):
            fail(line_no, 'unsupported directive')
        else:
            tokens = line.split()
            if active is None or len(tokens) != 2:
                fail(line_no, 'coordinates outside stroke or incorrect columns')
            try:
                x, y = map(int, tokens)
            except ValueError:
                fail(line_no, 'coordinates must be integers')
            if min(x, y) < 0 or max(x, y) > 500:
                fail(line_no, 'coordinate outside documented 0..500 range')
            active.append(dict(x=x, y=y))
    finish(len(lines) + 1)
    if not samples:
        fail(1, 'no samples')
    return samples


def split_development(samples, seed=42, fraction=0.2):
    """Documented fallback: stratified sample split, writer identity unverified."""
    if not 0 < fraction < 1:
        raise ValueError('validation fraction must be strictly between 0 and 1')
    rng = np.random.default_rng(seed)
    labels = np.asarray([s['label'] for s in samples])
    if set(labels.tolist()) != set(range(10)):
        raise ValueError('development pool must contain exactly labels 0..9')
    train, val = [], []
    for label in range(10):
        ids = np.flatnonzero(labels == label)
        if len(ids) < 2:
            raise ValueError(f'class {label} needs at least two samples for train/validation')
        rng.shuffle(ids)
        size = min(len(ids) - 1, max(1, int(np.floor(len(ids) * fraction))))
        val.extend(ids[:size].tolist())
        train.extend(ids[size:].tolist())
    return sorted(train), sorted(val)


def trajectory_hash(sample):
    return hashlib.sha256(json.dumps(sample['strokes'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def stats(samples):
    points = [sum(map(len, s['strokes'])) for s in samples]
    stroke_counts = [len(s['strokes']) for s in samples]
    return dict(samples=len(samples), labels={str(k): v for k, v in sorted(Counter(s['label'] for s in samples).items())},
                points=dict(min=min(points), max=max(points), median=float(np.median(points))),
                strokes=dict(min=min(stroke_counts), max=max(stroke_counts),
                             distribution=dict(sorted(Counter(map(str, stroke_counts)).items()))),
                consecutive_duplicate_points=sum(sum(a == b for a, b in zip(stroke, stroke[1:])) for s in samples for stroke in s['strokes']),
                singleton_strokes=sum(len(stroke) == 1 for s in samples for stroke in s['strokes']),
                comment_field_2_values=sorted({s['comment'][1] for s in samples}),
                points_per_class={str(label): dict(min=min(p), max=max(p), median=float(np.median(p)))
                                 for label in range(10)
                                 for p in [[sum(map(len, s['strokes'])) for s in samples if s['label'] == label]]})


def plot_samples(train, target, length=64):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for label, ax in enumerate(axes.flat):
        sample = next(s for s in train if s['label'] == label)
        for stroke in sample['strokes']:
            x, y = [p['x'] for p in stroke], [-p['y'] for p in stroke]
            ax.plot(x, y, linewidth=2)
            ax.scatter(x[0], y[0], marker='o', s=24)
            ax.annotate('start', (x[0], y[0]), fontsize=7)
            if len(stroke) > 1:
                end = min(len(stroke) - 1, max(1, len(stroke) // 4))
                ax.annotate('', xy=(x[end], y[end]), xytext=(x[0], y[0]),
                            arrowprops=dict(arrowstyle='->', color='black'))
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title(f"digit {label} / sample {sample['source_index']}")
        ax.grid(alpha=0.2)
    fig.suptitle('PenDigits original / y converted to canvas (down) / fixed first train sample per class')
    fig.tight_layout()
    fig.savefig(target, dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for label, ax in enumerate(axes.flat):
        sample = next(s for s in train if s['label'] == label)
        value = preprocess(sample['strokes'], length, 'up')
        starts = np.flatnonzero(value[2]).tolist() + [length]
        for begin, end in zip(starts, starts[1:]):
            ax.plot(value[0, begin:end], value[1, begin:end], '.-', markersize=2, linewidth=1)
            ax.scatter(value[0, begin], value[1, begin], marker='o', s=24)
        ax.set(xlim=(-0.6,0.6), ylim=(0.6,-0.6), title=f'digit {label} / {length} tokens')
        ax.set_aspect('equal')
        ax.grid(alpha=0.2)
    fig.suptitle('Normalized per-stroke resampling / circles mark stroke starts / no pen-up connectors')
    fig.tight_layout()
    fig.savefig(target.with_name(target.stem + '-resampled.png'), dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--length', type=int, default=64, choices=[64, 128])
    args = parser.parse_args()
    raw = ROOT / 'data/raw/pendigits'
    raw.mkdir(parents=True, exist_ok=True)
    if args.download:
        urllib.request.urlretrieve(URL, raw / 'source.zip')
        with zipfile.ZipFile(raw / 'source.zip') as archive:
            # Extract only known data/document files, never arbitrary archive paths.
            for name in ['pendigits-orig.tra.Z', 'pendigits-orig.tes.Z', 'pendigits-orig.names', 'pendigits.names']:
                (raw / name).write_bytes(archive.read(name))
    sources = {}
    parsed = {}
    for split, name in [('development', 'pendigits-orig.tra'), ('test', 'pendigits-orig.tes')]:
        compressed = raw / (name + '.Z')
        content = unlzw(compressed.read_bytes())
        (raw / name).write_bytes(content)
        parsed[split] = parse_unipen(content.decode('ascii'), name)
        sources[name] = dict(compressed_sha256=digest(compressed), uncompressed_sha256=digest(raw / name),
                             compressed_bytes=compressed.stat().st_size)
    for split, count in [('development', 7494), ('test', 3498)]:
        if len(parsed[split]) != count:
            raise ValueError(f'{split} count differs from source documentation')
    train_ids, val_ids = split_development(parsed['development'])
    groups = dict(train=[parsed['development'][i] for i in train_ids],
                  validation=[parsed['development'][i] for i in val_ids], test=parsed['test'])
    target = ROOT / f'data/processed/pendigits-drawai-v1/l{args.length}'
    target.mkdir(parents=True, exist_ok=True)
    manifest = dict(version=VERSION, seed=42, validation_fraction=0.2, source_url=URL,
                    split_method='stratified-sample; writer IDs unverified',
                    validation_rounding='floor per class; minimum 1; leave at least 1 train sample',
                    writer_id_verified=False, official_test_unchanged=True, sources=sources,
                    archive_sha256=digest(raw/'source.zip') if (raw/'source.zip').exists() else None,
                    source_documents={n:digest(raw/n) for n in ['pendigits-orig.names','pendigits.names']}, splits={})
    invalid = []
    tensors = {}
    for split, samples in groups.items():
        rows = []
        for sample in samples:
            try:
                rows.append(preprocess(sample['strokes'], args.length, 'up'))
            except ValueError as error:
                invalid.append(dict(id=sample['id'], reason=str(error)))
        if invalid:
            write_json(ROOT / 'artifacts/drawai/data/invalid-samples.json', invalid)
            raise ValueError('Invalid samples: see invalid-samples.json; no samples silently discarded')
        X = np.stack(rows)
        y = np.asarray([s['label'] for s in samples], dtype=np.int64)
        ids = np.asarray([s['id'] for s in samples])
        # Hash logical tensors, not ZIP container timestamps.
        data_hash = hashlib.sha256(X.tobytes() + y.tobytes() + '\n'.join(ids).encode()).hexdigest()
        np.savez_compressed(target / f'{split}.npz', X=X, y=y, sample_ids=ids)
        tensors[split] = X
        manifest['splits'][split] = dict(samples=len(samples), tensor_sha256=data_hash,
                                        entries=[dict(id=s['id'], label=s['label'], source_line=s['source_line'],
                                                      source=s['source'], source_index=s['source_index'],
                                                      segment_range=s['segment_range'],
                                                      trajectory_sha256=trajectory_hash(s),
                                                      writer_id=None, comment=s['comment']) for s in samples])
    sets = {split: {s['id'] for s in samples} for split, samples in groups.items()}
    assert not sets['train'] & sets['validation']
    assert sets['train'] | sets['validation'] == {s['id'] for s in parsed['development']}
    assert not (sets['train'] | sets['validation']) & sets['test']
    hashes = {split: Counter(trajectory_hash(s) for s in samples) for split, samples in groups.items()}
    overlaps = {f'{a}-{b}': len(set(hashes[a]) & set(hashes[b])) for a, b in
                [('train', 'validation'), ('train', 'test'), ('validation', 'test')]}
    report = dict(status='passed', length=args.length, source=SOURCE_PAGE, sources=sources,
                  archive_sha256=manifest['archive_sha256'],source_documents=manifest['source_documents'],
                  preprocessing=contract(args.length), writer_id_verified=False,
                  writer_limitation='COMMENT field meanings undocumented; validation may share writers with train.',
                  coordinate_axis_evidence='Digit shapes inspected after converting source y-up to canvas y-down.',
                  stats={split: stats(samples) for split, samples in groups.items()},
                  development_stats=stats(parsed['development']),
                  selected_plot_samples=[dict(label=label, sample_id=s['id'],
                                             source_line=s['source_line'],
                                             selection='First train sample of class in source order')
                                         for label in range(10)
                                         for s in [next(s for s in groups['train'] if s['label'] == label)]],
                  duplicate_trajectories_within={s: sum(n-1 for n in c.values()) for s, c in hashes.items()},
                  duplicate_trajectories_across=overlaps, invalid_samples=invalid,
                  tensor_hashes={s: v['tensor_sha256'] for s, v in manifest['splits'].items()})
    write_json(target / 'manifest.json', manifest)
    write_json(target / 'preprocessing.json', contract(args.length))
    write_json(target / 'labels.json', dict(classes=[dict(id=i, name=str(i)) for i in range(10)]))
    fixtures = []
    for label in range(10):
        i = next(i for i, s in enumerate(groups['train']) if s['label'] == label)
        sample = groups['train'][i]
        fixtures.append(dict(name=f'train-digit-{label}', sample_id=sample['id'], label=label,
                             y_axis='up', strokes=sample['strokes'], input=tensors['train'][i].reshape(-1).tolist()))
    write_json(target / 'fixtures.json', dict(version=VERSION, length=args.length, cases=fixtures))
    plot_path = ROOT / f'artifacts/drawai/data/l{args.length}-samples.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_samples(groups['train'], plot_path, args.length)
    write_json(ROOT / f'artifacts/drawai/data/l{args.length}-report.json', report)
    print(json.dumps(dict(status=report['status'], length=args.length,
                         samples={s: len(v) for s, v in groups.items()}, overlaps=overlaps,
                         max_points=max(v['points']['max'] for v in report['stats'].values()),
                         max_strokes=max(v['strokes']['max'] for v in report['stats'].values())), indent=2))


if __name__ == '__main__':
    main()
