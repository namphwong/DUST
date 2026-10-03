"""Independent NumPy reference for CSV timing/unit/window contract v1.

Fixtures are synthetic format tests, NOT labelled activities or new accuracy data.
"""
import csv
import io
import math
import re
import numpy as np

MAX_ROWS = 100_000


def parse_csv(text):
    if len(text.encode('utf-8')) > 10 * 1024 * 1024:
        raise ValueError('FILE_TOO_LARGE')
    rows = list(csv.reader(io.StringIO(text.lstrip('\ufeff')), strict=True))
    while rows and all(not cell.strip() for cell in rows[-1]):
        rows.pop()
    if len(rows) < 2:
        raise ValueError('EMPTY_FILE')
    if len(rows) - 1 > MAX_ROWS:
        raise ValueError('TOO_MANY_ROWS')
    headers = [cell.strip() for cell in rows[0]]
    if len(set(headers)) != len(headers):
        raise ValueError('DUPLICATE_COLUMN')
    required = ['timestamp_ms', 'acc_x', 'acc_y', 'acc_z']
    if any(name not in headers for name in required):
        raise ValueError('MISSING_COLUMN')
    indices = [headers.index(name) for name in required]
    values = []
    pattern = re.compile(r'^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$')
    for row in rows[1:]:
        if len(row) != len(headers):
            raise ValueError('INVALID_CSV')
        cells = [row[i].strip() for i in indices]
        if any(not pattern.fullmatch(cell) for cell in cells):
            raise ValueError('INVALID_NUMBER')
        numbers = [float(cell) for cell in cells]
        if not all(math.isfinite(v) for v in numbers):
            raise ValueError('INVALID_NUMBER')
        if numbers[0] < 0 or (values and numbers[0] <= values[-1][0]):
            raise ValueError('INVALID_TIMESTAMP')
        values.append(numbers)
    return np.asarray(values, dtype=np.float64)


def prepare_csv(values, unit='g', gravity=True, demo=False):
    if not gravity:
        raise ValueError('INCOMPATIBLE_SIGNAL')
    if unit not in ('g', 'ms2'):
        raise ValueError('INVALID_UNIT')
    if demo and (unit != 'g' or len(values) != 128 or not np.array_equal(values[:,0],np.arange(128)*20)):
        raise ValueError('INVALID_TIMESTAMP')
    values = values.copy()
    values[:,0] -= values[0,0]
    boundaries = np.flatnonzero(np.diff(values[:, 0]) > 100) + 1
    groups = np.split(values, boundaries)
    if demo and (unit != 'g' or len(groups) != 1 or len(values) != 128 or not np.array_equal(values[:,0], np.arange(128)*20)):
        raise ValueError('INVALID_TIMESTAMP')
    info = dict(sourceRows=len(values), resampledRows=0, segmentCount=len(groups),
                interpolatedPoints=0, droppedSegments=0, droppedPoints=0,
                tailPoints=0, windowCount=0, sourceUnit=unit)
    result = []
    for segment_id, group in enumerate(groups):
        count = int(math.floor((group[-1,0]-group[0,0])/20)) + 1
        if count + info['resampledRows'] > MAX_ROWS:
            raise ValueError('TOO_MANY_ROWS')
        info['resampledRows'] += count
        if count < 128:
            info['droppedSegments'] += 1
            info['droppedPoints'] += count
            continue
        median = float(np.median(np.diff(group[:,0])))
        if not 10 <= median <= 25:
            raise ValueError('UNSUPPORTED_FREQUENCY')
        grid = group[0,0] + np.arange(count)*20
        signals = np.stack([np.interp(grid, group[:,0], group[:,channel]) for channel in (1,2,3)], axis=1)
        signals /= 9.80665 if unit == 'ms2' else 1
        info['interpolatedPoints'] += int(np.count_nonzero(~np.isin(grid, group[:,0])))
        times = grid
        count_windows = (count-128)//64 + 1
        info['windowCount'] += count_windows
        info['tailPoints'] += count-((count_windows-1)*64+128)
        for start in range(0, count-127, 64):
            result.append(dict(segmentId=segment_id, windowIndex=len(result),
                               startMs=float(times[start]), endMs=float(times[start+127]),
                               raw=signals[start:start+128].T.copy()))
    if not result:
        raise ValueError('TOO_SHORT')
    return info, result


def normalize(raw, prep):
    mean = np.asarray(prep['mean'], dtype=np.float32)[:, None]
    std = np.maximum(np.asarray(prep['std'], dtype=np.float32), np.float32(1e-8))[:, None]
    return (np.asarray(raw, dtype=np.float32)-mean)/std
