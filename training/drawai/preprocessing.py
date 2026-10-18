"""Versioned geometry contract mirrored by web/src/drawai/preprocessing.ts."""
import numpy as np
from numbers import Real

VERSION = 'drawai-geometry-v1'
MAX_STROKES = 16
MAX_POINTS = 8192


class DrawingError(ValueError):
    pass


def preprocess(strokes, length=64, y_axis='down'):
    """Return channel-major float32 [3,L], with no inter-stroke interpolation.

    Coordinates may be arbitrary finite drawing units. Timestamp is ignored.
    Singleton strokes are preserved as a single token. Consecutive duplicate
    points are removed, but the input object is never modified.
    """
    if length not in (64, 128):
        raise DrawingError('LENGTH: expected 64 or 128')
    if y_axis not in ('up', 'down'):
        raise DrawingError('AXIS: expected up or down')
    if not isinstance(strokes, (list, tuple)) or not strokes:
        raise DrawingError('EMPTY: draw a digit first')
    if len(strokes) > MAX_STROKES:
        raise DrawingError('STROKES: maximum 16 strokes')
    count = 0
    clean = []
    for stroke in strokes:
        if not isinstance(stroke, (list, tuple)) or not stroke:
            raise DrawingError('EMPTY_STROKE: stroke cannot be empty')
        count += len(stroke)
        if count > MAX_POINTS:
            raise DrawingError('POINTS: maximum 8192 points')
        coords = []
        for point in stroke:
            if not isinstance(point, dict) or any(
                not isinstance(point.get(key), Real) or isinstance(point.get(key), (bool, np.bool_))
                for key in ('x', 'y')
            ):
                raise DrawingError('NUMBER: x/y must be numbers, not strings/bools/containers')
            try:
                x, y = float(point['x']), float(point['y'])
            except (TypeError, KeyError, ValueError, OverflowError):
                raise DrawingError('NUMBER: x/y must be finite numbers') from None
            if not np.isfinite([x, y]).all() or max(abs(x), abs(y)) > 1e6:
                raise DrawingError('NUMBER: finite coordinates within +/-1e6 required')
            if isinstance(point['x'], (str, bool)) or isinstance(point['y'], (str, bool)):
                raise DrawingError('NUMBER: x/y must be numbers, not strings/bools')
            value = [x, -y if y_axis == 'up' else y]
            if not coords or value != coords[-1]:
                coords.append(value)
        clean.append(np.asarray(coords, dtype=np.float64))
    all_points = np.concatenate(clean)
    minimum, maximum = all_points.min(0), all_points.max(0)
    scale = float((maximum - minimum).max())
    if scale <= 0:
        raise DrawingError('TOO_SHORT: need a stroke with movement')
    center = (minimum + maximum) / 2
    clean = [(stroke - center) / scale for stroke in clean]
    distances = [np.concatenate(([0.], np.cumsum(np.hypot(*np.diff(s, axis=0).T)))) for s in clean]
    arc = np.asarray([d[-1] for d in distances])
    total = float(arc.sum())
    if total <= 1e-12:
        raise DrawingError('TOO_SHORT: isolated points are not a moving stroke')
    budgets = np.asarray([2 if len(s) > 1 else 1 for s in clean], dtype=np.int64)
    if budgets.sum() > length:
        raise DrawingError('BUDGET: not enough tokens to retain all strokes')
    remaining = length - int(budgets.sum())
    shares = remaining * arc / total
    extra = np.floor(shares).astype(np.int64)
    # Stable sort breaks ties by original stroke order, matching JavaScript.
    order = np.argsort(-(shares - extra), kind='stable')
    for index in order[:remaining - int(extra.sum())]:
        extra[index] += 1
    budgets += extra
    samples, starts = [], []
    for stroke, distance, budget in zip(clean, distances, budgets):
        if len(stroke) == 1:
            sampled = stroke.copy()
        else:
            target = np.linspace(0., distance[-1], int(budget))
            sampled = np.column_stack([np.interp(target, distance, stroke[:, c]) for c in range(2)])
        samples.append(sampled)
        starts.extend([1.] + [0.] * (len(sampled) - 1))
    values = np.column_stack([np.concatenate(samples), np.asarray(starts)]).T.astype(np.float32)
    if values.shape != (3, length) or not np.isfinite(values).all():
        raise DrawingError('INTERNAL: non-finite or incorrect tensor shape')
    return values


def contract(length=64):
    if length not in (64, 128):
        raise DrawingError('LENGTH: expected 64 or 128')
    return dict(version=VERSION, layout='NCT', dtype='float32', channels=['x', 'y', 'stroke_start'],
                length=length, y_axis='down', coordinate_range=[-0.5, 0.5],
                normalization='bbox-center/max-span; preserve aspect ratio',
                resampling='per-stroke arc length; minimum endpoints; largest remainder allocation',
                max_strokes=MAX_STROKES, max_points=MAX_POINTS, timestamps_used=False,
                padding=False, learned_normalization=False)
