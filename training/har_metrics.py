import numpy as np


def classification_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 1 or not len(y_true):
        raise ValueError('Expected matching non-empty one-dimensional labels')
    if not np.isin(y_true, range(6)).all() or not np.isin(y_pred, range(6)).all():
        raise ValueError('Labels must be 0..5')
    matrix = np.zeros((6, 6), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    tp = matrix.diagonal().astype(float)
    support = matrix.sum(axis=1)
    precision = np.divide(tp, matrix.sum(axis=0), out=np.zeros(6), where=matrix.sum(axis=0) != 0)
    recall = np.divide(tp, support, out=np.zeros(6), where=support != 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(6), where=precision+recall != 0)
    return dict(accuracy=float(tp.sum()/len(y_true)), macro_f1=float(f1.mean()),
                per_class=[dict(id=i, precision=float(precision[i]), recall=float(recall[i]),
                                f1=float(f1[i]), support=int(support[i])) for i in range(6)],
                confusion_matrix=matrix.tolist(), samples=len(y_true))
