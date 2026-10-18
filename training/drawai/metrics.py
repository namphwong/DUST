"""Ten-class metrics; rows=true, columns=predicted; absent-class F1=0."""
import numpy as np


def classification_metrics(truth, predicted):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if truth.ndim != 1 or truth.shape != predicted.shape or not len(truth):
        raise ValueError('Matching nonempty label vectors required')
    if not np.isin(truth, range(10)).all() or not np.isin(predicted, range(10)).all():
        raise ValueError('Labels must be integers 0..9')
    matrix = np.zeros((10, 10), np.int64)
    np.add.at(matrix, (truth.astype(np.int64), predicted.astype(np.int64)), 1)
    tp = matrix.diagonal().astype(float)
    support, counts = matrix.sum(1), matrix.sum(0)
    precision = np.divide(tp, counts, out=np.zeros(10), where=counts != 0)
    recall = np.divide(tp, support, out=np.zeros(10), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(10), where=precision + recall != 0)
    return dict(samples=len(truth), accuracy=float(tp.sum() / len(truth)), macro_f1=float(f1.mean()),
                confusion_matrix=matrix.tolist(),
                per_class=[dict(id=i, precision=float(precision[i]), recall=float(recall[i]),
                                f1=float(f1[i]), support=int(support[i])) for i in range(10)])
