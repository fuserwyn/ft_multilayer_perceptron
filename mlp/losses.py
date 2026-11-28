import numpy as np


class CategoricalCrossentropy:
    EPS = 1e-15

    def __call__(self, Y, P):
        P = np.clip(P, self.EPS, 1.0)
        return float(-np.mean(np.sum(Y * np.log(P), axis=1)))


class BinaryCrossentropy:
    EPS = 1e-15

    def __call__(self, y, p):
        p = np.clip(p, self.EPS, 1.0 - self.EPS)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


LOSSES = {
    'categoricalCrossentropy': CategoricalCrossentropy(),
}


class Metrics:
    def __init__(self, y, pred):
        y = np.asarray(y).astype(int)
        pred = np.asarray(pred).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1)))
        tn = int(np.sum((pred == 0) & (y == 0)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        self.values = {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1,
        }

    def __getitem__(self, key):
        return self.values[key]
