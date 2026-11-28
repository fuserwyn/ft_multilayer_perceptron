"""The data layer: one class that reads the Wisconsin file and can split it."""

import numpy as np
import pandas as pd


class Dataset:
    """A table of cell-nucleus measurements and a diagnosis, M or B."""

    FEATURES = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean',
        'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
        'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
        'smoothness_worst', 'compactness_worst', 'concavity_worst',
        'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst',
    ]
    CLASSES = ['B', 'M']  # index 0 = benign, index 1 = malignant
    COLUMNS = ['id', 'diagnosis'] + FEATURES

    def __init__(self, frame):
        self.frame = frame

    @classmethod
    def read(cls, path):
        """Read a raw dataset file (no header, 32 columns)."""
        frame = pd.read_csv(path, header=None, names=cls.COLUMNS)
        bad = ~frame['diagnosis'].isin(cls.CLASSES)
        if bad.any():
            values = frame.loc[bad, 'diagnosis'].unique()
            raise ValueError(f"{path}: unexpected diagnosis values {values}")
        return cls(frame)

    @property
    def X(self):
        """Features, shape (m, 30)."""
        return self.frame[self.FEATURES].to_numpy(dtype=float)

    @property
    def y(self):
        """Integer labels, M -> 1, B -> 0."""
        return (self.frame['diagnosis'] == 'M').to_numpy(dtype=int)

    @staticmethod
    def encode(y, n_classes=2):
        """One-hot labels, shape (m, n_classes)."""
        out = np.zeros((len(y), n_classes))
        out[np.arange(len(y)), y] = 1.0
        return out

    def one_hot(self, n_classes=2):
        return self.encode(self.y, n_classes)

    def stratified_split(self, valid_ratio, seed):
        """Same M/B ratio in both parts. Returns (train, valid)."""
        rng = np.random.default_rng(seed)
        labels = self.frame['diagnosis'].to_numpy()
        train_idx, valid_idx = [], []
        for cls in np.unique(labels):
            idx = np.flatnonzero(labels == cls)
            rng.shuffle(idx)
            n_valid = int(round(len(idx) * valid_ratio))
            valid_idx.append(idx[:n_valid])
            train_idx.append(idx[n_valid:])
        train_idx = rng.permutation(np.concatenate(train_idx))
        valid_idx = rng.permutation(np.concatenate(valid_idx))
        return Dataset(self.frame.iloc[train_idx]), Dataset(self.frame.iloc[valid_idx])

    def save(self, path):
        self.frame.to_csv(path, header=False, index=False)

    def counts(self):
        return self.frame['diagnosis'].value_counts()
