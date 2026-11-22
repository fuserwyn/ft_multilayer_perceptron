"""
Core of the multilayer perceptron, written from scratch with numpy.

Conventions
-----------
- A batch X has shape (m, n_features): one example per row.
- Layer l has weights W[l] of shape (n_in, n_out) and bias b[l] of shape (1, n_out),
  so the weighted sum is  Z = A_prev @ W + b.
- The output layer always uses softmax; labels are one-hot encoded (m, n_classes).
"""

import json

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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
COLUMNS = ['id', 'diagnosis'] + FEATURES
CLASSES = ['B', 'M']  # index 0 = benign, index 1 = malignant


def read_csv(path):
    """Read a raw dataset file (no header, 32 columns)."""
    df = pd.read_csv(path, header=None, names=COLUMNS)
    bad = ~df['diagnosis'].isin(CLASSES)
    if bad.any():
        raise ValueError(f"{path}: unexpected diagnosis values {df.loc[bad, 'diagnosis'].unique()}")
    return df


def load_xy(path):
    """Return features X (m, 30) and integer labels y (m,), with M -> 1, B -> 0."""
    df = read_csv(path)
    X = df[FEATURES].to_numpy(dtype=float)
    y = (df['diagnosis'] == 'M').to_numpy(dtype=int)
    return X, y


def one_hot(y, n_classes=2):
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


# ---------------------------------------------------------------------------
# Activations: f(z) and f'(z)
# ---------------------------------------------------------------------------

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def relu(z):
    return np.maximum(0.0, z)


def relu_prime(z):
    return (z > 0).astype(float)


def tanh(z):
    return np.tanh(z)


def tanh_prime(z):
    return 1.0 - np.tanh(z) ** 2


ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_prime),
    'relu': (relu, relu_prime),
    'tanh': (tanh, tanh_prime),
}


def softmax(z):
    """Row-wise softmax. Subtracting the row max avoids overflow in exp."""
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

EPS = 1e-15


def categorical_crossentropy(Y, P):
    """Mean over the batch of  -sum_k Y_k log P_k."""
    return float(-np.mean(np.sum(Y * np.log(np.clip(P, EPS, 1.0)), axis=1)))


def binary_crossentropy(y, p):
    """E = -1/N sum[ y log p + (1-y) log(1-p) ], y in {0,1}, p = P(class 1)."""
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


LOSSES = {
    'categoricalCrossentropy': categorical_crossentropy,
}


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def init_weights(name, n_in, n_out, rng):
    if name == 'heUniform':          # good default for relu, works for sigmoid too
        limit = np.sqrt(6.0 / n_in)
        return rng.uniform(-limit, limit, (n_in, n_out))
    if name == 'heNormal':
        return rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, n_out))
    if name == 'xavierUniform':      # a.k.a. Glorot, designed for sigmoid / tanh
        limit = np.sqrt(6.0 / (n_in + n_out))
        return rng.uniform(-limit, limit, (n_in, n_out))
    raise ValueError(f"unknown weights initializer '{name}'")


WEIGHT_INITS = ['heUniform', 'heNormal', 'xavierUniform']


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP:
    """Fully connected network: hidden layers use `activation`, output uses softmax."""

    def __init__(self, layer_sizes, activation='sigmoid', weights_init='heUniform', seed=42):
        if len(layer_sizes) < 3:
            raise ValueError("need at least an input, one hidden and an output layer")
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation '{activation}'")
        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        self.weights_init = weights_init
        self.act, self.act_prime = ACTIVATIONS[activation]

        rng = np.random.default_rng(seed)
        self.W = [init_weights(weights_init, n_in, n_out, rng)
                  for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.b = [np.zeros((1, n_out)) for n_out in layer_sizes[1:]]

        # Normalisation parameters, filled in by fit_normalization()
        self.mean = np.zeros(layer_sizes[0])
        self.std = np.ones(layer_sizes[0])

    # --- preprocessing -----------------------------------------------------

    def fit_normalization(self, X):
        """Learn z-score parameters from the training set only."""
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        self.std = np.where(std == 0, 1.0, std)

    def normalize(self, X):
        return (X - self.mean) / self.std

    # --- feedforward -------------------------------------------------------

    def forward(self, X):
        """
        Feedforward. X must already be normalised.
        Stores Z (weighted sums) and A (activations) of every layer for backprop.
        Returns the softmax output P of shape (m, n_classes).
        """
        self.A = [X]
        self.Z = []
        last = len(self.W) - 1
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            Z = self.A[-1] @ W + b
            A = softmax(Z) if l == last else self.act(Z)
            self.Z.append(Z)
            self.A.append(A)
        return self.A[-1]

    # --- backpropagation ---------------------------------------------------

    def backward(self, Y):
        """
        Backpropagation of the mean categorical cross-entropy.
        Must be called right after forward() on the same batch.

        Output layer: for softmax + cross-entropy, dL/dZ = (P - Y) / m.
        Then, going backwards (chain rule):
            dW[l]  = A[l]^T @ dZ
            db[l]  = sum of dZ over the batch
            dA     = dZ @ W[l]^T
            dZ     = dA * f'(Z[l-1])
        """
        m = Y.shape[0]
        dZ = (self.A[-1] - Y) / m
        dW = [None] * len(self.W)
        db = [None] * len(self.b)
        for l in range(len(self.W) - 1, -1, -1):
            dW[l] = self.A[l].T @ dZ
            db[l] = dZ.sum(axis=0, keepdims=True)
            if l > 0:
                dZ = (dZ @ self.W[l].T) * self.act_prime(self.Z[l - 1])
        return dW, db

    # --- gradient descent --------------------------------------------------

    def step(self, dW, db, learning_rate):
        """Move every parameter a small step against its gradient."""
        for l in range(len(self.W)):
            self.W[l] -= learning_rate * dW[l]
            self.b[l] -= learning_rate * db[l]

    # --- inference ---------------------------------------------------------

    def predict_proba(self, X_raw):
        """Raw (unnormalised) features -> class probabilities (m, n_classes)."""
        return self.forward(self.normalize(X_raw))

    def predict(self, X_raw):
        return self.predict_proba(X_raw).argmax(axis=1)

    # --- persistence -------------------------------------------------------

    def save(self, path, extra=None):
        topology = {
            'layer_sizes': self.layer_sizes,
            'activation': self.activation,
            'output_activation': 'softmax',
            'weights_init': self.weights_init,
            'classes': CLASSES,
            'features': FEATURES,
            'training': extra or {},
        }
        arrays = {'topology': np.array(json.dumps(topology)),
                  'mean': self.mean, 'std': self.std}
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            arrays[f'W{l}'] = W
            arrays[f'b{l}'] = b
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        topology = json.loads(str(data['topology']))
        model = cls(topology['layer_sizes'], topology['activation'], topology['weights_init'])
        model.W = [data[f'W{l}'] for l in range(len(model.W))]
        model.b = [data[f'b{l}'] for l in range(len(model.b))]
        model.mean = data['mean']
        model.std = data['std']
        model.topology = topology
        return model
