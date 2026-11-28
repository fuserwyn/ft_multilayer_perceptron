"""Activation classes. Each one is f(z), and f'(z) for the hidden layers."""

import numpy as np


class Activation:
    """f(z) via the instance itself, f'(z) via derivative()."""

    def __call__(self, z):
        raise NotImplementedError

    def derivative(self, z):
        raise NotImplementedError


class Sigmoid(Activation):
    def __call__(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def derivative(self, z):
        s = self(z)
        return s * (1.0 - s)


class ReLU(Activation):
    def __call__(self, z):
        return np.maximum(0.0, z)

    def derivative(self, z):
        return (z > 0).astype(float)


class Tanh(Activation):
    def __call__(self, z):
        return np.tanh(z)

    def derivative(self, z):
        return 1.0 - np.tanh(z) ** 2


class Softmax(Activation):
    """Row-wise softmax. Subtracting the row max avoids overflow in exp.

    Combined with cross-entropy, the backward step uses (P - Y) directly,
    so this class has no separate derivative.
    """

    def __call__(self, z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


ACTIVATIONS = {
    'sigmoid': Sigmoid(),
    'relu': ReLU(),
    'tanh': Tanh(),
}
