import numpy as np


class Activation:
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
    def __call__(self, z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


ACTIVATIONS = {
    'sigmoid': Sigmoid(),
    'relu': ReLU(),
    'tanh': Tanh(),
}
