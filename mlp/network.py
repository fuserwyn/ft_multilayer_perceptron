import json

import numpy as np

from .activations import ACTIVATIONS
from .dataset import Dataset
from .initializers import INITIALIZERS
from .layers import DenseLayer
from .optimizers import OPTIMIZERS


class Normalizer:
    def __init__(self, n_features):
        self.mean = np.zeros(n_features)
        self.std = np.ones(n_features)

    def fit(self, X):
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        self.std = np.where(std == 0, 1.0, std)

    def transform(self, X):
        return (X - self.mean) / self.std


class MLP:
    def __init__(self, layer_sizes, activation='sigmoid', weights_init='heUniform', seed=42,
                 optimizer='sgd'):
        if len(layer_sizes) < 3:
            raise ValueError("need at least an input, one hidden and an output layer")
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation '{activation}'")
        if weights_init not in INITIALIZERS:
            raise ValueError(f"unknown weights initializer '{weights_init}'")
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer '{optimizer}'")

        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        self.weights_init = weights_init
        self.optimizer_name = optimizer
        self._activation = ACTIVATIONS[activation]
        self.optimizer = OPTIMIZERS[optimizer]()
        self.normalizer = Normalizer(layer_sizes[0])

        rng = np.random.default_rng(seed)
        initializer = INITIALIZERS[weights_init]
        pairs = list(zip(self.layer_sizes[:-1], self.layer_sizes[1:]))
        self.layers = [
            DenseLayer(n_in, n_out, self._activation, initializer, rng, output=(i == len(pairs) - 1))
            for i, (n_in, n_out) in enumerate(pairs)
        ]

    @property
    def W(self):
        return [layer.W for layer in self.layers]

    @W.setter
    def W(self, values):
        for layer, value in zip(self.layers, values):
            layer.W = value

    @property
    def b(self):
        return [layer.b for layer in self.layers]

    @b.setter
    def b(self, values):
        for layer, value in zip(self.layers, values):
            layer.b = value

    @property
    def mean(self):
        return self.normalizer.mean

    @mean.setter
    def mean(self, value):
        self.normalizer.mean = value

    @property
    def std(self):
        return self.normalizer.std

    @std.setter
    def std(self, value):
        self.normalizer.std = value

    def fit_normalization(self, X):
        self.normalizer.fit(X)

    def normalize(self, X):
        return self.normalizer.transform(X)

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, Y):
        m = Y.shape[0]
        dZ = (self.layers[-1].A - Y) / m
        dW = [None] * len(self.layers)
        db = [None] * len(self.layers)
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            dW[i], db[i], dA_prev = layer.backward(dZ)
            if i > 0:
                prev = self.layers[i - 1]
                dZ = dA_prev * prev.activation.derivative(prev.Z)
        return dW, db

    def step(self, dW, db, learning_rate):
        self.optimizer.step(self.layers, dW, db, learning_rate)

    def predict_proba(self, X_raw):
        return self.forward(self.normalize(X_raw))

    def predict(self, X_raw):
        return self.predict_proba(X_raw).argmax(axis=1)

    def save(self, path, extra=None):
        topology = {
            'layer_sizes': self.layer_sizes,
            'activation': self.activation,
            'output_activation': 'softmax',
            'weights_init': self.weights_init,
            'optimizer': self.optimizer_name,
            'classes': Dataset.CLASSES,
            'features': Dataset.FEATURES,
            'training': extra or {},
        }
        arrays = {'topology': np.array(json.dumps(topology)),
                  'mean': self.mean, 'std': self.std}
        for i, layer in enumerate(self.layers):
            arrays[f'W{i}'] = layer.W
            arrays[f'b{i}'] = layer.b
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        topology = json.loads(str(data['topology']))
        model = cls(topology['layer_sizes'], topology['activation'], topology['weights_init'],
                    optimizer=topology.get('optimizer', 'sgd'))
        model.W = [data[f'W{i}'] for i in range(len(model.layers))]
        model.b = [data[f'b{i}'] for i in range(len(model.layers))]
        model.mean = data['mean']
        model.std = data['std']
        model.topology = topology
        return model
