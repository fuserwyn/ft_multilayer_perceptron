"""
Checks for the network classes. The important one compares backpropagation
with a numerical derivative (f(w+e) - f(w-e)) / 2e.

    python3 tests/test_mlp.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlp import (MLP, BinaryCrossentropy, CategoricalCrossentropy, Dataset, Metrics,  # noqa: E402
                 Sigmoid, Softmax)


class GradientCheck:
    """Compare analytic gradients with a central finite difference."""

    def __init__(self, activation):
        self.activation = activation
        self.loss = CategoricalCrossentropy()

    def run(self):
        rng = np.random.default_rng(1)
        model = MLP([4, 5, 3, 2], activation=self.activation, seed=1)
        X = rng.normal(size=(6, 4))
        Y = Dataset.encode(rng.integers(0, 2, 6))

        model.forward(X)
        dW, db = model.backward(Y)

        eps = 1e-6
        for params, grads in ((model.W, dW), (model.b, db)):
            for P, G in zip(params, grads):
                num = np.zeros_like(P)
                for idx in np.ndindex(P.shape):
                    old = P[idx]
                    P[idx] = old + eps
                    plus = self.loss(Y, model.forward(X))
                    P[idx] = old - eps
                    minus = self.loss(Y, model.forward(X))
                    P[idx] = old
                    num[idx] = (plus - minus) / (2 * eps)
                rel = np.linalg.norm(G - num) / max(np.linalg.norm(G) + np.linalg.norm(num), 1e-12)
                assert rel < 1e-6, f'{self.activation}: gradient mismatch, relative error {rel:.2e}'


class TestSuite:
    """One method per check. `run` prints which ones passed."""

    def activations(self):
        sigmoid = Sigmoid()
        assert abs(sigmoid(0.0) - 0.5) < 1e-12
        assert sigmoid(1000.0) == 1.0 and sigmoid(-1000.0) >= 0.0
        z = np.array([-2.0, 0.0, 3.0])
        num = (sigmoid(z + 1e-6) - sigmoid(z - 1e-6)) / 2e-6
        assert np.allclose(sigmoid.derivative(z), num, atol=1e-8)

    def softmax(self):
        softmax = Softmax()
        P = softmax(np.array([[1.0, 2.0], [1000.0, 1000.0], [-5.0, 5.0]]))
        assert np.allclose(P.sum(axis=1), 1.0)
        assert np.all(P >= 0)
        assert np.allclose(P[1], [0.5, 0.5])
        assert P[2, 1] > P[2, 0]

    def losses(self):
        y = np.array([1, 0, 1])
        binary = BinaryCrossentropy()
        categorical = CategoricalCrossentropy()
        assert binary(y, np.array([1.0, 0.0, 1.0])) < 1e-10
        assert binary(y, np.array([0.9, 0.1, 0.9])) < binary(y, np.array([0.6, 0.4, 0.6]))
        P = np.array([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
        assert np.isclose(categorical(Dataset.encode(y), P), binary(y, P[:, 1]))

    def shapes(self):
        model = MLP([5, 4, 3, 2], seed=0)
        assert [W.shape for W in model.W] == [(5, 4), (4, 3), (3, 2)]
        assert [b.shape for b in model.b] == [(1, 4), (1, 3), (1, 2)]
        P = model.forward(np.random.default_rng(0).normal(size=(7, 5)))
        assert P.shape == (7, 2)
        assert np.allclose(P.sum(axis=1), 1.0)

    def gradient_check(self):
        for activation in ('sigmoid', 'tanh', 'relu'):
            GradientCheck(activation).run()

    def _learns(self, optimizer, steps, learning_rate):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(200, 2))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        Y = Dataset.encode(y)
        model = MLP([2, 8, 8, 2], optimizer=optimizer, seed=2)
        model.fit_normalization(X)
        Xn = model.normalize(X)
        loss = CategoricalCrossentropy()
        before = loss(Y, model.forward(Xn))
        for _ in range(steps):
            model.forward(Xn)
            model.step(*model.backward(Y), learning_rate=learning_rate)
        after = loss(Y, model.forward(Xn))
        assert after < before / 3, (before, after)
        assert np.mean(model.predict(X) == y) > 0.95

    def training_reduces_loss(self):
        self._learns('sgd', steps=300, learning_rate=0.5)

    def adam_reduces_loss(self):
        self._learns('adam', steps=80, learning_rate=0.05)

    def binary_metrics(self):
        m = Metrics([1, 1, 0, 0], [1, 0, 1, 0])
        assert (m['tp'], m['fn'], m['fp'], m['tn']) == (1, 1, 1, 1)
        assert abs(m['precision'] - 0.5) < 1e-12 and abs(m['recall'] - 0.5) < 1e-12
        assert abs(m['f1'] - 0.5) < 1e-12

    def reproducibility(self):
        a, b = MLP([3, 4, 4, 2], seed=7), MLP([3, 4, 4, 2], seed=7)
        assert all(np.array_equal(x, y) for x, y in zip(a.W, b.W))

    def save_load(self):
        model = MLP([3, 4, 4, 2], activation='tanh', seed=3)
        model.fit_normalization(np.random.default_rng(3).normal(5, 2, size=(10, 3)))
        X = np.random.default_rng(4).normal(size=(5, 3))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'm.npz')
            model.save(path)
            loaded = MLP.load(path)
        assert loaded.layer_sizes == [3, 4, 4, 2] and loaded.activation == 'tanh'
        assert np.allclose(loaded.predict_proba(X), model.predict_proba(X))

    def cases(self):
        return [getattr(self, name) for name in (
            'activations', 'softmax', 'losses', 'shapes', 'gradient_check',
            'training_reduces_loss', 'binary_metrics', 'adam_reduces_loss',
            'reproducibility', 'save_load',
        )]

    def run(self):
        failed = 0
        cases = self.cases()
        for case in cases:
            try:
                case()
                print(f'ok    {case.__name__}')
            except AssertionError as e:
                failed += 1
                print(f'FAIL  {case.__name__}: {e}')
        print(f'\n{len(cases) - failed}/{len(cases)} tests passed')
        return failed


if __name__ == '__main__':
    sys.exit(1 if TestSuite().run() else 0)
