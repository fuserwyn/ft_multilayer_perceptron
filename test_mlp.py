"""
Unit tests for mlp.py. The key one is the gradient check: backpropagation
must agree with a numerical derivative (f(w+e) - f(w-e)) / 2e.

    python3 test_mlp.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlp import (MLP, binary_crossentropy, binary_metrics, categorical_crossentropy, one_hot,  # noqa: E402
                 sigmoid, sigmoid_prime, softmax)


def test_activations():
    assert abs(sigmoid(0.0) - 0.5) < 1e-12
    assert sigmoid(1000.0) == 1.0 and sigmoid(-1000.0) >= 0.0   # no overflow
    z = np.array([-2.0, 0.0, 3.0])
    num = (sigmoid(z + 1e-6) - sigmoid(z - 1e-6)) / 2e-6
    assert np.allclose(sigmoid_prime(z), num, atol=1e-8)


def test_softmax():
    P = softmax(np.array([[1.0, 2.0], [1000.0, 1000.0], [-5.0, 5.0]]))
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.all(P >= 0)
    assert np.allclose(P[1], [0.5, 0.5])                        # stable for big inputs
    assert P[2, 1] > P[2, 0]


def test_losses():
    y = np.array([1, 0, 1])
    assert binary_crossentropy(y, np.array([1.0, 0.0, 1.0])) < 1e-10
    assert binary_crossentropy(y, np.array([0.9, 0.1, 0.9])) < binary_crossentropy(y, np.array([0.6, 0.4, 0.6]))
    # for two classes, categorical CE on softmax == binary CE on P(class 1)
    P = np.array([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    assert np.isclose(categorical_crossentropy(one_hot(y), P), binary_crossentropy(y, P[:, 1]))


def test_shapes():
    model = MLP([5, 4, 3, 2], seed=0)
    assert [W.shape for W in model.W] == [(5, 4), (4, 3), (3, 2)]
    assert [b.shape for b in model.b] == [(1, 4), (1, 3), (1, 2)]
    P = model.forward(np.random.default_rng(0).normal(size=(7, 5)))
    assert P.shape == (7, 2)
    assert np.allclose(P.sum(axis=1), 1.0)


def numerical_gradient_check(activation):
    rng = np.random.default_rng(1)
    model = MLP([4, 5, 3, 2], activation=activation, seed=1)
    X = rng.normal(size=(6, 4))
    Y = one_hot(rng.integers(0, 2, 6))

    model.forward(X)
    dW, db = model.backward(Y)

    eps = 1e-6
    for params, grads in ((model.W, dW), (model.b, db)):
        for P, G in zip(params, grads):
            num = np.zeros_like(P)
            for idx in np.ndindex(P.shape):
                old = P[idx]
                P[idx] = old + eps
                plus = categorical_crossentropy(Y, model.forward(X))
                P[idx] = old - eps
                minus = categorical_crossentropy(Y, model.forward(X))
                P[idx] = old
                num[idx] = (plus - minus) / (2 * eps)
            rel = np.linalg.norm(G - num) / max(np.linalg.norm(G) + np.linalg.norm(num), 1e-12)
            assert rel < 1e-6, f'{activation}: gradient mismatch, relative error {rel:.2e}'


def test_gradient_check():
    for activation in ('sigmoid', 'tanh', 'relu'):
        numerical_gradient_check(activation)


def test_training_reduces_loss():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    Y = one_hot(y)
    model = MLP([2, 8, 8, 2], seed=2)
    model.fit_normalization(X)
    Xn = model.normalize(X)
    before = categorical_crossentropy(Y, model.forward(Xn))
    for _ in range(300):
        model.forward(Xn)
        model.step(*model.backward(Y), learning_rate=0.5)
    after = categorical_crossentropy(Y, model.forward(Xn))
    assert after < before / 3, (before, after)
    assert np.mean(model.predict(X) == y) > 0.95


def test_binary_metrics():
    m = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert (m['tp'], m['fn'], m['fp'], m['tn']) == (1, 1, 1, 1)
    assert abs(m['precision'] - 0.5) < 1e-12 and abs(m['recall'] - 0.5) < 1e-12
    assert abs(m['f1'] - 0.5) < 1e-12


def test_adam_reduces_loss():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    Y = one_hot(y)
    model = MLP([2, 8, 8, 2], optimizer='adam', seed=2)
    model.fit_normalization(X)
    Xn = model.normalize(X)
    before = categorical_crossentropy(Y, model.forward(Xn))
    for _ in range(80):
        model.forward(Xn)
        model.step(*model.backward(Y), learning_rate=0.05)
    after = categorical_crossentropy(Y, model.forward(Xn))
    assert after < before / 3, (before, after)
    assert np.mean(model.predict(X) == y) > 0.95


def test_reproducibility():
    a, b = MLP([3, 4, 4, 2], seed=7), MLP([3, 4, 4, 2], seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a.W, b.W))


def test_save_load():
    model = MLP([3, 4, 4, 2], activation='tanh', seed=3)
    model.fit_normalization(np.random.default_rng(3).normal(5, 2, size=(10, 3)))
    X = np.random.default_rng(4).normal(size=(5, 3))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'm.npz')
        model.save(path)
        loaded = MLP.load(path)
    assert loaded.layer_sizes == [3, 4, 4, 2] and loaded.activation == 'tanh'
    assert np.allclose(loaded.predict_proba(X), model.predict_proba(X))


def main():
    tests = [v for k, v in globals().items() if k.startswith('test_') and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f'ok    {test.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'FAIL  {test.__name__}: {e}')
    print(f'\n{len(tests) - failed}/{len(tests)} tests passed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
