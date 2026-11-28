"""One dense layer: weighted sum, bias, then an activation."""

import numpy as np

from .activations import Softmax


class DenseLayer:
    """Fully connected layer. The output layer uses softmax instead of `activation`."""

    def __init__(self, n_in, n_out, activation, initializer, rng, output=False):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.output = output
        self.softmax = Softmax()
        self.W = initializer(n_in, n_out, rng)
        self.b = np.zeros((1, n_out))

    def forward(self, A_prev):
        """Z = A_prev · W + b, then f(Z). Caches A_prev, Z and A for backprop."""
        self.A_prev = A_prev
        self.Z = A_prev @ self.W + self.b
        self.A = self.softmax(self.Z) if self.output else self.activation(self.Z)
        return self.A

    def backward(self, dZ):
        """dW and db for this layer, and the gradient to pass to the layer below."""
        dW = self.A_prev.T @ dZ
        db = dZ.sum(axis=0, keepdims=True)
        dA_prev = dZ @ self.W.T
        return dW, db, dA_prev
