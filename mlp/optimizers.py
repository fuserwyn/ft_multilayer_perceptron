import numpy as np


class SGD:
    def step(self, layers, dW, db, learning_rate):
        for layer, grad_w, grad_b in zip(layers, dW, db):
            layer.W -= learning_rate * grad_w
            layer.b -= learning_rate * grad_b


class Adam:
    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.ready = False

    def step(self, layers, dW, db, learning_rate):
        if not self.ready:
            self.mW = [np.zeros_like(layer.W) for layer in layers]
            self.vW = [np.zeros_like(layer.W) for layer in layers]
            self.mb = [np.zeros_like(layer.b) for layer in layers]
            self.vb = [np.zeros_like(layer.b) for layer in layers]
            self.ready = True
        self.t += 1
        b1, b2, eps, t = self.beta1, self.beta2, self.eps, self.t
        for i, layer in enumerate(layers):
            self.mW[i] = b1 * self.mW[i] + (1.0 - b1) * dW[i]
            self.vW[i] = b2 * self.vW[i] + (1.0 - b2) * (dW[i] ** 2)
            self.mb[i] = b1 * self.mb[i] + (1.0 - b1) * db[i]
            self.vb[i] = b2 * self.vb[i] + (1.0 - b2) * (db[i] ** 2)
            mW = self.mW[i] / (1.0 - b1 ** t)
            vW = self.vW[i] / (1.0 - b2 ** t)
            mb = self.mb[i] / (1.0 - b1 ** t)
            vb = self.vb[i] / (1.0 - b2 ** t)
            layer.W -= learning_rate * mW / (np.sqrt(vW) + eps)
            layer.b -= learning_rate * mb / (np.sqrt(vb) + eps)


OPTIMIZERS = {
    'sgd': SGD,
    'adam': Adam,
}
