"""How a dense layer draws its starting weights."""

import numpy as np


class HeUniform:
    """U(-sqrt(6 / n_in), +sqrt(6 / n_in)). A good default for ReLU, fine for sigmoid too."""

    def __call__(self, n_in, n_out, rng):
        limit = np.sqrt(6.0 / n_in)
        return rng.uniform(-limit, limit, (n_in, n_out))


class HeNormal:
    def __call__(self, n_in, n_out, rng):
        return rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, n_out))


class XavierUniform:
    """U(-sqrt(6 / (n_in + n_out)), +sqrt(6 / (n_in + n_out))). Designed for sigmoid and tanh."""

    def __call__(self, n_in, n_out, rng):
        limit = np.sqrt(6.0 / (n_in + n_out))
        return rng.uniform(-limit, limit, (n_in, n_out))


INITIALIZERS = {
    'heUniform': HeUniform(),
    'heNormal': HeNormal(),
    'xavierUniform': XavierUniform(),
}
WEIGHT_INITS = list(INITIALIZERS)
