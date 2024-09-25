import numpy as np


def add_polynomial_features(x, power):
    if (
        isinstance(power, int)
        and power > 0
        and isinstance(x, np.ndarray)
        and np.issubdtype(x.dtype, np.number)
        and x.ndim == 2
    ):
        return np.hstack([x**i for i in range(1, power + 1)])
