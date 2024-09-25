import numpy as np


def add_intercept(x):
    if not isinstance(x, np.ndarray) or x.ndim >= 3:
        return None
    if x.ndim <= 1:
        x = x.reshape((x.size, 1))
    ones = np.ones((x.shape[0], 1))
    return np.hstack((ones, x))
