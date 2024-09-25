import numpy as np


def predict_(x, theta):
    if (
        isinstance(x, np.ndarray)
        and x.ndim == 2
        and x.shape[0] >= 1
        and x.shape[1] >= 1
        and isinstance(theta, np.ndarray)
        and theta.shape == (x.shape[1] + 1, 1)
    ):
        return np.hstack((np.ones((x.shape[0], 1)), x)) @ theta
