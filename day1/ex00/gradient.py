import numpy as np


def __predict(x, theta):
    return np.hstack((np.ones((x.shape[0], 1)), x)) @ theta


def simple_gradient(x, y, theta):
    if (
        isinstance(x, np.ndarray)
        and isinstance(y, np.ndarray)
        and isinstance(theta, np.ndarray)
        and x.shape == y.shape
        and x.ndim == 2
        and x.shape[0] >= 1
        and x.shape[1] == 1
        and theta.shape == (2, 1)
    ):
        y_hat = __predict(x, theta)
        y_diff = y_hat - y
        return np.array([[y_diff.mean()], [(y_diff * x).mean()]])
