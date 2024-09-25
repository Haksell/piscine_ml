import numpy as np


def gradient(x, y, theta):
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
        x_stack = np.hstack((np.ones((x.shape[0], 1)), x))
        return x_stack.T @ (x_stack @ theta - y) / y.size
