import numpy as np


def gradient(x, y, theta):
    if (
        isinstance(x, np.ndarray)
        and x.ndim == 2
        and x.shape[0] >= 1
        and x.shape[1] >= 1
        and isinstance(y, np.ndarray)
        and y.shape == (x.shape[0], 1)
        and isinstance(theta, np.ndarray)
        and theta.shape == (x.shape[1] + 1, 1)
    ):
        x_stack = np.hstack((np.ones((x.shape[0], 1)), x))
        return x_stack.T @ (x_stack @ theta - y) / y.size
