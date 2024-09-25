import numpy as np
from gradient import gradient


def fit_(x, y, theta, alpha, max_iter):
    if (
        isinstance(x, np.ndarray)
        and x.ndim == 2
        and x.shape[0] >= 1
        and x.shape[1] >= 1
        and isinstance(y, np.ndarray)
        and y.shape == (x.shape[0], 1)
        and isinstance(theta, np.ndarray)
        and theta.shape == (x.shape[1] + 1, 1)
        and isinstance(alpha, float)
        and 0 < alpha < 1
        and isinstance(max_iter, int)
        and max_iter >= 1
    ):
        theta = theta.copy()
        for _ in range(max_iter):
            theta -= alpha * gradient(x, y, theta)
        return theta
