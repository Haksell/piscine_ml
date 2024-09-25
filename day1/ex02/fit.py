import numpy as np
from vec_gradient import gradient


def fit_(x, y, theta, alpha, max_iter):
    if (
        isinstance(x, np.ndarray)
        and isinstance(y, np.ndarray)
        and isinstance(theta, np.ndarray)
        and x.shape == y.shape
        and x.ndim == 2
        and x.shape[0] >= 1
        and x.shape[1] == 1
        and theta.shape == (2, 1)
        and isinstance(alpha, float)
        and 0 < alpha < 1
        and isinstance(max_iter, int)
        and max_iter >= 1
    ):
        theta = theta.copy()
        for _ in range(max_iter):
            theta -= alpha * gradient(x, y, theta)
        return theta
