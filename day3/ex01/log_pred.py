import numpy as np
from sigmoid import sigmoid_


def logistic_predict_(x, theta):
    if (
        isinstance(x, np.ndarray)
        and x.ndim == 2
        and isinstance(theta, np.ndarray)
        and theta.shape == (x.shape[1] + 1, 1)
    ):
        return sigmoid_(np.hstack((np.ones((x.shape[0], 1)), x)) @ theta)
