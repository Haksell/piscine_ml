import numpy as np


def simple_predict(x, theta):
    if (
        isinstance(x, np.ndarray)
        and x.ndim == 2
        and x.shape[0] >= 1
        and x.shape[1] >= 1
        and isinstance(theta, np.ndarray)
        and theta.shape == (x.shape[1] + 1, 1)
    ):
        y_hat = np.full(x.shape[0], theta[0])
        for i, t in enumerate(theta[1:]):
            y_hat += t * x[:, i]
        return y_hat.reshape((-1, 1))
