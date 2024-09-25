from math import log
import numpy as np


def log_loss_(y, y_hat, eps=1e-15):
    if (
        isinstance(y, np.ndarray)
        and y.ndim == 2
        and y.shape[0] >= 1
        and y.shape[1] == 1
        and all(yi == 0 or yi == 1 for yi in y.flatten())
        and isinstance(y_hat, np.ndarray)
        and y.shape == y_hat.shape
        and all(0 <= yi_hat <= 1 for yi_hat in y_hat.flatten())
        and isinstance(eps, float)
        and 0 < eps < 0.1
    ):
        return sum(
            yi * log(max(eps, yi_hat)) + (1 - yi) * log(max(eps, 1 - yi_hat))
            for yi, yi_hat in zip(y.flatten(), y_hat.flatten())
        ) / -len(y)
