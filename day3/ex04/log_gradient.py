import math
import numpy as np
from operator import mul


def __sigmoid(x):
    return 1 / (1 + math.exp(-x))


def log_gradient(x, y, theta):
    if (
        isinstance(x, np.ndarray)
        and x.ndim == 2
        and x.shape[0] >= 1
        and x.shape[1] >= 1
        and isinstance(y, np.ndarray)
        and all(yi == 0 or yi == 1 for yi in y.flatten())
        and y.shape == (x.shape[0], 1)
        and isinstance(theta, np.ndarray)
        and theta.shape == (x.shape[1] + 1, 1)
    ):
        y_diff = [
            __sigmoid(
                theta[0, 0]
                + sum(xi_row * ti for xi_row, ti in zip(x_row, theta[1:, 0]))
            )
            - yi
            for x_row, yi in zip(x, y)
        ]
        return np.vstack(
            [
                sum(y_diff) / len(y_diff),
                [sum(map(mul, x_col, y_diff)) / len(y_diff) for x_col in x.T],
            ]
        )
