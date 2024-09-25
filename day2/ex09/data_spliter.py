from math import floor
import numpy as np
import random


def data_spliter(x, y, proportion, *, seed=None):
    if (
        isinstance(x, np.ndarray)
        and x.ndim == 2
        and isinstance(y, np.ndarray)
        and y.shape == (x.shape[0], 1)
        and isinstance(proportion, float)
        and 0 <= proportion <= 1
    ):
        n = x.shape[0]
        indices = list(range(n))
        train_size = floor(n * proportion)
        if seed is not None:
            random.seed(seed)
        random.shuffle(indices)
        x_train = np.array([x[i] for i in indices[:train_size]]).reshape(
            (-1, x.shape[1])
        )
        x_test = np.array([x[i] for i in indices[train_size:]]).reshape(
            (-1, x.shape[1])
        )
        y_train = np.array([y[i] for i in indices[:train_size]]).reshape((-1, 1))
        y_test = np.array([y[i] for i in indices[train_size:]]).reshape((-1, 1))
        return (x_train, x_test, y_train, y_test)
