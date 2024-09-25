import numpy as np


def sigmoid_(x):
    if isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] == 1:
        with np.errstate(over="ignore"):
            return 1 / (1 + np.exp(-x))
