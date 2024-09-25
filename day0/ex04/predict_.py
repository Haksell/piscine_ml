# Calling the file predict_.py to avoid conflicts when running pytest

import numpy as np
from tools import add_intercept


def predict_(x, theta):
    return (
        add_intercept(x) @ theta.flatten().reshape((2, 1))
        if (
            isinstance(x, np.ndarray)
            and (x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1))
            and isinstance(theta, np.ndarray)
            and theta.size == 2
        )
        else None
    )
