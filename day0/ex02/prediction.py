import numpy as np


def simple_predict(x, theta):
    """simple_predict accepts only 1d arrays for x, which is what the examples
    show, although it does not agree with the problem statement."""
    return (
        theta[0] + x * theta[1]
        if (
            isinstance(x, np.ndarray)
            and x.ndim == 1
            and len(x) != 0
            and isinstance(theta, np.ndarray)
            and theta.ndim == 1
            and len(theta) == 2
        )
        else None
    )


def poly_predict(x, theta):
    """poly_predict is a more generic version of simple_predict.

    It accepts any np.ndarray as x: scalars, vectors, matrices and higher-rank
    tensors. It also accepts any number of arguments in theta, representing the
    factors for x^0, x^1, x^2 and so on.
    """
    if (
        not isinstance(x, np.ndarray)
        or not isinstance(theta, np.ndarray)
        or theta.ndim != 1
    ):
        return None
    theta = theta.flatten()
    res = np.zeros_like(x)
    for arg in reversed(theta):
        res *= x
        res += arg
    return res
