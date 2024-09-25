from functools import wraps
import numpy as np


def validate_loss_args(func):
    @wraps(func)
    def wrapper(y, y_hat):
        return (
            func(y, y_hat)
            if (
                isinstance(y, np.ndarray)
                and isinstance(y_hat, np.ndarray)
                and y.shape == y_hat.shape
                and (y.ndim == 1 or (y.ndim == 2 and 1 in y.shape))
                and y.size > 0
            )
            else None
        )

    return wrapper


@validate_loss_args
def loss_elem_(y, y_hat):
    return (y - y_hat) ** 2


@validate_loss_args
def mse_(y, y_hat):
    return loss_elem_(y, y_hat).mean()


@validate_loss_args
def loss_(y, y_hat):
    return mse_(y, y_hat) / 2
