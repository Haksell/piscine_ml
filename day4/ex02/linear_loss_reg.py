import numpy as np


def __true_l2(arr):
    return arr.flatten() @ arr.flatten()


def validate_reg_loss_args(func):
    def wrapper(y, y_hat, theta, lambda_):
        return (
            func(y, y_hat, theta, lambda_)
            if (
                isinstance(y, np.ndarray)
                and y.ndim == 2
                and y.shape[1] == 1
                and isinstance(y_hat, np.ndarray)
                and y.shape == y_hat.shape
                and isinstance(theta, np.ndarray)
                and theta.ndim == 2
                and theta.shape[0] >= 1
                and theta.shape[1] == 1
                and (isinstance(lambda_, float) or isinstance(lambda_, int))
                and lambda_ >= 0
            )
            else None
        )

    return wrapper


@validate_reg_loss_args
def reg_loss_(y, y_hat, theta, lambda_):
    return (__true_l2(y - y_hat) + lambda_ * __true_l2(theta[1:, 0])) / (2 * y.shape[0])
