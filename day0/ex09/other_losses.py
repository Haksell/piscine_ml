from math import sqrt
import numpy as np
from vec_loss import validate_loss_args


def __ssr(y, y_hat):
    return ((y - y_hat) ** 2).sum()


@validate_loss_args
def mse_(y, y_hat):
    return __ssr(y, y_hat) / y.size


@validate_loss_args
def rmse_(y, y_hat):
    return sqrt(mse_(y, y_hat))


@validate_loss_args
def mae_(y, y_hat):
    return (np.abs(y - y_hat)).sum() / y.size


@validate_loss_args
def r2score_(y, y_hat):
    return 1 - __ssr(y, y_hat) / __ssr(y, y.mean())
