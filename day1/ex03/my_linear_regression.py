from functools import wraps
import numpy as np


def validate_loss_args(func):
    @wraps(func)
    def wrapper(y, y_hat):
        assert isinstance(y, np.ndarray)
        assert isinstance(y_hat, np.ndarray)
        assert y.shape == y_hat.shape
        assert y.ndim == 1 or (y.ndim == 2 and 1 in y.shape)
        assert y.size > 0
        return func(y, y_hat)

    return staticmethod(wrapper)


class MyLinearRegression:
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        assert isinstance(thetas, np.ndarray)
        assert thetas.shape == (2, 1)
        assert isinstance(alpha, float)
        assert 0 < alpha < 1
        assert isinstance(max_iter, int)
        assert max_iter >= 1
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas.astype(float)

    def gradient_(self, x, y):
        x_stack = np.hstack((np.ones((x.shape[0], 1)), x))
        return x_stack.T @ (x_stack @ self.thetas - y) / y.size

    def fit_(self, x, y):
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert x.shape == y.shape
        assert x.ndim == 2
        assert x.shape[0] >= 1
        assert x.shape[1] == 1
        for _ in range(self.max_iter):
            self.thetas -= self.alpha * self.gradient_(x, y)

    def predict_(self, x):
        return np.hstack((np.ones((x.shape[0], 1)), x)) @ self.thetas

    @validate_loss_args
    def loss_elem_(y, y_hat):
        return (y - y_hat) ** 2

    @validate_loss_args
    def mse_(y, y_hat):
        return MyLinearRegression.loss_elem_(y, y_hat).mean()

    @validate_loss_args
    def loss_(y, y_hat):
        return MyLinearRegression.mse_(y, y_hat) / 2
