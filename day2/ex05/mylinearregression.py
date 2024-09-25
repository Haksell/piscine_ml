from functools import wraps
import numpy as np


class MyLinearRegression:
    def __init__(self, theta, alpha=0.001, max_iter=1000):
        def check_theta(theta):
            assert isinstance(theta, list) or isinstance(theta, np.ndarray)
            if isinstance(theta, list):
                assert len(theta) >= 1
                assert all(isinstance(row, list) for row in theta)
                assert all(len(row) == 1 for row in theta for row in theta)
                assert all(
                    isinstance(row[0], float) or isinstance(row[0], int)
                    for row in theta
                )
                return np.array(theta, dtype=np.float64)
            else:
                return theta.astype(dtype=np.float64)

        self.theta = check_theta(theta)
        assert isinstance(alpha, float)
        assert alpha > 0
        self.alpha = alpha
        assert isinstance(max_iter, int)
        assert max_iter >= 1
        self.max_iter = max_iter

    @staticmethod
    def __validate_loss_args(func):
        @wraps(func)
        def wrapper(y, y_hat):
            assert isinstance(y, np.ndarray)
            assert isinstance(y_hat, np.ndarray)
            assert y.shape == y_hat.shape
            assert y.ndim == 1 or (y.ndim == 2 and 1 in y.shape)
            assert y.size > 0
            return func(y, y_hat)

        return staticmethod(wrapper)

    @staticmethod
    def __validate_xy(method):
        def wrapper(self, x, y=None):
            assert isinstance(x, np.ndarray)
            assert x.ndim == 2
            assert x.shape[0] >= 1
            assert x.shape[1] + 1 == self.theta.shape[0]
            if y is None:
                return method(self, x)
            else:
                assert isinstance(y, np.ndarray)
                assert y.shape == (x.shape[0], 1)
                return method(self, x, y)

        return wrapper

    @__validate_xy
    def gradient_(self, x, y):
        x_stack = np.hstack((np.ones((x.shape[0], 1)), x))
        return x_stack.T @ (x_stack @ self.theta - y) / y.size

    @__validate_xy
    def fit_(self, x, y):
        for _ in range(self.max_iter):
            self.theta -= self.alpha * self.gradient_(x, y)

    @__validate_xy
    def predict_(self, x):
        return np.hstack((np.ones((x.shape[0], 1)), x)) @ self.theta

    @__validate_loss_args
    def loss_elem_(y, y_hat):
        return (y - y_hat) ** 2

    @__validate_loss_args
    def mse_(y, y_hat):
        return MyLinearRegression.loss_elem_(y, y_hat).mean()

    @__validate_loss_args
    def loss_(y, y_hat):
        return MyLinearRegression.mse_(y, y_hat) / 2
