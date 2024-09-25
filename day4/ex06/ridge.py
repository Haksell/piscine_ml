from functools import wraps
import numpy as np


def _norm(arr):
    return arr.flatten() @ arr.flatten()


def _check_theta(theta):
    assert isinstance(theta, list) or isinstance(theta, np.ndarray)
    if isinstance(theta, list):
        assert len(theta) >= 1
        assert all(isinstance(row, list) for row in theta)
        assert all(len(row) == 1 for row in theta for row in theta)
        assert all(
            isinstance(row[0], float) or isinstance(row[0], int) for row in theta
        )
        return np.array(theta, dtype=np.float64)
    else:
        return theta.astype(dtype=np.float64)


def _validate_xy(method):
    @wraps(method)
    def wrapper(self, x, y=None):
        assert isinstance(x, np.ndarray)
        assert x.ndim == 2
        assert x.shape[0] >= 1
        assert x.shape[1] + 1 == self._theta.shape[0]
        if y is None:
            return method(self, x)
        else:
            assert isinstance(y, np.ndarray)
            assert y.shape == (x.shape[0], 1)
            return method(self, x, y)

    return wrapper


def _validate_reg_loss_args(func):
    @wraps(func)
    def wrapper(*args):
        assert 2 <= len(args) <= 3
        y, y_hat = args[-2:]
        assert isinstance(y, np.ndarray)
        assert isinstance(y_hat, np.ndarray)
        assert np.issubdtype(y.dtype, np.number)
        assert np.issubdtype(y_hat.dtype, np.number)
        assert y.shape == y_hat.shape
        return func(*args)

    return wrapper


class MyRidge:
    def __init__(self, theta, alpha=0.001, max_iter=1000, lambda_=0.5):
        self._theta = _check_theta(theta)
        assert isinstance(alpha, float)
        assert alpha > 0
        self.alpha = alpha
        assert isinstance(max_iter, int)
        assert max_iter >= 1
        self.max_iter = max_iter
        assert isinstance(lambda_, float) or isinstance(lambda_, int)
        assert lambda_ >= 0
        self.lambda_ = float(lambda_)

    def get_params_(self):
        return self._theta.copy()

    def set_params_(self, theta):
        self._theta = _check_theta(theta)

    @_validate_xy
    def gradient_(self, x, y):
        x_stack = np.hstack((np.ones((x.shape[0], 1)), x))
        theta_prime = self._theta.copy()
        theta_prime[0, 0] = 0
        return (
            x_stack.T @ ((x_stack @ self._theta) - y) + self.lambda_ * theta_prime
        ) / y.size

    @_validate_xy
    def fit_(self, x, y):
        for _ in range(self.max_iter):
            self._theta -= self.alpha * self.gradient_(x, y)

    @_validate_xy
    def predict_(self, x):
        return np.hstack((np.ones((x.shape[0], 1)), x)) @ self._theta

    @staticmethod
    @_validate_reg_loss_args
    def loss_elem_(y, y_hat):
        return (y - y_hat) ** 2

    @staticmethod
    @_validate_reg_loss_args
    def mse_(y, y_hat):
        return MyRidge.loss_elem_(y, y_hat).mean()

    @_validate_reg_loss_args
    def loss_(self, y, y_hat):
        return (_norm(y - y_hat) + self.lambda_ * _norm(self._theta[1:, 0])) / (
            2 * y.shape[0]
        )
