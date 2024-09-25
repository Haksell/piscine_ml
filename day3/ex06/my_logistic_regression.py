import numpy as np


class MyLogisticRegression:
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

    @staticmethod
    def sigmoid_(x):
        assert isinstance(x, np.ndarray)
        assert x.ndim == 2
        assert x.shape[1] == 1
        return 1 / (1 + np.exp(-x))

    @__validate_xy
    def gradient_(self, x, y):
        x_stack = np.hstack((np.ones((x.shape[0], 1)), x))
        return (
            x_stack.T
            @ (MyLogisticRegression.sigmoid_(x_stack @ self.theta) - y)
            / y.size
        )

    @__validate_xy
    def fit_(self, x, y):
        for _ in range(self.max_iter):
            self.theta -= self.alpha * self.gradient_(x, y)

    @__validate_xy
    def predict_(self, x):
        return MyLogisticRegression.sigmoid_(
            np.hstack((np.ones((x.shape[0], 1)), x)) @ self.theta
        )

    @staticmethod
    def loss_(y, y_hat, eps=1e-15):
        assert isinstance(y, np.ndarray)
        assert y.ndim == 2
        assert y.shape[0] >= 1
        assert y.shape[1] == 1
        assert all(yi == 0 or yi == 1 for yi in y.flatten())
        assert isinstance(y_hat, np.ndarray)
        assert y.shape == y_hat.shape
        assert all(0 <= yi_hat <= 1 for yi_hat in y_hat.flatten())
        assert isinstance(eps, float)
        assert 0 < eps < 0.1
        return -(
            y * np.log(np.maximum(eps, y_hat))
            + (1 - y) * np.log(np.maximum(eps, 1 - y_hat))
        ).mean()
