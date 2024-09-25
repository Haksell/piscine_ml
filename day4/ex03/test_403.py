from logistic_loss_reg import reg_log_loss_
import numpy as np


y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
y_hat = np.array([0.9, 0.79, 0.12, 0.04, 0.89, 0.93, 0.01]).reshape((-1, 1))
theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))


def test_valid():
    assert np.isclose(reg_log_loss_(y, y_hat, theta, 0.5), 0.43377043716475955)
    assert np.isclose(reg_log_loss_(y, y_hat, theta, 0.05), 0.13452043716475953)
    assert np.isclose(reg_log_loss_(y, y_hat, theta, 0.9), 0.6997704371647596)


def test_invalid():
    assert reg_log_loss_(y.flatten(), y_hat, theta, 0.1) is None
    assert reg_log_loss_(y, y_hat, theta, -0.1) is None
    assert reg_log_loss_(y, y_hat, np.array([[]]), 0.1) is None
    assert reg_log_loss_(y, y_hat, np.array([1, 2, 3]), 0.1) is None
