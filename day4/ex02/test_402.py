from linear_loss_reg import reg_loss_
import numpy as np

y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))


def test_valid():
    assert np.isclose(reg_loss_(y, y_hat, theta, 0), 0.5178571428571429)
    assert np.isclose(reg_loss_(y, y_hat, theta, 0.05), 0.5511071428571429)
    assert np.isclose(reg_loss_(y, y_hat, theta, 0.5), 0.8503571428571429)
    assert np.isclose(reg_loss_(y, y_hat, theta, 0.9), 1.116357142857143)


def test_invalid():
    assert reg_loss_(y.flatten(), y_hat, theta, 0.1) is None
    assert reg_loss_(y, y_hat, theta, -0.1) is None
    assert reg_loss_(y, y_hat, np.array([[]]), 0.1) is None
    assert reg_loss_(y, y_hat, np.array([1, 2, 3]), 0.1) is None
