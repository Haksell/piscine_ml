from math import log
from vec_log_loss import vec_log_loss_
from log_pred import logistic_predict_
import numpy as np


def test_valid():
    y1 = np.array([1]).reshape((-1, 1))
    x1 = np.array([4]).reshape((-1, 1))
    theta1 = np.array([[2], [0.5]])
    y_hat1 = logistic_predict_(x1, theta1)
    assert np.isclose(vec_log_loss_(y1, y_hat1), 0.01814992791780973)
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    y_hat2 = logistic_predict_(x2, theta2)
    assert np.isclose(vec_log_loss_(y2, y_hat2), 2.4825011602474483)
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    y_hat3 = logistic_predict_(x3, theta3)
    assert np.isclose(vec_log_loss_(y3, y_hat3), 2.9938533108607053)
    y4 = np.array([[0], [1], [1]])
    y_hat4 = np.array([[0], [1], [1]])
    assert np.isclose(vec_log_loss_(y4, y_hat4), 0)
    y5 = np.array([[0], [1], [1]])
    y_hat5 = np.array([[1], [0], [0]])
    assert np.isclose(vec_log_loss_(y5, y_hat5), 34.538776394910684)
    y6 = np.array([[0], [1], [1]])
    y_hat6 = np.array([[0.5], [0.5], [0.5]])
    assert np.isclose(vec_log_loss_(y6, y_hat6), -log(0.5))


def test_invalid():
    assert vec_log_loss_([[0], [1], [1]], [[0.5], [0.5], [0.5]]) is None
    assert vec_log_loss_(np.array([0, 1, 1]), np.array([[0.5], [0.5], [0.5]])) is None
    assert (
        vec_log_loss_(np.array([[0], [1], [1], [2]]), np.array([[0.5], [0.5], [0.5]]))
        is None
    )
    assert (
        vec_log_loss_(np.array([[0], [0.5], [1]]), np.array([[0.5], [0.5], [0.5]]))
        is None
    )
    assert (
        vec_log_loss_(np.array([[0], [2], [1]]), np.array([[0.5], [0.5], [0.5]]))
        is None
    )
    assert (
        vec_log_loss_(np.array([[0], [1], [1]]), np.array([[0.5], [-1], [0.5]])) is None
    )
    assert (
        vec_log_loss_(np.array([[0], [1], [1]]), np.array([[0.5], [2], [0.5]])) is None
    )
