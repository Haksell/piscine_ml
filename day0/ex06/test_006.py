import numpy as np
from predict_ import predict_
from loss import loss_, loss_elem_


def is_close(x, y, eps=1e-3):
    return abs(x - y) <= eps


def test_loss_valid():
    x1 = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    theta1 = np.array([[2.0], [4.0]])
    y_hat1 = predict_(x1, theta1)
    y1 = np.array([[2.0], [7.0], [12.0], [17.0], [22.0]])
    assert np.array_equal(loss_elem_(y1, y_hat1), np.array([[0], [1], [4], [9], [16]]))
    assert loss_(y1, y_hat1) == 3
    x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
    theta2 = np.array([[0.0], [1.0]]).reshape(-1, 1)
    y_hat2 = predict_(x2, theta2)
    y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)
    assert is_close(loss_(y2, y_hat2), 15 / 7)
    assert loss_(y2, y2) == 0
    assert loss_(np.arange(1, 6), np.arange(2, 7)) == 0.5


def test_loss_invalid():
    assert loss_(np.arange(1, 6), np.arange(1, 7)) is None
    assert loss_elem_(np.arange(1, 6), np.arange(1, 7)) is None
    assert loss_(np.arange(1, 9).reshape((2, 4)), np.full((2, 4), 42)) is None
    assert loss_elem_(np.arange(1, 9).reshape((2, 4)), np.full((2, 4), 42)) is None
    assert loss_(np.array([]), np.array([])) is None
    assert loss_elem_(np.array([]), np.array([])) is None
