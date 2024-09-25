import numpy as np
from vec_log_gradient import vec_log_gradient


def test_valid():
    y1 = np.array([1]).reshape((-1, 1))
    x1 = np.array([4]).reshape((-1, 1))
    theta1 = np.array([[2], [0.5]])
    assert np.isclose(
        vec_log_gradient(x1, y1, theta1), np.array([[-0.01798621], [-0.07194484]])
    ).all()
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    assert np.isclose(
        vec_log_gradient(x2, y2, theta2), np.array([[0.3715235], [3.25647547]])
    ).all()
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    assert np.isclose(
        vec_log_gradient(x3, y3, theta3),
        np.array(
            [[-0.55711039], [-0.90334809], [-2.01756886], [-2.10071291], [-3.27257351]]
        ),
    ).all()


def test_invalid():
    x = np.array(
        [
            [-6, -7, -9],
            [13, -2, 14],
            [-7, 14, -1],
            [-8, -4, 6],
            [-5, -9, 6],
            [1, -5, 11],
            [9, -11, 8],
        ]
    )
    y = np.array([1, 1, 1, 1, 0, 0, 1]).reshape((-1, 1))
    assert vec_log_gradient(x, y, np.array([3, 0.5, -6]).reshape((-1, 1))) is None
    assert vec_log_gradient(x, y, np.array([1, 0, 0, 0, 0]).reshape((-1, 1))) is None
    assert (
        vec_log_gradient(x, y[:, 1:], np.array([1, 2, 3, 4]).reshape((-1, 1))) is None
    )
    assert (
        vec_log_gradient(
            x,
            np.array([1, 1, 1, 1, 0, 0.5, 1]).reshape((-1, 1)),
            np.array([1, 2, 3, 4]).reshape((-1, 1)),
        )
        is None
    )
