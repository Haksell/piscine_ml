from gradient import gradient
import numpy as np


def test_valid():
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
    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    assert np.isclose(
        gradient(x, y, np.array([0, 3, 0.5, -6]).reshape((-1, 1))),
        np.array([[-33.71428571], [-37.35714286], [183.14285714], [-393.0]]),
    ).all()
    assert np.isclose(
        gradient(x, y, np.array([0, 0, 0, 0]).reshape((-1, 1))),
        np.array([[-0.71428571], [0.85714286], [23.28571429], [-26.42857143]]),
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
    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    assert gradient(x, y, np.array([3, 0.5, -6]).reshape((-1, 1))) is None
    assert gradient(x, y, np.array([1, 0, 0, 0, 0]).reshape((-1, 1))) is None
    assert gradient(x, y[:, 1:], np.array([1, 2, 3, 4]).reshape((-1, 1))) is None
