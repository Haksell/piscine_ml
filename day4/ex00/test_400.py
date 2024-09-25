import numpy as np
from polynomial_model_extended import add_polynomial_features

x = np.arange(1, 11).reshape(5, 2)


def test_valid():
    assert (
        add_polynomial_features(x, 3)
        == np.array(
            [
                [1, 2, 1, 4, 1, 8],
                [3, 4, 9, 16, 27, 64],
                [5, 6, 25, 36, 125, 216],
                [7, 8, 49, 64, 343, 512],
                [9, 10, 81, 100, 729, 1000],
            ]
        )
    ).all()
    assert (
        add_polynomial_features(x, 4)
        == np.array(
            [
                [1, 2, 1, 4, 1, 8, 1, 16],
                [3, 4, 9, 16, 27, 64, 81, 256],
                [5, 6, 25, 36, 125, 216, 625, 1296],
                [7, 8, 49, 64, 343, 512, 2401, 4096],
                [9, 10, 81, 100, 729, 1000, 6561, 10000],
            ]
        )
    ).all()


def test_invalid():
    add_polynomial_features(x, 0) is None
    add_polynomial_features(x.tolist(), 3) is None
    add_polynomial_features(x.flatten(), 3) is None
    add_polynomial_features(np.zeros((0, 1)), 3) is None
