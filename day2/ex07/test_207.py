import numpy as np
import pytest
from polynomial_model import add_multi_polynomial_features, add_polynomial_features

x_single = np.arange(1, 6).reshape(-1, 1)


def test_single_valid():
    assert (
        add_polynomial_features(x_single, 3)
        == np.array([[1, 1, 1], [2, 4, 8], [3, 9, 27], [4, 16, 64], [5, 25, 125]])
    ).all()
    assert (
        add_polynomial_features(x_single, 6)
        == np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [2, 4, 8, 16, 32, 64],
                [3, 9, 27, 81, 243, 729],
                [4, 16, 64, 256, 1024, 4096],
                [5, 25, 125, 625, 3125, 15625],
            ]
        )
    ).all()


def test_single_invalid():
    assert add_polynomial_features(x_single, 0) is None
    assert add_polynomial_features(x_single, 3.14) is None
    assert add_polynomial_features(x_single.flatten(), 3) is None
    assert add_polynomial_features(list(x_single.flatten()), 3) is None
    assert add_polynomial_features(np.zeros((3, 2)), 3) is None


x_multi = np.arange(1, 7).reshape((3, 2))


def test_multi_valid():
    assert (
        add_multi_polynomial_features(x_multi, (2, 4))
        == np.array(
            [[1, 1, 2, 4, 8, 16], [3, 9, 4, 16, 64, 256], [5, 25, 6, 36, 216, 1296]]
        )
    ).all()
    assert (
        add_multi_polynomial_features(x_multi, (0, 3))
        == np.array([[2, 4, 8], [4, 16, 64], [6, 36, 216]])
    ).all()


def test_multi_invalid():
    with pytest.raises(AssertionError):
        add_multi_polynomial_features(x_multi, 3)
    with pytest.raises(AssertionError):
        add_multi_polynomial_features(x_multi, (3,))
    with pytest.raises(AssertionError):
        add_multi_polynomial_features(x_multi, (0, 0))
    with pytest.raises(AssertionError):
        add_multi_polynomial_features(x_multi, (2, 3.14))
    with pytest.raises(AssertionError):
        add_multi_polynomial_features(x_multi, (2, 3, 4))
