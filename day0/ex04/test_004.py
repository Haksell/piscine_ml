from predict_ import predict_
import numpy as np


def test_predict_valid():
    x1 = np.arange(1, 6)
    x2 = x1.reshape(5, 1)
    assert np.array_equal(
        predict_(x1, np.array([[5], [0]])),
        np.array([[5], [5], [5], [5], [5]]),
    )
    assert np.array_equal(
        predict_(x2, np.array([[0], [1]])),
        np.array([[1], [2], [3], [4], [5]]),
    )
    assert np.array_equal(
        predict_(x1, np.array([[5], [3]])),
        np.array([[8], [11], [14], [17], [20]]),
    )
    assert np.array_equal(
        predict_(x2, np.array([[-3], [1]])),
        np.array([[-2], [-1], [0], [1], [2]]),
    )
    assert np.array_equal(
        predict_(x1, np.array([[2, 3]])),
        np.array([[5], [8], [11], [14], [17]]),
    )
    assert np.array_equal(
        predict_(x1, np.array([2, 3])), np.array([[5], [8], [11], [14], [17]])
    )


def test_predict_invalid():
    x = np.arange(1, 7)
    assert predict_(x.reshape((2, 3)), np.array([[2], [3]])) is None
    assert predict_(x.reshape((1, 6)), np.array([[2], [3]])) is None
