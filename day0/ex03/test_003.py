import numpy as np
from tools import add_intercept


def test_add_intercept():
    assert np.array_equal(
        add_intercept(np.arange(1, 10).reshape((3, 3))),
        np.array([[1, 1, 2, 3], [1, 4, 5, 6], [1, 7, 8, 9]]),
    )
    assert np.array_equal(
        add_intercept(np.arange(1, 6).reshape((5, 1))),
        np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]]),
    )
    assert np.array_equal(
        add_intercept(np.arange(1, 6)),
        np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]]),
    )
    assert np.array_equal(
        add_intercept(np.array(42)),
        np.array([[1, 42]]),
    )
