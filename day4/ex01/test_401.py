from l2_reg import iterative_l2, l2
import numpy as np

x1 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
y1 = 911
x2 = np.array([3, 0.5, -6]).reshape((-1, 1))
y2 = 36.25


def test_iterative_l2():
    assert iterative_l2(x1) == y1
    assert iterative_l2(x2) == y2


def test_l2():
    assert l2(x1) == y1
    assert l2(x2) == y2


def test_invalid():
    x = np.array([2, 14, -13, 5, 12, 4, -19])
    assert iterative_l2(x) is None
    assert l2(x) is None
    x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1)).tolist()
    assert iterative_l2(x) is None
    assert l2(x) is None
    x = np.array([]).reshape((-1, 1))
    assert iterative_l2(x) is None
    assert l2(x) is None
    x = np.arange(6).reshape((3, 2))
    assert iterative_l2(x) is None
    assert l2(x) is None
    x = np.arange(6).reshape((2, 3))
    assert iterative_l2(x) is None
    assert l2(x) is None
    x = np.arange(6).reshape((1, 6))
    assert iterative_l2(x) is None
    assert l2(x) is None
