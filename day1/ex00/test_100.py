from gradient import simple_gradient
import numpy as np


def test_simple_gradient_valid():
    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape(
        (-1, 1)
    )
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape(
        (-1, 1)
    )
    assert np.isclose(
        simple_gradient(x, y, np.array([2, 0.7]).reshape((-1, 1))),
        np.array([[-19.0342574], [-586.66875564]]),
    ).all()
    assert np.isclose(
        simple_gradient(x, y, np.array([1, -0.4]).reshape((-1, 1))),
        np.array([[-57.86823748], [-2230.12297889]]),
    ).all()


def test_simple_gradient_invalid():
    x = np.array([[1], [2], [3], [4], [5]])
    y = np.array([[6], [7], [8], [9], [10]])
    assert simple_gradient(x, y, np.array([27, 42])) is None
    assert simple_gradient(x, y, np.array([[27, 42]])) is None
    assert simple_gradient(x, y, np.array([[27], [42], [69]])) is None
    assert (
        simple_gradient(x, np.arange(8).reshape((-1, 1)), np.array([[27], [42], [69]]))
        is None
    )
    assert simple_gradient(x, np.arange(5), np.array([[27], [42], [69]])) is None
    assert (
        simple_gradient(x, np.arange(5).reshape((1, -1)), np.array([[27], [42], [69]]))
        is None
    )
