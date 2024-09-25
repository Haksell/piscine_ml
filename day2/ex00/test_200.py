import numpy as np
from simple_predict import simple_predict

x = np.arange(1, 13).reshape((4, -1))


def test_valid():
    assert np.isclose(
        simple_predict(x, np.array([5, 0, 0, 0]).reshape((-1, 1))),
        np.array([[5.0], [5.0], [5.0], [5.0]]),
    ).all()
    assert np.isclose(
        simple_predict(x, np.array([0, 1, 0, 0]).reshape((-1, 1))),
        np.array([[1.0], [4.0], [7.0], [10.0]]),
    ).all()
    assert np.isclose(
        simple_predict(x, np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))),
        np.array([[9.64], [24.28], [38.92], [53.56]]),
    ).all()
    assert np.isclose(
        simple_predict(x, np.array([-3, 1, 2, 3.5]).reshape((-1, 1))),
        np.array([[12.5], [32.0], [51.5], [71.0]]),
    ).all()


def test_invalid():
    assert simple_predict(x, np.array([5, 0, 0]).reshape((-1, 1))) is None
    assert simple_predict(x, np.array([1, 2, 3, 4, 5]).reshape((-1, 1))) is None
    assert (
        simple_predict(x.reshape((3, 4)), np.array([5, 0, 0, 0]).reshape((-1, 1)))
        is None
    )
    assert simple_predict(x, np.array([5, 0, 0, 0])) is None
