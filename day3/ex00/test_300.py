import numpy as np
from sigmoid import sigmoid_


def test_valid():
    assert (np.array([[]]) == np.array([[]])).all()
    assert np.isclose(
        sigmoid_(np.array([[-4]])), np.array([[0.01798620996209156]])
    ).all()
    assert np.isclose(sigmoid_(np.array([[2]])), np.array([[0.8807970779778823]])).all()
    assert np.isclose(
        sigmoid_(np.array([[-4], [2], [0], [-1000], [1000]])),
        np.array([[0.01798620996209156], [0.8807970779778823], [0.5], [0], [1]]),
    ).all()


def test_invalid():
    assert sigmoid_(np.array(-4)) is None
    assert sigmoid_(np.array([-4])) is None
    assert sigmoid_(np.array([-4, 2])) is None
    assert sigmoid_(np.array([[-4, 2]])) is None
