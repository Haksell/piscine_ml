import numpy as np
from minmax import minmax


def test_score_valid():
    assert np.isclose(
        minmax(np.array([0, 15, -9, 7, 12, 3, -21])),
        np.array(
            [0.58333333, 1.0, 0.33333333, 0.77777778, 0.91666667, 0.66666667, 0.0]
        ),
    ).all()
    assert np.isclose(
        minmax(np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))),
        np.array(
            [0.63636364, 1.0, 0.18181818, 0.72727273, 0.93939394, 0.6969697, 0.0]
        ).reshape((-1, 1)),
    ).all()
    assert np.isclose(
        minmax(np.array([[-3.14, 3.14], [3.14, -3.14]])), np.array([[0, 1], [1, 0]])
    ).all()
    assert np.isclose(minmax(np.array([42])), np.array([0.5])).all()
    assert np.isclose(minmax(np.array(3.14)), np.array(0.5)).all()


def test_score_invalid():
    assert minmax(np.array([["lol", "xd"], ["lol", "lol"]])) is None
    assert minmax(np.array([[], []])) is None
    assert minmax([1, 2, 3, 4]) is None
