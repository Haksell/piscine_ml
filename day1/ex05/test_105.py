import numpy as np
from zscore import zscore


def test_score_valid():
    assert np.isclose(
        zscore(np.array([0, 15, -9, 7, 12, 3, -21])),
        np.array(
            [
                -0.08620324,
                1.2068453,
                -0.86203236,
                0.51721942,
                0.94823559,
                0.17240647,
                -1.89647119,
            ]
        ),
    ).all()
    assert np.isclose(
        zscore(np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))),
        np.array(
            [
                0.11267619,
                1.16432067,
                -1.20187941,
                0.37558731,
                0.98904659,
                0.28795027,
                -1.72770165,
            ]
        ).reshape((-1, 1)),
    ).all()
    assert np.isclose(
        zscore(np.array([[-3.14, 3.14], [3.14, -3.14]])), np.array([[-1, 1], [1, -1]])
    ).all()
    assert np.isclose(zscore(np.array([42])), np.array([0])).all()
    assert np.isclose(zscore(np.array(3.14)), np.array(0)).all()


def test_score_invalid():
    assert zscore(np.array([["lol", "xd"], ["lol", "lol"]])) is None
    assert zscore(np.array([[], []])) is None
    assert zscore([1, 2, 3, 4]) is None
