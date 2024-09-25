import numpy as np
from prediction import poly_predict, simple_predict


def test_simple_predict_valid():
    x = np.arange(1, 6)
    assert np.array_equal(
        simple_predict(x, np.array([5, 0])), np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    )
    assert np.array_equal(
        simple_predict(x, np.array([0, 1])), np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    )
    assert np.array_equal(
        simple_predict(x, np.array([5, 3])), np.array([8.0, 11.0, 14.0, 17.0, 20.0])
    )
    assert np.array_equal(
        simple_predict(x, np.array([-3, 1])), np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    )


def test_simple_predict_invalid():
    x = np.arange(1, 7)
    assert simple_predict(x, np.array([5])) is None
    assert simple_predict(x, np.array([1, 2, 3])) is None
    assert simple_predict(x.reshape((1, 6)), np.array([1, 2])) is None
    assert simple_predict(x.reshape((2, 3)), np.array([1, 2])) is None
    assert simple_predict(x.reshape((6, 1)), np.array([1, 2])) is None
    assert simple_predict(42, np.array([1, 2])) is None


def test_poly_predict():
    x = np.arange(1, 6)
    assert np.array_equal(poly_predict(x, np.array([5, 0])), np.array([5, 5, 5, 5, 5]))
    assert np.array_equal(
        poly_predict(x.reshape((5, 1)), np.array([0, 1])),
        np.array([1, 2, 3, 4, 5]).reshape((5, 1)),
    )
    assert np.array_equal(
        poly_predict(x.reshape((1, 5)), np.array([5, 3])),
        np.array([8, 11, 14, 17, 20]).reshape((1, 5)),
    )
    assert np.array_equal(
        poly_predict(x, np.array([-3, 1])), np.array([-2, -1, 0, 1, 2])
    )
    assert np.array_equal(poly_predict(np.array(42), np.array([2, 3])), np.array(128))
    assert np.array_equal(poly_predict(np.array([]), np.array([2, 3])), np.array([]))
    assert np.array_equal(
        poly_predict(np.array([[1, 2], [3, 4]]), np.array([2, 3])),
        np.array([[5, 8], [11, 14]]),
    )
    assert np.array_equal(
        poly_predict(np.array([[[1, 2, 3]]]), np.array([2, 3])),
        np.array([[[5, 8, 11]]]),
    )
    assert np.array_equal(poly_predict(np.arange(1, 6), np.array([2])), np.full(5, 2))
    assert np.array_equal(
        poly_predict(np.arange(1, 6), np.array([2, 3, 4])),
        np.array([9, 24, 47, 78, 117]),
    )
