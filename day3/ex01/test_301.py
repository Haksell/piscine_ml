import numpy as np
from log_pred import logistic_predict_


def test_valid():
    assert np.isclose(
        logistic_predict_(np.array([4]).reshape((-1, 1)), np.array([[2], [0.5]])),
        np.array([[0.98201379]]),
    ).all()
    assert np.isclose(
        logistic_predict_(
            np.array([[4], [7.16], [3.2], [9.37], [0.56]]), np.array([[2], [0.5]])
        ),
        np.array(
            [[0.98201379], [0.99624161], [0.97340301], [0.99875204], [0.90720705]]
        ),
    ).all()
    assert np.isclose(
        logistic_predict_(
            np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]]),
            np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]]),
        ),
        np.array([[0.03916572], [0.00045262], [0.2890505]]),
    ).all()


def test_invalid():
    assert logistic_predict_(42, 42) is None
    assert (
        logistic_predict_(
            np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]]),
            np.array([[-2.4], [-1.5], [0.3], [-1.4]]),
        )
        is None
    )
    assert (
        logistic_predict_(
            np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]]),
            np.array([[-2.4], [-1.5], [0.3], [-1.4], [-1.4], [-1.4]]),
        )
        is None
    )
    assert (
        logistic_predict_(
            np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]]),
            np.array([[-2.4, -1.5, 0.3, -1.4, 0.7]]),
        )
        is None
    )
