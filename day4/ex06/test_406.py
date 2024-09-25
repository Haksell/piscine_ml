import numpy as np
import pytest
from ridge import MyRidge


def test_lambda0():
    X = np.array(
        [[1.0, 1.0, 2.0, 3.0], [5.0, 8.0, 13.0, 21.0], [34.0, 55.0, 89.0, 144.0]]
    )
    Y = np.array([[23.0], [48.0], [218.0]])
    ridge = MyRidge([[1.0], [1.0], [1.0], [1.0], [1]], lambda_=0)
    y_hat = ridge.predict_(X)
    assert np.isclose(y_hat, np.array([[8.0], [48.0], [323.0]])).all()
    assert np.isclose(
        ridge.loss_elem_(Y, y_hat), np.array([[225.0], [0.0], [11025.0]])
    ).all()
    assert np.isclose(ridge.loss_(Y, y_hat), 1875.0)
    ridge.alpha = 1.6e-4
    ridge.max_iter = 200000
    ridge.fit_(X, Y)
    y_hat = ridge.predict_(X)
    assert np.isclose(
        ridge._theta,
        np.array([[18.188], [2.767], [-0.374], [1.392], [0.017]]),
        atol=1e-3,
    ).all()
    assert np.isclose(y_hat, np.array([[23.417], [47.489], [218.065]]), atol=1e-3).all()
    assert np.isclose(
        ridge.loss_elem_(Y, y_hat), np.array([[0.174], [0.260], [0.004]]), atol=1e-3
    ).all()
    assert np.isclose(ridge.loss_(Y, y_hat), 0.0732, atol=1e-4).all()


def test_get_set():
    ridge = MyRidge(np.ones((5, 1)))
    assert (ridge.get_params_() == np.ones((5, 1))).all()
    ridge.set_params_(np.zeros((5, 1)))
    assert (ridge.get_params_() == np.zeros((5, 1))).all()
    with pytest.raises(AssertionError):
        ridge.set_params_([[1], [2, 3]])
