from fit import fit_
import numpy as np

x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
theta = np.array([[1.0], [1.0]])


def __predict(x, theta):
    return np.hstack((np.ones((x.shape[0], 1)), x)) @ theta


def test_fit_valid():
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    assert np.isclose(theta1, np.array([[1.40709365], [1.1150909]])).all()
    assert np.isclose(
        __predict(x, theta1),
        np.array(
            [[15.3408728], [25.38243697], [36.59126492], [55.95130097], [65.53471499]]
        ),
    ).all()


def test_fit_invalid():
    assert fit_(x, y, theta, alpha=-0.001, max_iter=1500000) is None
    assert fit_(x, y, theta, alpha=0.1, max_iter=-1500000) is None
    assert fit_(x, y, np.array([[1.0], [1.0], [1.0]]), alpha=0.1, max_iter=1500) is None
    assert (
        fit_(
            x,
            np.arange(4).reshape((-1, 1)),
            theta,
            alpha=0.1,
            max_iter=1,
        )
        is None
    )
