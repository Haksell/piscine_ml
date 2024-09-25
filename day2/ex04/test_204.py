from fit import fit_
import numpy as np
from prediction import predict_

x = np.array([[0.2, 2.0, 20.0], [0.4, 4.0, 40.0], [0.6, 6.0, 60.0], [0.8, 8.0, 80.0]])
y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
theta = np.array([[42.0], [1.0], [1.0], [1.0]])


def test_fit_valid():
    theta2 = fit_(x, y, theta, alpha=0.0005, max_iter=42000)
    assert np.isclose(
        theta2, np.array([[41.99], [0.97], [0.77], [-1.20]]), atol=1e-2
    ).all()
    assert np.isclose(
        predict_(x, theta2),
        np.array([[19.5992], [-2.8003], [-25.1999], [-47.5996]]),
        atol=1e-4,
    ).all()


def test_fit_invalid():
    assert fit_(x, y, theta, alpha=-0.001, max_iter=1500000) is None
    assert fit_(x, y, theta, alpha=0.1, max_iter=-1500000) is None
    assert fit_(x, y, np.array([[1.0], [1.0], [1.0]]), alpha=0.1, max_iter=1500) is None
