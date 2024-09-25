from my_logistic_regression import MyLogisticRegression
import numpy as np


def test_subject():
    x = np.array([[1.0, 1.0, 2.0, 3.0], [5.0, 8.0, 13.0, 21.0], [3.0, 5.0, 9.0, 14.0]])
    y = np.array([[1], [0], [1]])
    thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
    mylr = MyLogisticRegression(thetas)
    y_hat = mylr.predict_(x)
    assert np.isclose(np.array([[0.99930437], [1.0], [1.0]]), y_hat).all()
    assert np.isclose(mylr.loss_(y, y_hat), 11.513157421577004)
    mylr.fit_(x, y)
    assert np.isclose(
        mylr.theta,
        np.array(
            [[2.11826435], [0.10154334], [6.43942899], [-5.10817488], [0.6212541]]
        ),
    ).all()
    y_hat = mylr.predict_(x)
    assert np.isclose(np.array([[0.57606717], [0.68599807], [0.06562156]]), y_hat).all()
    assert np.isclose(mylr.loss_(y, y_hat), 1.4779126923052268)
