from reg_linear_grad import reg_linear_grad, vec_reg_linear_grad
import numpy as np

x = np.array(
    [
        [-6, -7, -9],
        [13, -2, 14],
        [-7, 14, -1],
        [-8, -4, 6],
        [-5, -9, 6],
        [1, -5, 11],
        [9, -11, 8],
    ]
)
y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
theta = np.array([[7.01], [3], [10.5], [-6]])
out_1_0 = np.array([[-60.99], [-195.64714286], [863.46571429], [-644.52142857]])
out_0_5 = np.array([[-60.99], [-195.86142857], [862.71571429], [-644.09285714]])
out_0_0 = np.array([[-60.99], [-196.07571429], [861.96571429], [-643.66428571]])


def test_reg_linear_grad():
    assert np.isclose(reg_linear_grad(y, x, theta, 1), out_1_0).all()
    assert np.isclose(reg_linear_grad(y, x, theta, 0.5), out_0_5).all()
    assert np.isclose(reg_linear_grad(y, x, theta, 0), out_0_0).all()


def test_vec_reg_linear_grad():
    assert np.isclose(vec_reg_linear_grad(y, x, theta, 1), out_1_0).all()
    assert np.isclose(vec_reg_linear_grad(y, x, theta, 0.5), out_0_5).all()
    assert np.isclose(vec_reg_linear_grad(y, x, theta, 0), out_0_0).all()


def test_invalid():
    assert reg_linear_grad(y, x, theta, -0.5) is None
    assert reg_linear_grad(y[1:], x, theta, 1) is None
