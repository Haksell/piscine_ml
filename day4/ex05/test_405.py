from reg_logistic_grad import reg_logistic_grad, vec_reg_logistic_grad
import numpy as np

x = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
y = np.array([[0], [1], [1]])
theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
out_1_0 = np.array(
    [[-0.55711039], [-1.40334809], [-1.91756886], [-2.56737958], [-3.03924017]]
)
out_0_5 = np.array(
    [[-0.55711039], [-1.15334809], [-1.96756886], [-2.33404624], [-3.15590684]]
)
out_0_0 = np.array(
    [[-0.55711039], [-0.90334809], [-2.01756886], [-2.10071291], [-3.27257351]]
)


def test_reg_logistic_grad():
    assert np.isclose(reg_logistic_grad(y, x, theta, 1), out_1_0).all()
    assert np.isclose(reg_logistic_grad(y, x, theta, 0.5), out_0_5).all()
    assert np.isclose(reg_logistic_grad(y, x, theta, 0), out_0_0).all()


def test_vec_reg_logistic_grad():
    assert np.isclose(vec_reg_logistic_grad(y, x, theta, 1), out_1_0).all()
    assert np.isclose(vec_reg_logistic_grad(y, x, theta, 0.5), out_0_5).all()
    assert np.isclose(vec_reg_logistic_grad(y, x, theta, 0), out_0_0).all()


def test_invalid():
    assert reg_logistic_grad(y, x, theta, -0.5) is None
    assert reg_logistic_grad(y[1:], x, theta, 1) is None
