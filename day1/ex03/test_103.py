from my_linear_regression import MyLinearRegression
import numpy as np

x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])


def test_lr1():
    lr1 = MyLinearRegression(np.array([[2], [0.7]]))
    y_hat = lr1.predict_(x)
    assert np.isclose(
        y_hat,
        np.array(
            (
                [
                    [10.74695094],
                    [17.05055804],
                    [24.08691674],
                    [36.24020866],
                    [42.25621131],
                ]
            )
        ),
    ).all()
    assert np.isclose(
        lr1.loss_elem_(y, y_hat),
        np.array(
            [
                [710.45867381],
                [364.68645485],
                [469.96221651],
                [108.97553412],
                [299.37111101],
            ]
        ),
    ).all()
    assert np.isclose(lr1.loss_(y, y_hat), 195.34539903032385)


def test_lr2():
    lr2 = MyLinearRegression(np.array([[1], [1]]), 5e-8, 1500000)
    lr2.fit_(x, y)
    y_hat = lr2.predict_(x)
    assert np.isclose(lr2.thetas, np.array([[1.40709365], [1.1150909]])).all()
    assert np.isclose(
        y_hat,
        np.array(
            [[15.3408728], [25.38243697], [36.59126492], [55.95130097], [65.53471499]]
        ),
    ).all()
    assert np.isclose(
        lr2.loss_elem_(y, y_hat),
        np.array(
            [
                [486.66604863],
                [115.88278416],
                [84.16711596],
                [85.96919719],
                [35.71448348],
            ]
        ),
    ).all()
    assert np.isclose(lr2.loss_(y, y_hat), 80.83996294128525)
