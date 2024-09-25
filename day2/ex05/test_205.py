from mylinearregression import MyLinearRegression
import numpy as np

x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])


def test_day1_lr1():
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


def test_day1_lr2():
    lr2 = MyLinearRegression(np.array([[1], [1]]), 5e-8, 1500000)
    lr2.fit_(x, y)
    y_hat = lr2.predict_(x)
    assert np.isclose(lr2.theta, np.array([[1.40709365], [1.1150909]])).all()
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


def test_day2():
    X = np.array(
        [[1.0, 1.0, 2.0, 3.0], [5.0, 8.0, 13.0, 21.0], [34.0, 55.0, 89.0, 144.0]]
    )
    Y = np.array([[23.0], [48.0], [218.0]])
    mylr = MyLinearRegression([[1.0], [1.0], [1.0], [1.0], [1]])
    y_hat = mylr.predict_(X)
    assert np.isclose(y_hat, np.array([[8.0], [48.0], [323.0]])).all()
    assert np.isclose(
        mylr.loss_elem_(Y, y_hat), np.array([[225.0], [0.0], [11025.0]])
    ).all()
    assert np.isclose(mylr.loss_(Y, y_hat), 1875.0)
    mylr.alpha = 1.6e-4
    mylr.max_iter = 200000
    mylr.fit_(X, Y)
    y_hat = mylr.predict_(X)
    assert np.isclose(
        mylr.theta, np.array([[18.188], [2.767], [-0.374], [1.392], [0.017]]), atol=1e-3
    ).all()
    assert np.isclose(y_hat, np.array([[23.417], [47.489], [218.065]]), atol=1e-3).all()
    assert np.isclose(
        mylr.loss_elem_(Y, y_hat), np.array([[0.174], [0.260], [0.004]]), atol=1e-3
    ).all()
    assert np.isclose(mylr.loss_(Y, y_hat), 0.0732, atol=1e-4).all()
