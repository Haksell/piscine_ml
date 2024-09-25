from math import sqrt
import numpy as np
from other_losses import mse_, rmse_, mae_, r2score_
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def is_close(x, y, eps=1e-6):
    return abs(x - y) <= eps


def test_other_losses():
    x = np.array([0, 15, -9, 7, 12, 3, -21])
    y = np.array([2, 14, -13, 5, 12, 4, -19])
    assert is_close(mse_(x, y), mean_squared_error(x, y))
    assert is_close(rmse_(x, y), sqrt(mean_squared_error(x, y)))
    assert is_close(mae_(x, y), mean_absolute_error(x, y))
    assert is_close(r2score_(x, y), r2_score(x, y))
