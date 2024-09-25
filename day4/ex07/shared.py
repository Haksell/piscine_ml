import numpy as np
import pandas as pd
from ridge import MyRidge
import sys


X_COLUMNS = ["weight", "prod_distance", "time_delivery"]
Y_COLUMN = "target"


def __train_val_test_split(x, y, p_train, p_val, *, seed=None):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 2
    assert isinstance(y, np.ndarray)
    assert y.shape == (x.shape[0], 1)
    assert isinstance(p_train, float)
    assert isinstance(p_val, float)
    assert 0 <= p_train <= 1
    assert 0 <= p_val <= 1
    assert 0 <= p_train + p_val <= 1
    if seed is not None:
        np.random.seed(seed)
    n = x.shape[0]
    indices = np.random.permutation(n)
    train_idx = int(np.floor(n * p_train))
    val_idx = int(np.floor(n * (p_train + p_val)))
    x_train = x[indices[:train_idx]]
    x_val = x[indices[train_idx:val_idx]]
    x_test = x[indices[val_idx:]]
    y_train = y[indices[:train_idx]]
    y_val = y[indices[train_idx:val_idx]]
    y_test = y[indices[val_idx:]]
    return x_train, x_val, x_test, y_train, y_val, y_test


def read_dataset():
    try:
        training_data = pd.read_csv(
            "space_avocado.csv", usecols=[*X_COLUMNS, Y_COLUMN]
        ).astype("float64")
        x = training_data[X_COLUMNS].values
        y = training_data[[Y_COLUMN]].values
        return __train_val_test_split(x, y, 0.7, 0.15, seed=42)
    except Exception as e:
        print(f"Failed to read the dataset: {e}")
        sys.exit(1)


def __normalize(x):
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    return (x - means) / stds, means, stds


def fit_normalized(x, y, *, lambda_, alpha, max_iter):
    x_norm, x_means, x_stds = __normalize(x)
    y_norm, y_means, y_stds = __normalize(y)
    ridge = MyRidge(
        np.zeros((x.shape[1] + 1, 1)), alpha=alpha, max_iter=max_iter, lambda_=lambda_
    )
    ridge.fit_(x_norm, y_norm)
    theta_norm = ridge.get_params_()
    ridge.set_params_(
        np.hstack(
            [
                y_means
                + y_stds
                * (theta_norm[0, 0] - (theta_norm[1:, 0] * x_means / x_stds).sum()),
                theta_norm[1:, 0] * y_stds / x_stds,
            ]
        ).reshape((-1, 1))
    )
    return ridge
