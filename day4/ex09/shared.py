from dataclasses import dataclass
from my_logistic_regression import MyLogisticRegression
import numpy as np
import pandas as pd
import sys

DEGREE = 3
X_COLUMNS = ["weight", "height", "bone_density"]
Y_COLUMN = "Origin"


@dataclass
class Location:
    name: str
    color: str


LOCATIONS = [
    Location("The flying cities of Venus", "tab:green"),
    Location("United Nations of Earth", "tab:blue"),
    Location("Mars Republic", "tab:red"),
    Location("The Asteroids' Belt colonies", "slategrey"),
]


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


def __read_dataset(filename, usecols, astype):
    try:
        data = pd.read_csv(filename, usecols=usecols).astype(astype)
        return data[usecols].values
    except Exception as e:
        print(f"Failed to read the dataset: {e}")
        sys.exit(1)


def read_datasets():
    x = __read_dataset("solar_system_census.csv", X_COLUMNS, float)
    y = __read_dataset("solar_system_census_planets.csv", [Y_COLUMN], int)
    assert len(x) == len(y)
    return __train_val_test_split(x, y, 0.6, 0.2, seed=42)


def __normalize(x):
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    return (x - means) / stds, means, stds


def __fit_normalized(x, y, *, lambda_, alpha, max_iter):
    x_norm, x_means, x_stds = __normalize(x)
    y_norm, y_means, y_stds = __normalize(y)
    model = MyLogisticRegression(
        np.zeros((x.shape[1] + 1, 1)), alpha=alpha, max_iter=max_iter, lambda_=lambda_
    )
    model.fit_(x_norm, y_norm)
    theta_norm = model.get_params_()
    model.set_params_(
        np.hstack(
            [
                y_means
                + y_stds
                * (theta_norm[0, 0] - (theta_norm[1:, 0] * x_means / x_stds).sum()),
                theta_norm[1:, 0] * y_stds / x_stds,
            ]
        ).reshape((-1, 1))
    )
    return model


def train_ovr(x_train, x_test, y_train, y_test, *, lambda_, alpha, max_iter):
    all_thetas = []
    for i in range(len(LOCATIONS)):
        y_train_01 = (y_train == i).astype(int)
        y_test_01 = (y_test == i).astype(int)
        model = __fit_normalized(
            x_train, y_train_01, lambda_=lambda_, alpha=alpha, max_iter=max_iter
        )
        predictions = (model.predict_(x_test) > 0.5).astype(int)
        print(
            f"accuracy: {(predictions == y_test_01).mean():.3f} | lambda={lambda_:.1f} | model: {LOCATIONS[i].name} vs Rest"
        )
        all_thetas.append(model._theta)
    return np.column_stack(all_thetas)


def predict_ovr(x, model):
    x_stack = np.hstack([np.ones((x.shape[0], 1)), x])
    return np.argmax(x_stack @ model, axis=1).reshape((-1, 1))
