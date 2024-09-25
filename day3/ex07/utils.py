from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys

X_COLUMNS = ["weight", "height", "bone_density"]
Y_COLUMN = "Origin"
MARKER_SUCCESS = "o"
SIZE_SUCCESS = 40
MARKER_FAILURE = "x"
SIZE_FAILURE = 70


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
    return x, y


def train_test_split(x, y, proportion):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 2
    assert isinstance(y, np.ndarray)
    assert y.shape == (x.shape[0], 1)
    assert isinstance(proportion, float)
    assert 0 <= proportion <= 1
    n = x.shape[0]
    indices = np.random.permutation(n)
    train_size = int(np.floor(n * proportion))
    x_train = x[indices[:train_size]]
    x_test = x[indices[train_size:]]
    y_train = y[indices[:train_size]]
    y_test = y[indices[train_size:]]
    return x_train, x_test, y_train, y_test


def normalize(x):
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    return (x - means) / stds, means, stds


def denormalize(x, means, stds):
    return (x * stds) + means


def scatter_plots(x_test, y_test, predictions, scatter_plot_1v1):
    fig = plt.figure(figsize=(10, 10))
    fig.canvas.manager.set_window_title("day3/ex07")
    for i in range(3):
        j = (i + 1) % 3
        column1 = X_COLUMNS[i]
        column2 = X_COLUMNS[j]
        ax = plt.subplot2grid((2, 2), divmod(i, 2))
        ax.grid(True, zorder=1)
        scatter_plot_1v1(
            ax,
            x_test[:, i],
            x_test[:, j],
            y_test,
            predictions,
        )
        ax.set_title(f"{column1} vs {column2}")
        ax.set_xlabel(column1)
        ax.set_ylabel(column2)
        if i == 0:
            legend = fig.legend(loc="lower right")
            legend_box = legend.get_window_extent(fig.canvas.get_renderer())
            legend.set_bbox_to_anchor(
                (
                    0.75 + legend_box.width / (2 * fig.bbox.width),
                    0.25 - legend_box.height / (2 * fig.bbox.height),
                )
            )

    plt.tight_layout()
    plt.show()
