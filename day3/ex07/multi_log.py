from dataclasses import dataclass
from my_logistic_regression import MyLogisticRegression
import numpy as np
from utils import (
    MARKER_FAILURE,
    MARKER_SUCCESS,
    SIZE_FAILURE,
    SIZE_SUCCESS,
    denormalize,
    normalize,
    read_datasets,
    scatter_plots,
    train_test_split,
)


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


def train_models(x_train, x_test, y_train, y_test):
    models = []
    for i in range(len(LOCATIONS)):
        y_train_01 = (y_train == i).astype(int)
        y_test_01 = (y_test == i).astype(int)
        theta = np.random.normal(0, 1, (x_train.shape[1] + 1, 1))
        model = MyLogisticRegression(theta, alpha=0.1, max_iter=1000)
        model.fit_(x_train, y_train_01)
        predictions = (model.predict_(x_test) > 0.5).astype(int)
        print(
            f"Accuracy: {(predictions == y_test_01).mean():.3f} | Model: {LOCATIONS[i].name} vs Rest"
        )
        models.append(model)
    return models


def predict_ovr(x, all_thetas):
    x_stack = np.hstack([np.ones((x.shape[0], 1)), x])
    return np.argmax(x_stack @ all_thetas, axis=1).reshape((-1, 1))


def scatter_plot_1v1(ax, x1, x2, y_test, predictions):
    for i, location in enumerate(LOCATIONS):
        actual = (y_test == i).flatten()
        ax.scatter(
            x1[actual],
            x2[actual],
            color=location.color,
            marker=MARKER_SUCCESS,
            s=SIZE_SUCCESS,
            zorder=2,
            label=f"Actual {location.name}",
        )
        false_positives = ((predictions == i) & (predictions != y_test)).flatten()
        ax.scatter(
            x1[false_positives],
            x2[false_positives],
            color=location.color,
            marker=MARKER_FAILURE,
            s=SIZE_FAILURE,
            zorder=3,
            label=f"Wrongly predicted {location.name}",
        )


def main():
    x, y = read_datasets()
    x, means, stds = normalize(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, 0.75)
    models = train_models(x_train, x_test, y_train, y_test)
    all_thetas = np.column_stack([model.theta for model in models])
    predictions = predict_ovr(x_test, all_thetas)
    print(f"Full model accuracy: {(predictions == y_test).mean():.3f}")
    scatter_plots(
        denormalize(x_test, means, stds), y_test, predictions, scatter_plot_1v1
    )


if __name__ == "__main__":
    main()
