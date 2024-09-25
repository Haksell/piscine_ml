import argparse
from my_logistic_regression import MyLogisticRegression
import numpy as np
from utils import (
    denormalize,
    normalize,
    read_datasets,
    scatter_plots,
    train_test_split,
    MARKER_SUCCESS,
    SIZE_SUCCESS,
    MARKER_FAILURE,
    SIZE_FAILURE,
)


COLOR_POSITIVE = "tab:green"
COLOR_NEGATIVE = "tab:red"


def parse_args():
    parser = argparse.ArgumentParser(description="asfsa")
    parser.add_argument(
        "--zipcode",
        type=str,
        choices=list("0123"),
        help="Zipcode of the planet.",
        required=True,
    )
    return int(parser.parse_args().zipcode)


def scatter_plot_1v1(ax, x1, x2, y_test, predictions):
    true_negatives = ((y_test == 0) & (predictions == 0)).flatten()
    false_positives = ((y_test == 0) & (predictions == 1)).flatten()
    false_negatives = ((y_test == 1) & (predictions == 0)).flatten()
    true_positives = ((y_test == 1) & (predictions == 1)).flatten()
    ax.scatter(
        x1[true_negatives],
        x2[true_negatives],
        color=COLOR_NEGATIVE,
        marker=MARKER_SUCCESS,
        s=SIZE_SUCCESS,
        zorder=2,
        label="True Negatives",
    )
    ax.scatter(
        x1[false_positives],
        x2[false_positives],
        color=COLOR_POSITIVE,
        marker=MARKER_FAILURE,
        s=SIZE_FAILURE,
        zorder=2,
        label="False Positives",
    )
    ax.scatter(
        x1[false_negatives],
        x2[false_negatives],
        color=COLOR_NEGATIVE,
        marker=MARKER_FAILURE,
        s=SIZE_FAILURE,
        zorder=2,
        label="False Negatives",
    )
    ax.scatter(
        x1[true_positives],
        x2[true_positives],
        color=COLOR_POSITIVE,
        marker=MARKER_SUCCESS,
        s=SIZE_SUCCESS,
        zorder=2,
        label="True Positives",
    )


def main():
    np.random.seed(42)
    zipcode = parse_args()
    x, y = read_datasets()
    y = (y == zipcode).astype(int)
    x, means, stds = normalize(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, 0.75)
    # random theta because otherwise we get 98% accuracy in one step
    theta = np.random.normal(0, 1, (x_train.shape[1] + 1, 1))
    model = MyLogisticRegression(theta, alpha=0.1, max_iter=100)
    model.fit_(x_train, y_train)
    predictions = (model.predict_(x_test) > 0.5).astype(int)
    print("Accuracy:", (predictions == y_test).mean())
    scatter_plots(
        denormalize(x_test, means, stds), y_test, predictions, scatter_plot_1v1
    )


if __name__ == "__main__":
    main()
