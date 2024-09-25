from dataclasses import dataclass
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression
import numpy as np
import pandas as pd
import sys


@dataclass
class Column:
    actual_color: str
    predicted_color: str
    xlabel: str


X_COLUMNS = {
    "Age": Column("#080F7D", "#2D93FA", "x₁: age (in years)"),
    "Thrust_power": Column("#308A59", "#80FC38", "x₂: thrust power (in 10km/s)"),
    "Terameters": Column(
        "#9424CF", "#ED88EB", "x₃: distance totalizer value of spacecraft (in Tmeters)"
    ),
}
Y_COLUMN = "Sell_price"


def read_dataset():
    try:
        data = pd.read_csv("spacecraft_data.csv")
        assert sorted(data.columns) == sorted([*X_COLUMNS, Y_COLUMN])
        return data[list(X_COLUMNS)], data[Y_COLUMN]
    except Exception:
        print("Failed to read the dataset")
        sys.exit(1)


def plot_linear_regression(
    ax,
    x,
    y,
    y_hat,
    col,
    title_prefix,
):
    ax.grid(True, zorder=1)
    ax.scatter(
        x, y, color=X_COLUMNS[col].actual_color, label="Sell price", s=30, zorder=2
    )
    ax.scatter(
        x,
        y_hat,
        color=X_COLUMNS[col].predicted_color,
        label="Predicted sell price",
        s=10,
        zorder=3,
    )
    ax.set_title(f"{title_prefix} {col}")
    ax.set_xlabel(X_COLUMNS[col].xlabel)
    ax.set_ylabel("y: sell price (in keuros)")
    ax.legend()


def fit_normalized(x, y, *, alpha=0.1, max_iter=300):
    def normalize(data):
        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape((-1, 1))
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        return data, (data - mean) / std, mean, std

    x, x_norm, x_mean, x_std = normalize(x)
    y, y_norm, y_mean, y_std = normalize(y)
    lr_normalized = MyLinearRegression(
        np.zeros((x.shape[1] + 1, 1)), alpha=alpha, max_iter=max_iter
    )
    lr_normalized.fit_(x_norm, y_norm)
    return (
        x,
        y,
        MyLinearRegression(
            np.hstack(
                [
                    y_mean
                    + y_std
                    * (
                        lr_normalized.theta[0, 0]
                        - (lr_normalized.theta[1:, 0] * x_mean / x_std).sum()
                    ),
                    lr_normalized.theta[1:, 0] * y_std / x_std,
                ]
            ).reshape((-1, 1))
        ),
    )


def multivariate(x, y, cols, axes):
    assert len(cols) == len(axes)

    def format_equation(theta, col):
        return f"{'+' if theta >= 0 else '-'} {abs(theta):.3f} * {col}"

    x, y, model = fit_normalized(x[cols], y)
    print(
        f"Model: {model.theta[0, 0]:.3f} {' '.join(format_equation(theta, col) for theta,col in zip(model.theta[1:,].flatten(), cols))}"
    )
    predictions = model.predict_(x)
    print(
        f"MSE: {model.mse_(y, predictions):.3f}",
    )
    for i, col in enumerate(cols):
        plot_linear_regression(
            axes[i],
            x[:, i : i + 1],
            y,
            predictions,
            col,
            "Univariate" if len(cols) == 1 else "Multivariate",
        )


def main():
    x, y = read_dataset()
    fig, (univariate_axes, multivariate_axes) = plt.subplots(
        2, len(X_COLUMNS), figsize=(20, 10)
    )
    fig.canvas.manager.set_window_title("day2/ex06")
    for i, col in enumerate(X_COLUMNS):
        multivariate(x, y, [col], univariate_axes[i : i + 1])
        print("=" * 73)
    multivariate(x, y, list(X_COLUMNS), multivariate_axes)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
