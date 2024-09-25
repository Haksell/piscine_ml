import matplotlib.pyplot as plt
from multivariate_linear_model import fit_normalized
import numpy as np
import pandas as pd
from polynomial_model import add_polynomial_features
import sys

MAX_POWER = 6
TITLE = "day2/ex08"


def read_dataset():
    try:
        data = pd.read_csv("are_blue_pills_magics.csv")
        x = np.array(data["Micrograms"]).reshape(-1, 1)
        y = np.array(data["Score"]).reshape(-1, 1)
        return x, y
    except Exception:
        print("Failed to read the dataset")
        sys.exit(1)


def plot_linear_regression(x_poly, y, model, power, ax):
    ax.grid(True)
    x = x_poly[:, 0]
    x_min = x.min() - 1
    x_max = x.max() + 1
    ax.set_xlim((x_min, x_max))
    ax.scatter(x, y, color="tab:blue", label="Actual scores")
    line_x = np.linspace(x_min, x_max, 100).reshape((-1, 1))
    line_y = model.predict_(add_polynomial_features(line_x, power))
    ax.plot(line_x, line_y, "--", color="tab:green", label="Predicted scores")
    ax.set_title(f"Polynomial of degree {power}")
    ax.set_xlabel("Quantity of blue pill (in micrograms)")
    ax.set_ylabel("Space driving score")
    ax.legend()


def compute_mse(x_poly, y, model):
    y_hat = model.predict_(x_poly)
    mse = model.mse_(y, y_hat)
    print(
        f"Model: {' + '.join(f'{theta:.3f} * x^{i}' for i, theta in enumerate(model.theta.flatten())).replace('+ -', '- ').replace(' * x^0', '').replace('* x^1', '* x')}",
        f"(MSE = {mse:.3f})",
    )
    return mse


def bar_plot(mses):
    fig = plt.figure(figsize=(10, 6))
    fig.canvas.manager.set_window_title(TITLE)
    plt.bar(range(1, MAX_POWER + 1), mses)
    plt.ylim((min(mses) - 2, max(mses) + 2))
    plt.xlabel("Polynomial Power")
    plt.ylabel("Mean Squared Error")
    plt.title("MSE for Different Polynomial Powers")
    plt.xticks(range(1, MAX_POWER + 1))
    plt.show()


def main():
    x, y = read_dataset()
    fig, axes = plt.subplots(2, (MAX_POWER + 1) >> 1, figsize=(18, 9))
    axes = axes.flatten()
    fig.canvas.manager.set_window_title(TITLE)
    mses = []
    for power in range(1, MAX_POWER + 1):
        x_poly = add_polynomial_features(x, power)
        _, _, model = fit_normalized(x_poly, y, alpha=0.2, max_iter=200_000)
        plot_linear_regression(x_poly, y, model, power, axes[power - 1])
        mses.append(compute_mse(x_poly, y, model))
    plt.tight_layout()
    plt.show()
    bar_plot(mses)


if __name__ == "__main__":
    main()
