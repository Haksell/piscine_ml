import numpy as np
from data_spliter import data_spliter
from itertools import product
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from multivariate_linear_model import fit_normalized
import pandas as pd
from polynomial_model import add_multi_polynomial_features
import sys

TITLE = "day2/ex10"
X_COLUMNS = ["weight", "prod_distance", "time_delivery"]
Y_COLUMN = "target"
MAX_POWER = 4
ALPHA = 0.2
ALL_MAX_ITER = 1000
BEST_MAX_ITER = 100_000


def read_dataset():
    try:
        data = pd.read_csv("space_avocado.csv", usecols=[*X_COLUMNS, Y_COLUMN]).astype(
            "float64"
        )
        return data[X_COLUMNS].values, data[[Y_COLUMN]].values
    except Exception as e:
        print(f"Failed to read the dataset: {e}")
        sys.exit(1)


def fit_polynormamultivariate(
    x_train, x_test, y_train, y_test, max_powers, *, alpha, max_iter
):
    x_train = add_multi_polynomial_features(x_train, max_powers)
    x_test = add_multi_polynomial_features(x_test, max_powers)
    _, _, model = fit_normalized(x_train, y_train, alpha=alpha, max_iter=max_iter)
    y_hat = model.predict_(x_test)
    mse = model.mse_(y_hat, y_test)
    return model, mse


def bar_plot(best_results):
    fig = plt.figure(figsize=(10, 6))
    fig.canvas.manager.set_window_title(TITLE)
    x = list(best_results.keys())
    y = [mse for mse, _ in best_results.values()]
    plt.bar(x, y)
    plt.xlabel("Sum of Powers")
    plt.ylabel("Mean Squared Error")
    plt.title(
        f"MSE for Different Model Complexities (alpha={ALPHA}, max_iter={ALL_MAX_ITER})"
    )
    plt.xticks(x)
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{round(x / 1e6)}M")
    )
    plt.show()


def scatter_plot_3d(x_test, y_test, y_hat, feature_names):
    assert x_test.shape[1] == len(feature_names) == 3
    assert y_test.shape == y_hat.shape == (len(x_test), 1)

    def gen_subplot(subplot_pos, y, cmap, cbar_label):
        ax = fig.add_subplot(subplot_pos, projection="3d")
        ax.view_init(30, 30)
        plt.colorbar(
            ax.scatter(x0, x1, zeros, c=y, cmap=cmap),
            ax=ax,
            location="bottom",
            fraction=0.035,
            pad=0.01,
        ).set_label(cbar_label)
        ax.scatter(x0, zeros, x2, c=y, cmap=cmap)
        ax.scatter(zeros, x1, x2, c=y, cmap=cmap)
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_zlabel(feature_names[2])
        ax.set_xlim([0, max(x0) * 1.1])
        ax.set_ylim([0, max(x1) * 1.1])
        ax.set_zlim([0, max(x2) * 1.1])
        ax.xaxis._axinfo["juggled"] = (0, 0, 0)
        ax.yaxis._axinfo["juggled"] = (1, 1, 1)
        ax.zaxis._axinfo["juggled"] = (2, 2, 2)

    x0 = x_test[:, 0]
    x1 = x_test[:, 1]
    x2 = x_test[:, 2]
    zeros = np.zeros(len(x_test))
    fig = plt.figure(figsize=(16, 8))
    fig.canvas.manager.set_window_title(TITLE)
    gen_subplot(121, y_test, plt.cm.Reds, "Actual target")
    gen_subplot(122, y_hat, plt.cm.Blues, "Predicted target")
    plt.tight_layout()
    plt.show()


def print_model(model, max_powers):
    def format_theta(theta):
        return f"{theta:#.3g}"

    i = 1
    output = [format_theta(model.theta[0, 0])]
    for col, max_power in zip(X_COLUMNS, max_powers):
        subtheta = model.theta[i : i + max_power].flatten()
        for p in range(1, max_power + 1):
            output.append(f"{format_theta(subtheta[p - 1])}*{col[0]}^{p}")
        i += max_power
    print(
        "Final model:",
        " + ".join(output)
        .replace("+ -", "- ")
        .replace("^1", "")
        .replace("e-0", "e-")
        .replace("e+0", "e")
        .replace("e+", "e"),
    )


def main():
    x, y = read_dataset()
    best_results = dict()
    print(
        f"Training {MAX_POWER+1}^{len(X_COLUMNS)}-1 = {(MAX_POWER+1)**len(X_COLUMNS)-1} models..."
    )
    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8, seed=42)
    for max_powers in product(range(0, MAX_POWER + 1), repeat=len(X_COLUMNS)):
        complexity = sum(max_powers)
        if complexity == 0:
            continue
        _, mse = fit_polynormamultivariate(
            x_train,
            x_test,
            y_train,
            y_test,
            max_powers,
            alpha=ALPHA,
            max_iter=ALL_MAX_ITER,
        )
        result = (mse, max_powers)
        best_results[complexity] = min(best_results.get(complexity, result), result)
    print("Best models by sum of powers:")
    for complexity, (mse, max_powers) in best_results.items():
        print(f"Complexity: {complexity:>2} | Model: {max_powers} | MSE: {mse:>13,.1f}")
    best_powers = min(best_results.items(), key=lambda t: t[0] * t[1][0])[1][1]
    bar_plot(best_results)
    print("Best overall model by Occam's razor:", best_powers)
    print(f"Training {best_powers} for longer...")
    model, mse = fit_polynormamultivariate(
        x_train,
        x_test,
        y_train,
        y_test,
        best_powers,
        alpha=ALPHA,
        max_iter=BEST_MAX_ITER,
    )
    print_model(model, best_powers)
    print(f"MSE: {mse:,.1f}")
    y_hat = model.predict_(add_multi_polynomial_features(x_test, best_powers))
    scatter_plot_3d(x_test, y_test, y_hat, X_COLUMNS)


if __name__ == "__main__":
    main()
