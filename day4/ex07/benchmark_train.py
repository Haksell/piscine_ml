import sys
from matplotlib import pyplot as plt
import numpy as np
import pickle
from polynomial_model_extended import add_polynomial_features
from shared import X_COLUMNS, Y_COLUMN, fit_normalized, read_dataset
import warnings


def benchmark_degree_lambda(x_train, x_val, y_train, y_val, degrees, lambdas):
    training_data = []
    for degree in degrees:
        x_train_poly = add_polynomial_features(x_train, degree)
        x_val_poly = add_polynomial_features(x_val, degree)
        for lambda_ in lambdas:
            print(
                f"\r  Training a model of degree {degree} with lambda={lambda_:.1f}...     ",
                end="\r",
            )
            ridge = fit_normalized(
                x_train_poly, y_train, lambda_=lambda_, alpha=0.2, max_iter=1000
            )
            y_hat = ridge.predict_(x_val_poly)
            mse = ridge.mse_(y_val, y_hat)
            results = (mse, degree, lambda_, ridge)
            training_data.append(results)
    return training_data


def plot_mse(training_data, degrees, lambdas):
    training_data = np.array(training_data)
    reverse_lambda = {lambda_: i for i, lambda_ in enumerate(lambdas)}
    scatter = plt.scatter(
        training_data[:, 1],
        list(map(reverse_lambda.get, training_data[:, 2])),
        c=training_data[:, 0],
        cmap=plt.cm.viridis.reversed(),
    )
    plt.xlabel("Degree")
    plt.ylabel("Lambda")
    plt.xticks(degrees)
    plt.yticks(range(len(lambdas)), labels=[f"{lb:.1f}" for lb in lambdas])
    cbar = plt.colorbar(scatter)
    cbar.set_label("MSE")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cbar.set_ticklabels([f"{round(mse/1e6)}M" for mse in cbar.get_ticks()])
    plt.show(block=False)


def plot_predictions(training_data, x_val, y_val):
    def keep_one_feature(keep_idx):
        other_idxs = [i for i in range(3) if i != keep_idx]
        x_simple = x_val.copy()
        x_simple[:, other_idxs] = x_simple[:, other_idxs].mean(axis=0)
        x_simple = add_polynomial_features(x_simple, best_degree)
        x_simple = x_simple[x_simple[:, keep_idx].argsort()]
        return x_simple

    best_degree = min(training_data)[1]
    fig = plt.figure(figsize=(9, 9))
    axes = [plt.subplot2grid((2, 2), divmod(i, 2)) for i in range(3)]
    for i, ax in enumerate(axes):
        ax.scatter(x_val[:, i], y_val, label="True values" if i == 0 else None)
    x_simple = [keep_one_feature(i) for i in range(3)]
    for _, degree, lambda_, ridge in training_data:
        if degree != best_degree:
            continue
        for i, ax in enumerate(axes):
            ax.plot(
                x_simple[i][:, i],
                ridge.predict_(x_simple[i]),
                label=f"lambda={lambda_:.1f}" if i == 0 else None,
            )
    for i, ax in enumerate(axes):
        ax.set_xlabel(X_COLUMNS[i])
        ax.set_ylabel(Y_COLUMN)
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


def main():
    x_train, x_val, _, y_train, y_val, _ = read_dataset()
    degrees = list(range(1, 5))
    lambdas = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 3, 10, 30, 100])
    training_data = benchmark_degree_lambda(
        x_train, x_val, y_train, y_val, degrees, lambdas
    )
    try:
        pickle.dump(training_data, open("models.pickle", "wb"))
    except Exception as e:
        print(f"Failed to save the training data: {e}")
        sys.exit(1)
    best_results = min(training_data)
    print(
        f"Best model: degree={best_results[1]} | lambda={best_results[2]:.1f} | MSE={best_results[0]:,.1f}"
    )
    plot_mse(training_data, degrees, lambdas)
    plot_predictions(training_data, x_val, y_val)


if __name__ == "__main__":
    main()
