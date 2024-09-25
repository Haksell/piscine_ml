import pickle
import sys
from matplotlib import pyplot as plt
import numpy as np
from other_metrics import f1_score_
from polynomial_model_extended import add_polynomial_features
from shared import DEGREE, LOCATIONS, X_COLUMNS, predict_ovr, read_datasets, train_ovr

MARKER_SUCCESS = "o"
SIZE_SUCCESS = 40
MARKER_FAILURE = "x"
SIZE_FAILURE = 70


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


def scatter_plots(x_test, y_test, predictions):
    fig = plt.figure(figsize=(10, 10))
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


def get_data():
    x_train, x_val, x_test, y_train, y_val, y_test = read_datasets()
    x_train = add_polynomial_features(np.vstack([x_train, x_val]), DEGREE)
    x_test = add_polynomial_features(x_test, DEGREE)
    y_train = np.vstack([y_train, y_val])
    return x_train, x_test, y_train, y_test


def bar_plot(models):
    x = [lbd for _, lbd, _ in models]
    y = [f1 for f1, _, _ in models]
    plt.bar(x, y, width=0.8 * (x[1] - x[0]))
    plt.xlabel("λ (Logistic Regression Parameter)")
    plt.ylabel("F1 Score")
    plt.title("F1 Score for Different Values of λ")
    plt.show(block=False)


def main():
    x_train, x_test, y_train, y_test = get_data()
    try:
        models = pickle.load(open("models.pickle", "rb"))
    except Exception as e:
        print(f"Failed to read the models: {e}")
        sys.exit(1)
    for f1_score, lambda_, _ in models:
        print(f"Pre-trained f1-score for λ={lambda_:.1f}: {f1_score:.3f}")
    bar_plot(models)
    _, best_lambda, _ = max(models)
    model = train_ovr(
        x_train, x_test, y_train, y_test, lambda_=best_lambda, alpha=0.1, max_iter=100
    )
    predictions = predict_ovr(x_test, model)
    f1_score = f1_score_(predictions, y_test)
    print(f"Final model (λ={best_lambda:.1f}) f1-score: {f1_score:.3f}")
    scatter_plots(x_test, y_test, predictions)


if __name__ == "__main__":
    main()
