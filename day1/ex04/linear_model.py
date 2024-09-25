from my_linear_regression import MyLinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

DEBUG = False


def read_dataset():
    try:
        data = pd.read_csv("are_blue_pills_magics.csv")
        X = np.array(data["Micrograms"]).reshape(-1, 1)
        Y = np.array(data["Score"]).reshape(-1, 1)
        return X, Y
    except Exception:
        print("Failed to read the dataset")
        sys.exit(1)


def plot_linear_regression(X, Y, linear_model):
    def predict_one(x):
        return linear_model.thetas[0][0] + linear_model.thetas[1][0] * x

    plt.grid(True)
    plt.scatter(X, Y, color="tab:blue", label="Actual scores")
    line_x = [min(X), max(X)]
    line_y = list(map(predict_one, line_x))
    plt.plot(line_x, line_y, "--", color="tab:green", label="Predicted scores")
    for xi in X:
        plt.scatter(xi, predict_one(xi), color="tab:green", marker="x")
    plt.xlabel("Quantity of blue pill (in micrograms)")
    plt.ylabel("Space driving score")
    plt.legend()
    plt.show()


def plot_losses(X, Y, fitted_model):
    def compute_loss(theta0, theta1, X, Y):
        linear_model = MyLinearRegression(np.array([[theta0], [theta1]]))
        y_hat = linear_model.predict_(X)
        return linear_model.loss_(Y, y_hat)

    VAR_THETA1 = 6
    STEP_THETA0 = 6
    PLOTS_THETA0 = 5
    min_theta1 = fitted_model.thetas[1][0] - VAR_THETA1
    max_theta1 = fitted_model.thetas[1][0] + VAR_THETA1
    base_theta0 = round(fitted_model.thetas[0][0])
    thetas0 = list(
        range(
            base_theta0 - (PLOTS_THETA0 >> 1) * STEP_THETA0,
            base_theta0 + (PLOTS_THETA0 >> 1) * STEP_THETA0 + 1,
            STEP_THETA0,
        )
    )
    plt.grid(True)
    plt.xlim((min_theta1, max_theta1))
    plt.ylim((10, 150))
    for theta0 in thetas0:
        thetas1 = np.linspace(min_theta1, max_theta1, 100)
        losses = [compute_loss(theta0, theta1, X, Y) for theta1 in thetas1]
        plt.plot(thetas1, losses, label=f"J(θ₀={theta0}, θ₁)")
    plt.legend(loc="lower right")
    plt.xlabel("θ₁")
    plt.ylabel("Cost function J(θ₀, θ₁)")
    plt.show()


def compute_mse(X, Y, linear_model):
    y_hat = linear_model.predict_(X)
    print(f"  θ₀ = {linear_model.thetas[0][0]:.3f}")
    print(f"  θ₁ = {linear_model.thetas[1][0]:.3f}")
    print(f"loss = {linear_model.loss_(Y, y_hat):.3f}")
    print(f" mse = {linear_model.mse_(Y, y_hat):.3f}")
    if DEBUG:
        sklearn_mse = __import__(
            "sklearn.metrics", fromlist=["mean_squared_error"]
        ).mean_squared_error(Y, y_hat)
        print(f" mse = {sklearn_mse:.3f}")


def main():
    X, Y = read_dataset()
    linear_model = MyLinearRegression(np.zeros((2, 1)), max_iter=100_000)
    linear_model.fit_(X, Y)
    plot_linear_regression(X, Y, linear_model)
    plot_losses(X, Y, linear_model)
    compute_mse(X, Y, linear_model)


if __name__ == "__main__":
    main()
