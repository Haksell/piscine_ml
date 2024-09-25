import pickle
import sys

import numpy as np
from polynomial_model_extended import add_polynomial_features
from shared import fit_normalized, read_dataset

TOTAL_ITERATIONS = 100000


def main():
    x_train, x_val, x_test, y_train, y_val, y_test = read_dataset()
    x_train = np.vstack([x_train, x_val])
    y_train = np.vstack([y_train, y_val])
    try:
        training_data = pickle.load(open("models.pickle", "rb"))
    except Exception as e:
        print(f"Failed to read the training data: {e}")
        sys.exit(1)
    mse, degree, lambda_, ridge = min(training_data)
    print(
        f"Training a model of degree {degree} with lambda={lambda_:.1f} for {TOTAL_ITERATIONS:,} iterations..."
    )
    x_train_poly = add_polynomial_features(x_train, degree)
    x_test_poly = add_polynomial_features(x_test, degree)
    ridge = fit_normalized(
        x_train_poly, y_train, lambda_=lambda_, alpha=0.2, max_iter=TOTAL_ITERATIONS
    )
    y_hat = ridge.predict_(x_test_poly)
    mse = ridge.mse_(y_test, y_hat)
    print(f"MSE after complete training: {mse:,.1f}")


if __name__ == "__main__":
    main()
