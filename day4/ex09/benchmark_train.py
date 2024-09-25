import pickle
import sys
import numpy as np
from other_metrics import accuracy_score_, f1_score_
from polynomial_model_extended import add_polynomial_features
from shared import (
    DEGREE,
    predict_ovr,
    read_datasets,
    train_ovr,
)


def main():
    x_train, x_val, _, y_train, y_val, _ = read_datasets()
    x_train = add_polynomial_features(x_train, DEGREE)
    x_val = add_polynomial_features(x_val, DEGREE)
    models = []
    # If you actually want different results, try np.linspace(0, 3000, 6)
    for i, lambda_ in enumerate(np.linspace(0, 1, 6)):
        if i != 0:
            print()
        model = train_ovr(
            x_train, x_val, y_train, y_val, lambda_=lambda_, alpha=0.01, max_iter=10
        )
        predictions = predict_ovr(x_val, model)
        print(
            f"Full model accuracy (lambda={lambda_:.1f}): {accuracy_score_(predictions, y_val):.3f}"
        )
        f1_score = f1_score_(predictions, y_val)
        print(f"Full model f1-score (lambda={lambda_:.1f}): {f1_score:.3f}")
        models.append((f1_score, lambda_, model))
    try:
        pickle.dump(models, open("models.pickle", "wb"))
    except Exception as e:
        print(f"Failed to save the models: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
