import warnings
import numpy as np


def __validate_score(func):
    def wrapper(y, y_hat, pos_label=None):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                assert isinstance(y, np.ndarray)
                assert isinstance(y_hat, np.ndarray)
                assert y.shape == y_hat.shape
                return (
                    func(y, y_hat) if pos_label is None else func(y, y_hat, pos_label)
                )
            except Exception:
                return None

    return wrapper


@__validate_score
def accuracy_score_(y, y_hat):
    return (y == y_hat).mean()


@__validate_score
def precision_score_(y, y_hat, pos_label=1):
    positives = y_hat == pos_label
    true_positives = positives & (y == pos_label)
    return true_positives.sum() / positives.sum()


@__validate_score
def recall_score_(y, y_hat, pos_label=1):
    positives = y == pos_label
    true_positives = positives & (y_hat == pos_label)
    return true_positives.sum() / positives.sum()


@__validate_score
def f1_score_(y, y_hat, pos_label=1):
    tp = ((y == pos_label) & (y_hat == pos_label)).sum()
    fn = ((y == pos_label) & (y_hat != pos_label)).sum()
    fp = ((y != pos_label) & (y_hat == pos_label)).sum()
    return 2 * tp / (2 * tp + fp + fn)
