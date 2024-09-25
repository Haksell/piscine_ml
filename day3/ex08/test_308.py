import numpy as np
from other_metrics import accuracy_score_, precision_score_, recall_score_, f1_score_
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y1_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
y1 = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))

y2_hat = np.array(
    ["norminet", "dog", "norminet", "norminet", "dog", "dog", "dog", "dog"]
)
y2 = np.array(
    ["dog", "dog", "norminet", "norminet", "dog", "norminet", "dog", "norminet"]
)


def test_accuracy():
    assert np.isclose(accuracy_score_(y1, y1_hat), accuracy_score(y1, y1_hat))
    assert np.isclose(accuracy_score_(y2, y2_hat), accuracy_score(y2, y2_hat))


def test_precision():
    assert np.isclose(precision_score_(y1, y1_hat), precision_score(y1, y1_hat))
    assert np.isclose(
        precision_score_(y2, y2_hat, pos_label="dog"),
        precision_score(y2, y2_hat, pos_label="dog"),
    )
    assert np.isclose(
        precision_score_(y2, y2_hat, pos_label="norminet"),
        precision_score(y2, y2_hat, pos_label="norminet"),
    )


def test_recall():
    assert np.isclose(recall_score_(y1, y1_hat), recall_score(y1, y1_hat))
    assert np.isclose(
        recall_score_(y2, y2_hat, pos_label="dog"),
        recall_score(y2, y2_hat, pos_label="dog"),
    )
    assert np.isclose(
        recall_score_(y2, y2_hat, pos_label="norminet"),
        recall_score(y2, y2_hat, pos_label="norminet"),
    )


def test_f1():
    assert np.isclose(f1_score_(y1, y1_hat), f1_score(y1, y1_hat))
    assert np.isclose(
        f1_score_(y2, y2_hat, pos_label="dog"), f1_score(y2, y2_hat, pos_label="dog")
    )
    assert np.isclose(
        f1_score_(y2, y2_hat, pos_label="norminet"),
        f1_score(y2, y2_hat, pos_label="norminet"),
    )


def test_invalid_accuracy():
    assert accuracy_score_(np.ones(0), np.ones(0)) is None
    assert accuracy_score_(np.ones(4), np.ones(5)) is None


def test_invalid_f1():
    assert f1_score_(np.ones(0), np.ones(0)) is None
    assert f1_score_(np.ones(4), np.ones(5)) is None
    assert f1_score_(np.zeros(3), np.zeros(3)) is None
