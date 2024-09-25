from math import sqrt
import numpy as np
from TinyStatistician import PercentileMethod, TinyStatistician

A = [1, 42, 300, 10, 59]


def test_mean():
    assert TinyStatistician().mean(A) == 82.4
    assert TinyStatistician().mean(np.array(A)) == 82.4
    assert TinyStatistician().mean([42]) == 42
    assert TinyStatistician().mean([42, 43]) == 42.5
    assert TinyStatistician().mean([42, 43, 44, 46]) == 43.75
    assert TinyStatistician().mean([]) is None
    assert TinyStatistician().mean("lol") is None


def test_median():
    assert TinyStatistician().median(A) == 42
    assert TinyStatistician().median(np.array(A)) == 42
    assert TinyStatistician().median([42]) == 42
    assert TinyStatistician().median([42, 43]) == 42.5
    assert TinyStatistician().median([42, 43, 44, 46]) == 43.5
    assert TinyStatistician().median([]) is None
    assert TinyStatistician().median("lol") is None


def test_quartile():
    assert TinyStatistician().quartile(A) == [10, 59]
    assert TinyStatistician().quartile([42]) == [42, 42]
    assert TinyStatistician().quartile([42, 43]) == [42.25, 42.75]
    assert TinyStatistician().quartile([42, 43, 44, 46]) == [42.75, 44.5]


A = [1, 10, 42, 59, 300]


def test_percentile_lower():
    assert TinyStatistician().percentile(A, 0, method=PercentileMethod.LOWER) == 1
    assert TinyStatistician().percentile(A, 10, method=PercentileMethod.LOWER) == 1
    assert TinyStatistician().percentile(A, 15, method=PercentileMethod.LOWER) == 1
    assert (
        TinyStatistician().percentile(np.array(A), 20, method=PercentileMethod.LOWER)
        == 1
    )
    assert TinyStatistician().percentile(A, 100, method=PercentileMethod.LOWER) == 300
    assert TinyStatistician().percentile(A, -1, method=PercentileMethod.LOWER) is None
    assert TinyStatistician().percentile(A, 101, method=PercentileMethod.LOWER) is None


def test_percentile_upper():
    assert TinyStatistician().percentile(A, 0, method=PercentileMethod.UPPER) == 1
    assert TinyStatistician().percentile(A, 10, method=PercentileMethod.UPPER) == 10
    assert TinyStatistician().percentile(A, 15, method=PercentileMethod.UPPER) == 10
    assert (
        TinyStatistician().percentile(np.array(A), 20, method=PercentileMethod.UPPER)
        == 10
    )
    assert TinyStatistician().percentile(A, 100, method=PercentileMethod.UPPER) == 300
    assert TinyStatistician().percentile(A, -1, method=PercentileMethod.UPPER) is None
    assert TinyStatistician().percentile(A, 101, method=PercentileMethod.UPPER) is None


def test_percentile_closest():
    assert TinyStatistician().percentile(A, 0, method=PercentileMethod.CLOSEST) == 1
    assert TinyStatistician().percentile(A, 10, method=PercentileMethod.CLOSEST) == 1
    assert TinyStatistician().percentile(A, 15, method=PercentileMethod.CLOSEST) == 10
    assert (
        TinyStatistician().percentile(np.array(A), 20, method=PercentileMethod.CLOSEST)
        == 10
    )
    assert TinyStatistician().percentile(A, 100, method=PercentileMethod.CLOSEST) == 300
    assert TinyStatistician().percentile(A, -1, method=PercentileMethod.CLOSEST) is None
    assert (
        TinyStatistician().percentile(A, 101, method=PercentileMethod.CLOSEST) is None
    )


def test_percentile_middle():
    assert TinyStatistician().percentile(A, 0, method=PercentileMethod.MIDDLE) == 1
    assert TinyStatistician().percentile(A, 10, method=PercentileMethod.MIDDLE) == 5.5
    assert TinyStatistician().percentile(A, 15, method=PercentileMethod.MIDDLE) == 5.5
    assert (
        TinyStatistician().percentile(np.array(A), 20, method=PercentileMethod.MIDDLE)
        == 5.5
    )
    assert TinyStatistician().percentile(A, 100, method=PercentileMethod.MIDDLE) == 300
    assert TinyStatistician().percentile(A, -1, method=PercentileMethod.MIDDLE) is None
    assert TinyStatistician().percentile(A, 101, method=PercentileMethod.MIDDLE) is None


def test_percentile_interpolation():
    assert (
        TinyStatistician().percentile(A, 0, method=PercentileMethod.INTERPOLATION) == 1
    )
    assert (
        TinyStatistician().percentile(A, 10, method=PercentileMethod.INTERPOLATION)
        == 4.6
    )
    assert (
        TinyStatistician().percentile(A, 15, method=PercentileMethod.INTERPOLATION)
        == 6.4
    )
    assert (
        TinyStatistician().percentile(
            np.array(A), 20, method=PercentileMethod.INTERPOLATION
        )
        == 8.2
    )
    assert (
        TinyStatistician().percentile(A, 100, method=PercentileMethod.INTERPOLATION)
        == 300
    )
    assert (
        TinyStatistician().percentile(A, -1, method=PercentileMethod.INTERPOLATION)
        is None
    )
    assert (
        TinyStatistician().percentile(A, 101, method=PercentileMethod.INTERPOLATION)
        is None
    )


def is_close(x, y, eps=1e-3):
    return abs(x - y) <= eps


def test_var():
    assert is_close(TinyStatistician().var(A), 15349.3)
    assert TinyStatistician().var([42]) == 0
    assert TinyStatistician().var([42, 42, 42]) == 0
    assert TinyStatistician().var([40, 41, 42, 42, 43, 44]) == 2
    assert TinyStatistician().var([]) is None


def test_std():
    assert is_close(TinyStatistician().std(A), 123.89229193133849)
    assert TinyStatistician().std([42]) == 0
    assert TinyStatistician().std([42, 42, 42]) == 0
    assert is_close(TinyStatistician().std([40, 41, 42, 42, 43, 44]), sqrt(2))
    assert TinyStatistician().std([]) is None
