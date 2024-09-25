from enum import auto, Enum
from functools import wraps
from math import ceil, floor, sqrt
import numpy as np


class PercentileMethod(Enum):
    LOWER = auto()
    UPPER = auto()
    CLOSEST = auto()
    MIDDLE = auto()
    INTERPOLATION = auto()


def validate_array(func):
    @wraps(func)
    def wrapper(self, arr, *args, **kwargs):
        return (
            func(self, arr, *args, **kwargs)
            if isinstance(arr, (list, np.ndarray)) and len(arr) != 0
            else None
        )

    return wrapper


class TinyStatistician:
    @validate_array
    def mean(self, arr):
        return sum(arr) / len(arr)

    @validate_array
    def percentile(self, arr, p, *, method=PercentileMethod.INTERPOLATION):
        if p < 0 or p > 100:
            return None
        if p == 0:
            return min(arr)
        if p == 100:
            return max(arr)
        arr = sorted(arr)
        idx = p / 100 * (len(arr) - 1)
        before = float(arr[floor(idx)])
        after = float(arr[ceil(idx)])
        interp = idx % 1
        match method:
            case PercentileMethod.LOWER:
                return before
            case PercentileMethod.UPPER:
                return after
            case PercentileMethod.CLOSEST:
                return before if interp <= 0.5 else after
            case PercentileMethod.MIDDLE:
                return before if interp == 0 else (before + after) / 2
            case PercentileMethod.INTERPOLATION:
                return before * (1 - interp) + after * interp

    @validate_array
    def median(self, arr):
        return self.percentile(arr, 50)

    @validate_array
    def quartile(self, arr):
        return [self.percentile(arr, 25), self.percentile(arr, 75)]

    @validate_array
    def var(self, arr):
        mean = self.mean(arr)
        return (
            0 if len(arr) == 1 else sum((x - mean) ** 2 for x in arr) / (len(arr) - 1)
        )

    @validate_array
    def std(self, arr):
        return sqrt(self.var(arr))
