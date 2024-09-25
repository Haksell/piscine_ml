import numpy as np


def __validate_l2(func):
    def wrapper(theta):
        return (
            func(theta)
            if isinstance(theta, np.ndarray)
            and theta.ndim == 2
            and theta.shape[0] >= 1
            and theta.shape[1] == 1
            else None
        )

    return wrapper


@__validate_l2
def iterative_l2(theta):
    return sum(t * t for t in theta[1:, 0])


@__validate_l2
def l2(theta):
    truncated_theta = theta[1:, 0]
    return truncated_theta @ truncated_theta
