import numpy as np


def add_multi_polynomial_features(x, max_powers):
    assert isinstance(max_powers, tuple)
    assert all(isinstance(p, int) for p in max_powers)
    assert all(p >= 0 for p in max_powers)
    assert any(p > 0 for p in max_powers)
    assert isinstance(x, np.ndarray)
    assert np.issubdtype(x.dtype, np.number)
    assert x.ndim == 2
    assert x.shape[1] == len(max_powers)
    return np.column_stack(
        [
            x[:, col] ** p
            for col, max_power in enumerate(max_powers)
            for p in range(1, max_power + 1)
        ]
    )


def add_polynomial_features(x, power):
    try:
        return add_multi_polynomial_features(x, (power,))
    except AssertionError:
        return None
