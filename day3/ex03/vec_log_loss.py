import numpy as np


def vec_log_loss_(y, y_hat, eps=1e-15):
    if (
        isinstance(y, np.ndarray)
        and y.ndim == 2
        and y.shape[0] >= 1
        and y.shape[1] == 1
        and all(yi == 0 or yi == 1 for yi in y.flatten())
        and isinstance(y_hat, np.ndarray)
        and y.shape == y_hat.shape
        and all(0 <= yi_hat <= 1 for yi_hat in y_hat.flatten())
        and isinstance(eps, float)
        and 0 < eps < 0.1
    ):
        return -(
            y * np.log(np.maximum(eps, y_hat))
            + (1 - y) * np.log(np.maximum(eps, 1 - y_hat))
        ).mean()
