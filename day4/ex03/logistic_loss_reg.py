from l2_reg import l2
from linear_loss_reg import validate_reg_loss_args
import numpy as np


@validate_reg_loss_args
def reg_log_loss_(y, y_hat, theta, lambda_):
    y = y.flatten()
    y_hat = y_hat.flatten()
    return (
        (lambda_ / 2 * l2(theta)) - (y @ np.log(y_hat) + (1 - y) @ np.log(1 - y_hat))
    ) / len(y)
