import numpy as np
from reg_linear_grad import validate_reg_grad_args


def __sigmoid(x):
    with np.errstate(over="ignore"):
        return 1 / (1 + np.exp(-x))


@validate_reg_grad_args
def reg_logistic_grad(y, x, theta, lambda_):
    y_diff = [
        __sigmoid(theta[0, 0] + sum(xi * ti for xi, ti in zip(x_row, theta[1:, 0])))
        - yi
        for x_row, yi in zip(x, y)
    ]
    return np.vstack(
        [
            sum(y_diff) / len(y_diff),
            [
                (sum(xi * yi for xi, yi in zip(x_col, y_diff)) + lambda_ * theta_j)
                / len(y_diff)
                for x_col, theta_j in zip(x.T, theta[1:, 0])
            ],
        ]
    )


@validate_reg_grad_args
def vec_reg_logistic_grad(y, x, theta, lambda_):
    x_stack = np.hstack((np.ones((x.shape[0], 1)), x))
    theta_prime = theta.copy()
    theta_prime[0, 0] = 0
    return (
        x_stack.T @ (__sigmoid(x_stack @ theta) - y) + lambda_ * theta_prime
    ) / y.size
