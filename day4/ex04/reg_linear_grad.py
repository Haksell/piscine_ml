import numpy as np


def validate_reg_grad_args(func):
    def wrapper(y, x, theta, lambda_):
        return (
            func(y, x, theta, lambda_)
            if (
                isinstance(y, np.ndarray)
                and y.ndim == 2
                and y.shape[1] == 1
                and isinstance(x, np.ndarray)
                and x.ndim == 2
                and x.shape[0] == y.shape[0]
                and x.shape[1] >= 1
                and isinstance(theta, np.ndarray)
                and theta.shape == (x.shape[1] + 1, 1)
                and (isinstance(lambda_, float) or isinstance(lambda_, int))
                and lambda_ >= 0
            )
            else None
        )

    return wrapper


@validate_reg_grad_args
def reg_linear_grad(y, x, theta, lambda_):
    y_diff = [
        (theta[0, 0] + sum(xi * ti for xi, ti in zip(x_row, theta[1:, 0]))) - yi
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
def vec_reg_linear_grad(y, x, theta, lambda_):
    x_stack = np.hstack((np.ones((x.shape[0], 1)), x))
    theta_prime = theta.copy()
    theta_prime[0, 0] = 0
    return (x_stack.T @ ((x_stack @ theta) - y) + lambda_ * theta_prime) / y.size
