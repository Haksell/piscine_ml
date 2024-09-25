import numpy as np
import matplotlib.pyplot as plt


def plot_letter(ax, char, x, color):
    ax.text(
        x,
        0.5,
        char,
        color=color,
        fontsize=45,
        fontweight="bold",
        ha="center",
        va="center",
    )


def plot(x, y, theta):
    if (
        isinstance(x, np.ndarray)
        and isinstance(y, np.ndarray)
        and isinstance(theta, np.ndarray)
        and (x.ndim == 1 or x.ndim == 2 and 1 in x.shape)
        and (y.ndim == 1 or y.ndim == 2 and 1 in y.shape)
        and x.size == y.size != 0
        and theta.size == 2
    ):
        plt.scatter(x, y)
        line_x = np.array([x.min(), x.max()])
        theta = theta.flatten()
        line_y = theta[0] + theta[1] * line_x
        plt.plot(line_x, line_y, color="orange")
        plt.show()
    else:
        fig, ax = plt.subplots()
        plot_letter(ax, "N", 0.4, "red")
        plot_letter(ax, "O", 0.5, "blue")
        plot_letter(ax, "N", 0.6, "red")
        ax.axis("off")
        plt.show()
