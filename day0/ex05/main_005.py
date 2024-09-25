import numpy as np
from plot import plot

x = np.arange(1, 6)
y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])

plot(x.reshape((5, 1)), y.reshape((1, 5)), np.array([[4.5], [-0.2]]))
plot(x.reshape((1, 5)), y.reshape((1, 5)), np.array([[-1.5], [2]]))
plot(x.reshape((5, 1)), y, np.array([[3], [0.3]]))
plot(x, y, np.array([[3], [0.3], [42]]))
