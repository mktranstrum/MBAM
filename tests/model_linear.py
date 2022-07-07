import numpy as np


np.random.seed(1717)

# Dimensionality
M = 10
N = 5

# Time series
t = np.logspace(-4, 2, M)

# Define the model
J = np.array([t ** n for n in range(N)]).T


def f(x):
    return J @ x


# Initial conditions
xi = np.random.randn(N)
vi = np.random.randn(N)
