import numpy as np
from scipy.integrate import odeint

# For finite difference estimates of derivatives
from mbam.finite_difference import jacobian_func, Avv_func, CD4, AvvCD4

exp = np.exp
# Time points to sample model. We do not observe t = 0, but is necessary for
# the ODE solver
ts = np.array([0.0, 1.0, 2.0, 5.0])
M = len(ts) - 1  # Number of predictions that the model makes
N = 2  # Target number of parameters


def rhs(y, t, x):
    return -exp(x[0]) * y / (exp(x[1]) + y)


# Returns our observation vector.
def r(x):
    return odeint(rhs, [1.0], ts, (x,))[1:, 0]


# Sensitivities ODE
def drhs(y, t, x):
    # y[0] = y, y[1] = dy/dx[0], y[2] = dy/dx[1]

    # deriviatve of rhs with respect to y
    drhsdy = -exp(x[0] + x[1]) / (exp(x[1]) + y[0]) ** 2
    # deriviatve of rhs with respect to x[0]
    drhsdx0 = -exp(x[0]) * y[0] / (exp(x[1]) + y[0])
    # deriviatve of rhs with respect to x[1]
    drhsdx1 = exp(x[0] + x[1]) * y[0] / (exp(x[1]) + y[0]) ** 2
    return [
        -exp(x[0]) * y[0] / (exp(x[1]) + y[0]),
        drhsdy * y[1] + drhsdx0,
        drhsdy * y[2] + drhsdx1,
    ]


# Jacobian
def j(x):
    return odeint(drhs, [1.0, 0.0, 0.0], ts, (x,))[1:, 1:]


# Alternatively, calculate jacobian using finite differences. See useful,
# higher-order formulas in `finite_difference.py`.
j_FD = jacobian_func(r, M, N, CD4, h=1e-2)


# Directional second derivative. This can also be done by either solving
# the sensitivity equations or using finite differences.  Here, we use finite
# differences.
Avv = Avv_func(r, AvvCD4, h=1e-2)
