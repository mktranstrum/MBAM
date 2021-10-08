"""An example of calculating a geodesic. Model takes the form

.. math::

    y(t,x) = e^{-x[0]*t} + e^{-x[1]*t},

for time points :math:`t = 0.5, 1.0, 2.0`.

We enforce :math:`x[i] > 0` by going to log parameters:

.. math::

    y(t,x) = e^{-exp(x[0])*t} + e^{-exp(x[1])*t}

We adopt the convention that the model has N parameters and makes M
predictions. Then, the output of :math:`r(x)` should be a vector length M the
output of :math:`j(x)` (i.e., jacobian) should be an :math:`M \times N` matrix.
The output of :math:`Avv(x,v)` should be a vector of length M. In this example,
:math:`M = 3` (three time points) and :math:`N = 2`.
"""

from geodesic import geodesic, InitialVelocity
import numpy as np
import pylab

exp = np.exp

t = np.array([0.5, 1.0, 2.0])


def r(x):
    """Model predictions"""
    return exp(-exp(x[0]) * t) + exp(-exp(x[1]) * t)


def j(x):
    """Jacobian"""
    return np.array(
        [
            -t * exp(x[0]) * exp(-exp(x[0]) * t),
            -t * exp(x[1]) * exp(-exp(x[1]) * t),
        ]
    ).T


def Avv(x, v):
    """Directional second derivative"""
    h = 1e-4
    return (r(x + h * v) + r(x - h * v) - 2 * r(x)) / h / h


# Choose starting parameters
x = np.log([1.0, 2.0])
v = InitialVelocity(x, j, Avv)

# Set the dimensions
M = len(t)
N = len(x)


def callback(g):
    """Callback function used to monitor the geodesic after each step"""
    # Integrate until the norm of the velocity has grown by a factor of 10
    # and print out some diagnotistic along the way
    print(
        "Iteration: %i, tau: %f, |v| = %f"
        % (len(g.vs), g.ts[-1], np.linalg.norm(g.vs[-1]))
    )
    return np.linalg.norm(g.vs[-1]) < 10.0


# Construct the geodesic
# It is usually not necessary to be very accurate here, so we set small
# tolerances
geo = geodesic(r, j, Avv, M, N, x, v, atol=1e-2, rtol=1e-2, callback=callback)

# Integrate
geo.integrate(25.0)
# plot the geodesic path to find the limit
# This should show the singularity at the "fold line" x[0] = x[1]
pylab.plot(geo.ts, geo.xs)
pylab.xlabel("tau")
pylab.ylabel("Parameter Values")
pylab.show()
