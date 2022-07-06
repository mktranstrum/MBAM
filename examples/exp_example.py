"""An example of calculating a geodesic. Model takes the form

.. math::

    y(t,x) = e^{-x_0 t} + e^{-x_1 t},

for time points :math:`t = \{0.5, 1.0, 2.0\}`.

We enforce :math:`x_i > 0` by going to log parameters:

.. math::

    y(t,x) = e^{-\exp(x_0) t} + e^{-\exp(x_1) t}.

We adopt the convention that the model has N parameters and makes M
predictions. Then, the output of :math:`r(x)` should be a vector length M the
output of :math:`j(x)` (i.e., jacobian) should be an :math:`M \times N` matrix.
The output of :math:`Avv(x,v)` should be a vector of length M. In this example,
:math:`M = 3` (three time points) and :math:`N = 2`.
"""

from mbam import Geodesic, initial_velocity
import numpy as np
import matplotlib.pyplot as plt


exp = np.exp

t = np.array([0.5, 1.0, 2.0])


# Model predictions
def r(x):
    return exp(-exp(x[0]) * t) + exp(-exp(x[1]) * t)


# Jacobian
def j(x):
    return np.array(
        [
            -t * exp(x[0]) * exp(-exp(x[0]) * t),
            -t * exp(x[1]) * exp(-exp(x[1]) * t),
        ]
    ).T


# Directional second derivative
def Avv(x, v):
    h = 1e-4
    return (r(x + h * v) + r(x - h * v) - 2 * r(x)) / h / h


# Choose starting parameters
x = np.log([1.0, 2.0])
v = initial_velocity(x, j, Avv)


# Callback function used to monitor the geodesic after each step
def callback(g):
    # Integrate until the norm of the velocity has grown by a factor of 10
    # and print out some diagnotistic along the way
    print(
        "Iteration: %i, tau: %f, |v| = %f"
        % (len(g.vs), g.ts[-1], np.linalg.norm(g.vs[-1]))
    )
    return np.linalg.norm(g.vs[-1]) < 20.0


# Construct the geodesic
# It is usually not necessary to be very accurate here, so we set small
# tolerances
geo = Geodesic(r, j, Avv, x, v, atol=1e-2, rtol=1e-2, callback=callback)

# Integrate
geo.integrate(25.0)

# Plot the geodesic path to find the limit
# This should show the singularity at the "fold line" x[0] = x[1]
plt.figure()
plt.plot(geo.ts, geo.xs)
plt.xlabel("tau")
plt.ylabel("Parameter Values")
plt.show()
