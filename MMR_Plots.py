from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
import numpy as np
from MMR import M, r, j, Avv
from geodesic import geodesic, InitialVelocity


# Choose starting parameters
x = np.log([1.0, 1.0])
v = InitialVelocity(x, j, Avv)
N = len(x)


def callback(g):
    """Callback function used to monitor the geodesic after each step."""
    # Integrate until the norm of the velocity has grown by a factor of 10
    # and print out some diagnotistic along the way
    print(
        "Iteration: %i, tau: %f, |v| = %f"
        % (len(g.vs), g.ts[-1], np.linalg.norm(g.vs[-1]))
    )
    return np.linalg.norm(g.vs[-1]) < 100.0


# Construct the geodesic
# It is usually not necessary to be very accurate here, so we set small
# tolerances
geo_forward = geodesic(
    r, j, Avv, M, N, x, v, atol=1e-2, rtol=1e-2, callback=callback
)

# Integrate
geo_forward.integrate(25.0)
# plot the geodesic path to find the limit
# This should show the singularity at the "fold line" x[0] = x[1]
pylab.figure()
pylab.plot(geo_forward.ts, geo_forward.xs)
pylab.xlabel("tau")
pylab.ylabel("Parameter Values")
pylab.show()


# Now do opposite direction

# Choose starting parameters
# x = np.log([1.0, 1.0])
v *= -1  # v = -InitialVelocity(x, j, Avv)

# Construct the geodesic
# It is usually not necessary to be very accurate here, so we set small
# tolerances
geo_reverse = geodesic(
    r, j, Avv, M, N, x, v, atol=1e-2, rtol=1e-2, callback=callback
)

# Integrate
geo_reverse.integrate(25.0)
# plot the geodesic path to find the limit
# This should show the singularity at the "fold line" x[0] = x[1]
# Add The geodesic to the same plot, with negative time points
pylab.plot(-geo_reverse.ts, geo_reverse.xs)
pylab.xlabel("tau")
pylab.ylabel("Parameter Values")
pylab.show()


# Now construct contour plots in parameter space and model manifold in data
# space

r0 = r([0.0, 0.0])
xs = np.linspace(-5, 5, 101)
X = np.empty((101, 101))
Y = np.empty((101, 101))
Z = np.empty((101, 101))
C = np.empty((101, 101))
for i, x in enumerate(xs):
    for j, y in enumerate(xs):
        temp = r([x, y])
        X[j, i], Y[j, i], Z[j, i] = temp
        C[j, i] = np.linalg.norm(temp - r0) ** 2

# Plot geodesic path in parameter space with cost contours
pylab.figure()
pylab.contourf(xs, xs, C, 50)
pylab.plot(geo_forward.xs[:, 0], geo_forward.xs[:, 1], "r-")
pylab.plot(geo_reverse.xs[:, 0], geo_reverse.xs[:, 1], "g-")
pylab.plot([0], [0], "ro")
pylab.xlim(-5, 5)
pylab.ylim(-5, 5)
pylab.show()

# Plot surface / geodesic in data space

fig = plt.figure()
ax = fig.gca(projection="3d")
surf = ax.plot_surface(
    X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False
)

geo = geo_forward
X = np.empty(len(geo.xs))
Y = np.empty(len(geo.xs))
Z = np.empty(len(geo.xs))
for i, x in enumerate(geo.xs):
    X[i], Y[i], Z[i] = r(x)
ax.plot(X, Y, Z, color=(1, 0, 0))

geo = geo_reverse
X = np.empty(len(geo.xs))
Y = np.empty(len(geo.xs))
Z = np.empty(len(geo.xs))
for i, x in enumerate(geo.xs):
    X[i], Y[i], Z[i] = r(x)
ax.plot(X, Y, Z, color=(0, 1, 0))


# Plot starting point of geodesic as a red dot
ax.scatter([r0[0]], [r0[1]], [r0[2]], c="r", s=25)
plt.show()
