from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mbam import Geodesic, initial_velocity

from MMR import r, j, Avv


# Choose starting parameters
x = np.log([1.0, 1.0])
v = initial_velocity(x, j, Avv)


# Callback function used to monitor the geodesic after each step
def callback(g):
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
geo_forward = Geodesic(r, j, Avv, x, v, atol=1e-2, rtol=1e-2, callback=callback)

# Integrate
geo_forward.integrate(25.0)

# Plot the geodesic path to find the limit
# This should show the singularity at the "fold line" x[0] = x[1]
plt.figure()
plt.plot(geo_forward.ts, geo_forward.xs)
plt.xlabel("tau")
plt.ylabel("Parameter Values")
plt.show()


# Now do opposite direction

# Choose starting parameters
x = np.log([1.0, 1.0])
v = -initial_velocity(x, j, Avv)

# Construct the geodesic
# It is usually not necessary to be very accurate here, so we set small
# tolerances
geo_reverse = Geodesic(r, j, Avv, x, v, atol=1e-2, rtol=1e-2, callback=callback)

# Integrate
geo_reverse.integrate(25.0)

# Plot the geodesic path to find the limit
# This should show the singularity at the "fold line" x[0] = x[1]
# Add The geodesic to the same plot, with negative time points
plt.plot(-geo_reverse.ts, geo_reverse.xs)
plt.xlabel("tau")
plt.ylabel("Parameter Values")
plt.show()


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
plt.figure()
plt.contourf(xs, xs, C, 50)
plt.plot(geo_forward.xs[:, 0], geo_forward.xs[:, 1], "r-")
plt.plot(geo_reverse.xs[:, 0], geo_reverse.xs[:, 1], "g-")
plt.plot([0], [0], "ro")
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()

# Plot surface / geodesic in data space
plt.figure()
ax = plt.axes(projection="3d")
surf = ax.plot_surface(
    X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False
)

geo = geo_forward
X = np.empty(len(geo.xs))
Y = np.empty(len(geo.xs))
Z = np.empty(len(geo.xs))
for i, x in enumerate(geo.xs):
    X[i], Y[i], Z[i] = r(x)
ax.plot(X, Y, Z, color="r", zorder=10)

geo = geo_reverse
X = np.empty(len(geo.xs))
Y = np.empty(len(geo.xs))
Z = np.empty(len(geo.xs))
for i, x in enumerate(geo.xs):
    X[i], Y[i], Z[i] = r(x)
ax.plot(X, Y, Z, color="g", zorder=10)


# Plot starting point of geodesic as a red dot
ax.scatter([r0[0]], [r0[1]], [r0[2]], c="r", s=25, zorder=10)
plt.show()
