# An example of calculating a geodesic
# model takes the form y(t,x) = e^(-x[0]*t) + e^(-x[1]*t) for time points t = 0.5, 1.0, 2.0
# We enforce x[i] > 0 by going to log parameters:
# y(t,x) = e^(-exp(x[0])*t) + e^(-exp(x[1])*t)
# 
# We adopt the convention that the model has N parameters and makes M predictions.
# Then, the output of r(x) should be a vector length M
# the output of j(x) (i.e., jacobian) should be an M*N matrix
# theoutput of Avv(x,v) should be a vector of length M
# In this example, M = 3 (three time points) and N = 2.

from geodesic import geodesic, InitialVelocity
import numpy as np
exp = np.exp

t = np.array([0.5, 1.0, 2.0])

# Model predictions
def r(x):
    return exp(-exp(x[0])*t) + exp(-exp(x[1])*t)

    
# Jacobian
def j(x):
    return np.array([ -t*exp(x[0])*exp(-exp(x[0])*t), -t*exp(x[1])*exp(-exp(x[1])*t)]).T

# Directional second derivative
def Avv(x,v):
    h = 1e-4
    return (r(x + h*v) + r(x - h*v) - 2*r(x))/h/h
    

    
# Choose starting parameters
x = np.log([1.0, 2.0])
v = InitialVelocity(x, j, Avv)


# Callback function used to monitor the geodesic after each step
def callback(geo):
    # Integrate until the norm of the velocity has grown by a factor of 10
    # and print out some diagnotistic along the way
    print("Iteration: %i, tau: %f, |v| = %f" %(len(geo.vs), geo.ts[-1], np.linalg.norm(geo.vs[-1])))
    return np.linalg.norm(geo.vs[-1]) < 10.0

# Construct the geodesic
# It is usually not necessary to be very accurate here, so we set small tolerances
geo = geodesic(r, j, Avv, 2, 2, x, v, atol = 1e-2, rtol = 1e-2, callback = callback)  

# Integrate
geo.integrate(25.0)
import pylab
# plot the geodesic path to find the limit
# This should show the singularity at the "fold line" x[0] = x[1]
pylab.plot(geo.ts, geo.xs)
pylab.xlabel("tau")
pylab.ylabel("Parameter Values")
pylab.show()

