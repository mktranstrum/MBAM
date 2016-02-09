# An example of calculating a geodesic
# model takes the form y(t,x) = e^(-x[0]*t) + e^(-x[1]*t) for time points t = 0.5, 2.0
# We enforce x[i] > 0 by going to log parameters:
# y(t,x) = e^(-exp(x[0])*t) + e^(-exp(x[1])*t)

from geodesic import geodesic, InitialVelocity
import numpy as np
exp = np.exp

# Model predictions
def r(x):
    return np.array([ exp(-exp(x[0])*0.5) + exp(-exp(x[1])*0.5), exp(-exp(x[0])*2) + exp(-exp(x[1])*2)])

    
# Jacobian
def j(x):
    return np.array([ [ -0.5*exp(x[0])*exp(-x[0]*0.5), -2.0*exp(x[0])*exp(-x[0]*2)],
                          [ -0.5*exp(x[1])*exp(-x[1]*0.5), -2.0*exp(x[1])*exp(-x[1]*2)] ] )

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
    print "Iteration: %i, tau: %f, |v| = %f" %(len(geo.vs), geo.ts[-1], np.linalg.norm(geo.vs[-1]))
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
