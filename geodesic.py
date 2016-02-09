from scipy.integrate import ode
import numpy as np

# A default callback function
# the geodesic can be halted by the callback returning False
def callback_func(geo):
    return True
    
# We construct a geodesic class that inherits from scipy's ode class
class geodesic(ode):

    def __init__(self, r, j, Avv, M, N, x, v, lam = 0.0, dtd = None, atol = 1e-6, rtol = 1e-6, callback = None, parameterspacenorm = False,invSVD=False):
        """
        Construct an instance of the geodesic object
        r = function for calculating the model.  This is not needed for the geodesic, but is used to save the geodesic path in data space.  Called as r(x)
        j = function for calculating the jacobian of the model with respect to parameters.  Called as j(x)
        Avv = function for calculating the second directional derivative of the model in a direction v.  Called as Avv(x,v)
        M = Number of model predictions
        N = Number of model parameters
        x = vector of initial parameter values
        v = vector of initial velocity
        lam = set to nonzero to calculate geodesic on the model graph.
        dtd = metric for the parameter space contribution to the model graph
        atol = absolute tolerance for solving the geodesic
        rtol = relative tolerance for solving geodesic
        callback = function called after each geodesic step.  Called as callback(geo) where geo is the current instance of the geodesic class.
        parameterspacenorm = Set to True to reparameterize the geodesic to have a constant parameter space norm.  (Default is to have a constant data space norm.)
        invSVD = Set to true to use the singular value decomposition to calculate the inverse metric.  This is slower, but can help with nearly singular FIM.
        """
        self.r, self.j, self.Avv = r, j, Avv
        self.M, self.N = M, N
        self.lam = lam
        if dtd is None:
            self.dtd = np.eye(N)
        else:
            self.dtd = dtd
        self.atol = atol
        self.rtol = rtol
        ode.__init__(self, self.geodesic_rhs, jac = None)
        self.set_initial_value(x, v)
        ode.set_integrator(self, 'vode', atol = atol, rtol = rtol)
        if callback is None:
            self.callback = callback_func
        else:
            self.callback = callback
        self.parameterspacenorm = parameterspacenorm
        self.invSVD = False 
        
    def geodesic_rhs(self, t, xv):
        """
        This function implements the RHS of the geodesic equation
        This should not need to edited or called directly by the user
        t = "time" of the geodesic (tau)
        xv = vector of current parameters and velocities
        """
        x = xv[:self.N]
        v = xv[self.N:]
        j = self.j(x)
        g = np.dot(j.T, j) + self.lam*self.dtd
        Avv = self.Avv(x, v)
        if self.invSVD:
            u,s,vh = np.linalg.svd(j,0)
            a = - np.dot( vh.T, np.dot(u.T, Avv)/s )
        else:
            a = -np.linalg.solve(g, np.dot(j.T, Avv) )
        if self.parameterspacenorm:
            a -= np.dot(a,v)*v/np.dot(v,v)
        return np.append(v, a)
                                              
    def set_initial_value(self, x, v):
        """
        Set the initial parameter values and velocities
        x = vector of initial parameter values
        v = vector of initial velocity
        """
        self.xs = np.array([x])
        self.vs = np.array([v])
        self.ts = np.array([0.0])
        self.rs = np.array([ self.r(x) ] )
        self.vels = np.array([ np.dot(self.j(x), v) ] )
        ode.set_initial_value( self, np.append(x, v), 0.0 )

    def step(self, dt = 1.0):
        """
        Integrate the geodesic for one step
        dt = target time step to use
        """
        ode.integrate(self, self.t + dt, step = 1)
        self.xs = np.append(self.xs, [self.y[:self.N]], axis = 0)
        self.vs = np.append(self.vs, [self.y[self.N:]], axis = 0 )
        self.rs = np.append(self.rs, [self.r(self.xs[-1])], axis = 0)
        self.vels = np.append(self.vels, [np.dot(self.j(self.xs[-1]), self.vs[-1])], axis = 0)
        self.ts = np.append(self.ts, self.t)

    def integrate(self, tmax, maxsteps = 500):
        """
        Integrate the geodesic up to a fixed maximimum time (tau) or number of steps.
        After each step, the callback is called and the calculation will end if the callback returns False
        tmax = maximum time (tau) for integration
        maxsteps = maximum number of steps
        """
        cont = True
        while self.successful() and len(self.xs) < maxsteps and self.t < tmax and cont:
            self.step(tmax - self.t)
            cont = self.callback(self)

def InitialVelocity(x, jac, Avv):
    """
    Routine for calculating the initial velocity
    x: Initial Parameter Values
    jac: function for calculating the jacobian.  Called as jac(x)
    Avv: function for calculating a direction second derivative.  Called as Avv(x,v)
    """

    j = jac(x)
    u,s,vh = np.linalg.svd(j)
    v = vh[-1]
    a = -np.linalg.solve(  np.dot(j.T, j), np.dot(j.T, Avv(x,v) ))
    # We choose the direction in which the velocity will increase, since this is most likely to find a singularity quickest.
    if np.dot(v, a) < 0:
        v *= -1
    return v
    
