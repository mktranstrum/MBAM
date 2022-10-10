import numpy as np
from scipy.integrate import ode


# We construct a geodesic class that inherits from scipy's ode class
class Geodesic(ode):
    """A class to formulate and solve the geodesic equation of a model.

    Parameters
    ----------
    r: callable ``r(x)``
        Function for calculating the model. This is not needed for the
        geodesic, but is used to save the geodesic path in data space.
    j: callable ``j(x)``
        Function for calculating the jacobian of the model with respect to
        parameters. The default function uses forward difference method.
    Avv: callable ``Avv(x,v)``
        Function for calculating the second directional derivative of the
        model in a direction ``v``. The default function uses central
        difference method.
    x: (N,) np.ndarray
        Vector of initial parameter values.
    v: np.ndarray
        Vector of initial velocity.
    lam: float (optional)
        Set to non-zero to calculate geodesic on the model graph.
    dtd: (N, N) np.ndarray (optional)
        Metric for the parameter space contribution to the model graph.
    atol: float (optional)
        Absolute tolerance for solving the geodesic.
    rtol: float (optional)
        Relative tolerance for solving geodesic.
    callback: callable ``callback(geo)`` (optional)
        Function called after each geodesic step. Called as
        ``callback(geo)`` where ``geo`` is the current instance of the
        ``geodesic`` class.
    parameterspacenorm: bool (optional)
        Set to ``True`` to reparameterize the geodesic to have a constant
        parameter space norm and ``False`` to have a constant data space norm.
    invSVD: bool (optional)
        Set to true to use the singular value decomposition to calculate
        the inverse metric. This is slower, but can help with nearly
        singular FIM.

    Attributes
    ----------
    xs: (T, N) np.ndarray
        Geodesics in the parameter space. :math:`N` is the number of parameters.
    rs: (T, M) np.ndarray
        Geodesics in the data space. :math:`N` is the number of predictions.
    vs: (T, N) np.ndarray
        Velocity in the parameter space.
    ts: (T,) np.ndarray
        Geodesic arclength in the parameter space.
    taus: (T,) np.ndarray
        Geodesic arclength in the data space.

    Notes
    -----
    If the prediction points are weighted differently, then the weights need
    to be included in the Jacobian and directional second derivative
    functions. An option to do this is to use the residual function when
    computing the Jacobian and Avv, assuming that the weights are encoded
    in the residual.
    """

    def __init__(
        self,
        r,
        j,
        Avv,
        x,
        v,
        lam=0.0,
        dtd=None,
        atol=1e-6,
        rtol=1e-6,
        callback=None,
        parameterspacenorm=True,
        invSVD=False,
    ):

        self.N = len(x)

        # Functions needed in the calculation
        self.r, self.j, self.Avv = r, j, Avv

        # Initialize the ODE object
        ode.__init__(self, self.geodesic_rhs, jac=None)
        self.set_initial_value(x, v)
        ode.set_integrator(self, "vode", atol=atol, rtol=rtol)

        if callback is None:
            self.callback = callback_func
        else:
            self.callback = callback

        # Additional settings for RHS function
        self.lam = lam
        if dtd is None:
            self.dtd = np.eye(self.N)
        else:
            self.dtd = dtd
        self.parameterspacenorm = parameterspacenorm
        self.invSVD = invSVD

    def geodesic_rhs(self, t, xvtau):
        """This function implements the RHS of the geodesic equation. This
        equation is given by

        .. math::

            (J' J)^{-1} J' \mathbf{A}_{vv}

        This should not need to be edited or called directly by the user.

        Parameters
        ----------
        t: float
            "time" of the geodesic (tau).
        xvtau: (2N + 1,) np.ndarray
            Vector of current parameters, velocities, and data space distance.
        """
        x = xvtau[: self.N]
        v = xvtau[self.N : -1]
        v2 = v @ v
        if v2 == 0.0:
            return np.concatenate((v, 0, [0]))
        else:
            j = self.j(x)
            g = j.T @ j + self.lam * self.dtd
            Avv = self.Avv(x, v)
            if self.invSVD:
                u, s, vh = np.linalg.svd(j, 0)
                a = -vh.T @ (u.T @ Avv) / s
            else:
                a = -np.linalg.solve(g, j.T @ Avv)
            if self.parameterspacenorm:
                a -= a @ v * v / (v2)
            dtau = np.linalg.norm(j @ v)
            return np.concatenate((v, a, [dtau]))

    def set_initial_value(self, x, v):
        """Set the initial parameter values and velocity.

        Parameters
        ----------
        x: np.ndarray
            Vector of initial parameter values.
        v: np.ndarray
            Vector of initial velocity.
        """
        self.xs = np.array([x])
        self.vs = np.array([v])
        self.rs = np.array([self.r(x)])
        self.taus = np.array([0.0])
        self.ts = np.array([0.0])
        ode.set_initial_value(self, np.concatenate((x, v, [0.0])), 0.0)

    def step(self, dt=1.0):
        """Integrate the geodesic for one step.

        Parameters
        ----------
        dt: float
            target time step to use
        """
        ode.integrate(self, self.t + dt, step=1)
        self.xs = np.append(self.xs, [self.y[: self.N]], axis=0)
        self.vs = np.append(self.vs, [self.y[self.N : -1]], axis=0)
        self.rs = np.append(self.rs, [self.r(self.xs[-1])], axis=0)
        self.taus = np.append(self.taus, self.y[-1])
        self.ts = np.append(self.ts, self.t)

    def integrate(self, tmax, maxsteps=500):
        """Integrate the geodesic up to a fixed maximimum time (tau) or number
        of steps. After each step, the callback is called and the calculation
        will end if the callback returns False.

        Parameters
        ----------
        tmax: float
            Maximum time (tau) for integration
        maxsteps: int (optional)
            Maximum number of steps
        """
        cont = True
        while (
            self.successful()
            and len(self.xs) < maxsteps
            and self.t < tmax
            and cont
        ):
            self.step(tmax - self.t)
            cont = self.callback(self)


def callback_func(geo):
    """A default callback function. The geodesic can be halted by the callback
    returning False.

    Parameters
    ----------
    geo: object
        The current instance of the ``geodesic`` class.
    """
    return True
