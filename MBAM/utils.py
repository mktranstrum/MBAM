"""A script containing utility functions: get sloppiest initial velocity and
wrapper functions for computing the jacobian and Avv.
"""


import numpy as np

from MBAM.finite_difference import FD, AvvCD


def initial_velocity(x, jac, Avv):
    """A routine for calculating the initial velocity in the sloppiest
    direction.

    Parameters
    ----------
    x: (N,) np.ndarray
        Initial Parameter Values.
    jac: callable ``jac(x)``
        Function for calculating the jacobian.
    Avv: callable ``Avv(x, v)``
        Function for calculating a direction second derivative.

    Returns
    -------
    v:
        Initial velocity vector.
    """

    j = jac(x)
    _, _, vh = np.linalg.svd(j)
    v = vh[-1]
    a = -np.linalg.solve(j.T @ j, j.T @ Avv(x, v))
    # We choose the direction in which the velocity will increase, since this
    # is most likely to find a singularity quickest.
    if v @ a < 0:
        v *= -1
    return v


# Derivative of the model
def jacobian_func(f, M, N, deriv_fn=FD, h=0.1):
    """A wrapper to create a Jacobian function of a model.

    Parameters
    ----------
    f: callable ``f(x)``
        Function to evaluate the model.
    M: int
        Number of predictions the model makes.
    N: int
        Number of parameters the model takes.
    deriv_fn: callable ``deriv_fn(f, x, v, h)`` (optional)
        A function to compute the directional derivative of the model.
    h: float (optional)
        Finite difference step size.

    Returns
    -------
    jacobian: callable ``jacobian(x)``
        A function that evaluates the Jacobian of the model ``r`` at parameter
        ``x``.

    Notes
    -----
    If the predictions are weighted differently, the argument ``f`` should be
    the residual function, assuming that the weights are encoded in the
    residual.
    """

    def jacobian(x):
        # This matrix will be used to specify the direction of the derivative
        eye = np.diag(np.ones(N))
        # Compute the jacobian matrix
        jac = np.empty((M, N))
        for ii, v in enumerate(eye):
            jac[:, ii] = deriv_fn(f, x, v, h)
        return jac

    return jacobian


def Avv_func(f, Avv_fn=AvvCD, h=0.1):
    """A wrapper that create a function to compute directional second
    derivative of the model.

    Parameters
    ----------
    f: callable ``f(x)``
        Function to evaluate the model.
    Avv_fn: callable ``Avv_fn(f, x, v, h)`` (optional)
        A function to compute the directional second derivative of the model.
    h: float (optional)
        Finite difference step size.

    Returns
    -------
    Avv: callable ``Avv(x, v)``
        A function that evaluates the directional second derivative (Avv) of
        the model ``r`` at parameter ``x`` in the ``v`` direction.

    Notes
    -----
    If the predictions are weighted differently, the argument ``f`` should be
    the residual function, assuming that the weights are encoded in the
    residual.
    """

    def Avv(x, v):
        return Avv_fn(f, x, v, h)

    return Avv
