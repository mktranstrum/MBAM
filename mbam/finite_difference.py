"""Contains functions to compute the derivative of the model with finite
difference formula.
"""

import numpy as np


# Forward Difference Formulas
def FD(f, x, v, h):
    """2-point forward difference"""
    return (f(x + h * v) - f(x)) / h


def FD2(f, x, v, h):
    """3-point forward difference"""
    return (-f(x + 2 * h * v) + 4 * f(x + h * v) - 3 * f(x)) / (2 * h)


def FD3(f, x, v, h):
    """4-point forward difference"""
    return (
        f(x + 4 * h * v) - 12 * f(x + 2 * h * v) + 32 * f(x + h * v) - 21 * f(x)
    ) / (12 * h)


def FD4(f, x, v, h):
    """5-point forward difference"""
    return (
        -f(x + 8 * h * v)
        + 28 * f(x + 4 * h * v)
        - 224 * f(x + 2 * h * v)
        + 512 * f(x + h * v)
        - 315 * f(x)
    ) / (168 * h)


# Center Difference Formulas
def CD(f, x, v, h):
    """2-point central difference"""
    return (f(x + h * v) - f(x - h * v)) / (2 * h)


def CD4(f, x, v, h):
    """4-point central difference"""
    return (
        -f(x + 2 * h * v)
        + 8 * f(x + h * v)
        - 8 * f(x - h * v)
        + f(x - 2 * h * v)
    ) / (12 * h)


# Avv Formulas
def AvvCD(f, x, v, h):
    """Directional second derivative Avv with 2-point central difference"""
    return (f(x + h * v) + f(x - h * v) - 2 * f(x)) / (h ** 2)


def AvvCD4(f, x, v, h):
    """Directional second derivative Avv with 4-point central difference"""
    return (
        -f(x + 2 * h * v)
        + 16 * f(x + h * v)
        + 16 * f(x - h * v)
        - f(x - 2 * h * v)
        - 30 * f(x)
    ) / (12 * h ** 2)


# Auv formulas
def AuvCD(f, x, u, v, h):
    """Directional second derivative Auv with 2-point central difference"""
    return (
        f(x + h * u + h * v)
        - f(x + h * u - h * v)
        - f(x - h * u + h * v)
        + f(x - h * u - h * v)
    ) / (4 * h ** 2)


def AuvCD4(f, x, u, v, h):
    """Directional second derivative Auv with 4-point central difference"""
    return (
        -f(x + 2 * h * u + 2 * h * v)
        + f(x + 2 * h * u - 2 * h * v)
        + 16 * f(x + h * u + h * v)
        - 16 * f(x + h * u - h * v)
        - 16 * f(x - h * u + h * v)
        + 16 * f(x - h * u - h * v)
        + f(x - 2 * h * u + 2 * h * v)
        - f(x - 2 * h * u - 2 * h * v)
    ) / (48 * h ** 2)


################################################################################
# Wrapper functions to convert the finite difference formulae above to the
# functions required by Geodesic.


def jacobian_func(f, M, N, deriv_fn=FD, h=0.1):
    """A wrapper function convert functions to compute directional first
    derivative to the jacobian function needed in :class:`~mbam.geodesic`.

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
        A function that evaluates the Jacobian of the model ``f`` at parameter
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
    """A wrapper function convert functions to compute directional second
    derivative to the Avv function needed in :class:`~mbam.geodesic`.

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
