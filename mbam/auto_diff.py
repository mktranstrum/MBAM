"""Contains functions to compute the derivative of the model with automatic
differentiation using JAX.
"""

import jax.numpy as jnp
from jax import jacfwd


def jacobian_func(f):
    """Generate a function to compute jacobian of a model. The output function
    will be compatible for :class:`~mbam.Geodesic`.

    Parameters
    ----------
    f: callable ``f(x)``
        Function to evaluate the model.

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
    jacobian = jacfwd(f)
    return jacobian


def Avv_func(f):
    """Generate a function to compute directional second derivative Avv of a
    model. The output function will be compatible for :class:`~mbam.Geodesic`.

    Parameters
    ----------
    f: callable ``f(x)``
        Function to evaluate the model.

    Returns
    -------
    Avv: callable ``Avv(x, v)``
        A function that evaluates the directional second derivative (Avv) of
        the model ``f`` at parameter ``x`` in the ``v`` direction.

    Notes
    -----
    If the predictions are weighted differently, the argument ``f`` should be
    the residual function, assuming that the weights are encoded in the
    residual.
    """

    def Avv(x, v):
        def F(s):
            return f(x + v * s)

        return jacfwd(jacfwd(F))(0.0)

    return Avv
