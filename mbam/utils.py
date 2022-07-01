"""A script containing utility functions: get sloppiest initial velocity.
"""


import numpy as np


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
