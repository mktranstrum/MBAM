"""A script containing utility functions: get sloppiest initial velocity.
"""


import numpy as np


def initial_velocity(x, jac, Avv, i=0, forward=True):
    """A routine for calculating the initial velocity.

    Parameters
    ----------
    x: (N,) np.ndarray
        Initial Parameter Values.
    jac: callable ``jac(x)``
        Function for calculating the jacobian.
    Avv: callable ``Avv(x, v)``
        Function for calculating a direction second derivative.
    i: int (optional)
        Index to choose which eigenvector direction to choose: 0 = sloppiest,
        1 = second sloppiest, etc.
    forward: bool (optional)
        A flag where if it is set to ``True``, the direction in which the
        velocity increases around ``x`` is chosen. Otherwise, the opposite
        direction is chosen.

    Returns
    -------
    v:
        Initial velocity vector.
    """

    j = jac(x)
    _, _, vh = np.linalg.svd(j)
    v = vh[-1 - i]
    a = -np.linalg.solve(j.T @ j, j.T @ Avv(x, v))
    # We choose the direction in which the velocity will increase, since this
    # is most likely to find a singularity quickest.
    if v @ a < 0:
        v *= -1

    if forward:
        return v / np.linalg.norm(v)
    else:
        return -v / np.linalg.norm(v)
