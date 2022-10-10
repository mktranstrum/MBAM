import numpy as np
import mbam.finite_difference as fd
from mbam.utils import initial_velocity

from model_linear import M, N, f, J, xi


_, _, vh = np.linalg.svd(J)
jac_fn = fd.jacobian_func(f, M, N)
Avv_fn = fd.Avv_func(f)


def test_initial_velocity():
    """We test the utility function to get initial velocity using a linear
    model. Since Avv for a linear model is zero, then the initial velocities
    should be parallel to the right singular vectors of the Jacobian. The
    direction might be different, though, since the ``initial_velocity``
    returns the direction in which the velocity increases.
    """
    for i in range(N):
        for forward in [True, False]:
            v = initial_velocity(xi, jac_fn, Avv_fn, i=i, forward=forward)
            v_ref = vh[-1 - i] * (int(forward) * 2 - 1)
            assert np.isclose(np.abs(v @ v_ref), 1.0)


if __name__ == "__main__":
    test_initial_velocity()
