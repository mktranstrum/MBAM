import numpy as np
from mbam import Geodesic
from mbam.finite_difference import jacobian_func, Avv_func

from model_linear import M, N, f, J, xi, vi


J_fn = jacobian_func(f, M, N)
Avv_fn = Avv_func(f)

# Calculate the geodesic
geo = Geodesic(f, J_fn, Avv_fn, xi, vi)
geo.integrate(1000.0)
len_ts = len(geo.ts)


def test_attributes():
    assert hasattr(geo, "xs")
    assert hasattr(geo, "rs")
    assert hasattr(geo, "vs")
    assert hasattr(geo, "vels")
    assert hasattr(geo, "ts")


def test_derivative_shape():
    assert J_fn(xi).shape == (M, N)
    assert Avv_fn(xi, vi).shape == (M,)


def test_derivative_value():
    assert np.allclose(J_fn(xi), J)
    assert np.allclose(Avv_fn(xi, vi), 0.0)


def test_vs():
    """For a linear model, the velocitY in parameter space should be constant,
    and it should be the same as the initial velocity.
    """
    assert geo.vs.shape == (len_ts, N)
    for vs in geo.vs:
        assert np.allclose(vs, vi)


def test_xs():
    """Since vs is constant, then in each integration step, dx = vi * dt."""
    assert geo.xs.shape == (len_ts, N)
    for ii in range(len_ts - 1):
        dx = vi * (geo.ts[ii + 1] - geo.ts[ii])
        assert np.allclose(dx, (geo.xs[ii + 1] - geo.xs[ii]))


def test_rs():
    assert geo.rs.shape == (len_ts, M)
    for xs, rs in zip(geo.xs, geo.rs):
        assert np.allclose(rs, J @ xs)


def test_vels():
    assert geo.vels.shape == (len_ts, M)
    for vs, vels in zip(geo.vs, geo.vels):
        assert np.allclose(vels, J @ vs)


if __name__ == "__main__":
    test_attributes()
    test_derivative_shape()
    test_derivative_value()
    test_vs()
    test_xs()
    test_rs()
    test_vels()