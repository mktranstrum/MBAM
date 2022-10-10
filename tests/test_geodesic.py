import numpy as np
from scipy.integrate import cumulative_trapezoid
from mbam import Geodesic
from mbam.finite_difference import jacobian_func, Avv_func

import model_linear as mlin
import model_exp as mexp


# Derivative functions for the linear model
J_fn = jacobian_func(mlin.f, mlin.M, mlin.N)
Avv_fn = Avv_func(mlin.f)

# Calculate the geodesic
# Linear model
geo_lin = Geodesic(mlin.f, J_fn, Avv_fn, mlin.xi, mlin.vi)
geo_lin.integrate(1000.0)
len_ts = len(geo_lin.ts)

# Exponential model with parameterspacenorm=True
geo_exp1 = Geodesic(
    mexp.f, mexp.jac_an, mexp.Avv_an, mexp.xi, mexp.vi, parameterspacenorm=True
)
geo_exp1.integrate(1000.0)
# Exponential model with parameterspacenorm=False
geo_exp2 = Geodesic(
    mexp.f, mexp.jac_an, mexp.Avv_an, mexp.xi, mexp.vi, parameterspacenorm=False
)
geo_exp2.integrate(1000.0)


def test_attributes():
    assert hasattr(geo_lin, "xs")
    assert hasattr(geo_lin, "rs")
    assert hasattr(geo_lin, "vs")
    assert hasattr(geo_lin, "ts")
    assert hasattr(geo_lin, "taus")


def test_derivative_shape():
    assert J_fn(mlin.xi).shape == (mlin.M, mlin.N)
    assert Avv_fn(mlin.xi, mlin.vi).shape == (mlin.M,)


def test_derivative_value():
    assert np.allclose(J_fn(mlin.xi), mlin.J)
    assert np.allclose(Avv_fn(mlin.xi, mlin.vi), 0.0)


def test_vs():
    """For a linear model, the velocitY in parameter space should be constant,
    and it should be the same as the initial velocity.
    """
    assert geo_lin.vs.shape == (len_ts, mlin.N)
    for vs in geo_lin.vs:
        assert np.allclose(vs, mlin.vi)


def test_xs():
    """Since vs is constant, then in each integration step, dx = vi * dt."""
    assert geo_lin.xs.shape == (len_ts, mlin.N)
    for ii in range(len_ts - 1):
        dx = mlin.vi * (geo_lin.ts[ii + 1] - geo_lin.ts[ii])
        assert np.allclose(dx, (geo_lin.xs[ii + 1] - geo_lin.xs[ii]))


def test_rs():
    """To test geo.rs values, we only need to check if geo.rs == J @ geo.xs."""
    assert geo_lin.rs.shape == (len_ts, mlin.M)
    for xs, rs in zip(geo_lin.xs, geo_lin.rs):
        assert np.allclose(rs, mlin.J @ xs)


def test_taus():
    """The product J.v gives dtau/dt. So, we can integrate this quantity and
    compare it to geo.taus.
    """
    assert geo_lin.taus.shape == (len_ts,)
    dtaus = [
        np.linalg.norm(J_fn(xs) @ vs) for xs, vs in zip(geo_lin.xs, geo_lin.vs)
    ]
    taus = np.append(0.0, cumulative_trapezoid(dtaus, geo_lin.ts))
    assert np.allclose(geo_lin.taus, taus)


def test_parameterspacenorm():
    """Test if the the implementation to use constant parameter space norm is
    implemented correctly. This is done by checking the geodesic velocity in
    parameter space. If parameterspacenorm=True, then the geodesic velocity
    should be constant.
    """
    # parameterspacenorm=True
    # To check if the velocity is constant we can compare the speed (the norm
    # of velocity) in each step to the mean speed value over the entire steps.
    # If the velocity is constant, the speed in each step should be close
    # enough to the mean speed, where the deviation is probably from some
    # numerical error.
    geo_exp1_speed = np.linalg.norm(geo_exp1.vs, axis=1)
    geo_exp1_speed_mean = np.mean(geo_exp1_speed)
    assert np.allclose(
        geo_exp1_speed, geo_exp1_speed_mean, atol=1e-4, rtol=1e-4
    )

    # parameterspacenorm=False
    # On the other side, if parameterspacenorm=False, then the velocity in
    # parameter space is not constant. The deviation of the speed from the mean
    # speed should be large.
    geo_exp2_speed = np.linalg.norm(geo_exp2.vs, axis=1)
    geo_exp2_speed_mean = np.mean(geo_exp2_speed)
    assert not np.allclose(geo_exp2_speed, geo_exp2_speed_mean)


if __name__ == "__main__":
    test_attributes()
    test_derivative_shape()
    test_derivative_value()
    test_vs()
    test_xs()
    test_rs()
    test_taus()
    test_parameterspacenorm()
