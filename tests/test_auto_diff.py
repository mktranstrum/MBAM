import pytest

import numpy as np

import model
from model import jac_an, Avv_an, xi, vi

if model.jax_avail:
    import mbam.auto_diff as ad

J = jac_an(xi)
Avv = Avv_an(xi, vi)


@pytest.mark.skipif(not model.jax_avail, reason="JAX not available")
def test_jac_AD():
    J_ad = ad.jacobian_func(model.f_ad)(xi)
    assert np.allclose(J_ad, J, atol=1e-4, rtol=1e-4), "Failed Jacobian AD"


@pytest.mark.skipif(not model.jax_avail, reason="JAX not available")
def test_Avv_AD():
    Avv_ad = ad.Avv_func(model.f_ad)(xi, vi)
    assert np.allclose(Avv_ad, Avv, atol=1e-4, rtol=1e-4), "Failed Avv AD"


if __name__ == "__main__":
    if model.jax_avail:
        test_jac_AD()
        test_Avv_AD()
