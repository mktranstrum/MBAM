import numpy as np
import mbam.finite_difference as fd
import mbam.auto_diff as ad

from model import M, N, f, jac_an, Avv_an, xi, vi


J = jac_an(xi)
Avv = Avv_an(xi, vi)


def test_jac_FD():
    J_fd = fd.jacobian_func(f, M, N, fd.FD, h=0.01)(xi)
    assert np.allclose(J_fd, J, atol=1e-2, rtol=1e-2), "Failed Jacobian FD"

    J_fd2 = fd.jacobian_func(f, M, N, fd.FD2, h=0.01)(xi)
    assert np.allclose(J_fd2, J, atol=1e-2, rtol=1e-2), "Failed Jacobian FD2"

    J_fd3 = fd.jacobian_func(f, M, N, fd.FD3, h=0.01)(xi)
    assert np.allclose(J_fd3, J, atol=1e-2, rtol=1e-2), "Failed Jacobian FD3"

    J_fd4 = fd.jacobian_func(f, M, N, fd.FD4, h=0.01)(xi)
    assert np.allclose(J_fd4, J, atol=1e-2, rtol=1e-2), "Failed Jacobian FD4"


def test_jac_CD():
    J_cd = fd.jacobian_func(f, M, N, fd.CD, h=0.01)(xi)
    assert np.allclose(J_cd, J, atol=1e-2, rtol=1e-2), "Failed Jacobian CD"

    J_cd4 = fd.jacobian_func(f, M, N, fd.CD4, h=0.01)(xi)
    assert np.allclose(J_cd4, J, atol=1e-2, rtol=1e-2), "Failed Jacobian CD4"


def test_Avv_CD():
    Avv_cd = fd.Avv_func(f, fd.AvvCD)(xi, vi)
    assert np.allclose(Avv_cd, Avv, atol=1e-2, rtol=1e-2), "Failed Avv CD"

    Avv_cd4 = fd.Avv_func(f, fd.AvvCD4)(xi, vi)
    assert np.allclose(Avv_cd4, Avv, atol=1e-2, rtol=1e-2), "Failed Avv CD4"


if __name__ == "__main__":
    test_jac_FD()
    test_jac_CD()
    test_Avv_CD()
