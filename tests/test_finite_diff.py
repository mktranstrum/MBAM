import numpy as np
import mbam.finite_difference as fd

from model_exp import M, N, f, jac_an, Avv_an, xi, vi


J = jac_an(xi)
Avv = Avv_an(xi, vi)


def test_jac_FD():
    J_fd = fd.jacobian_func(f, M, N, fd.FD, h=0.01)(xi)
    assert np.allclose(J_fd, J, rtol=1e-1), "Failed Jacobian FD"

    J_fd2 = fd.jacobian_func(f, M, N, fd.FD2, h=0.01)(xi)
    assert np.allclose(J_fd2, J, rtol=1e-1), "Failed Jacobian FD2"

    J_fd3 = fd.jacobian_func(f, M, N, fd.FD3, h=0.01)(xi)
    assert np.allclose(J_fd3, J, rtol=1e-1), "Failed Jacobian FD3"

    J_fd4 = fd.jacobian_func(f, M, N, fd.FD4, h=0.01)(xi)
    assert np.allclose(J_fd4, J, rtol=1e-1), "Failed Jacobian FD4"


def test_jac_CD():
    J_cd = fd.jacobian_func(f, M, N, fd.CD, h=0.01)(xi)
    assert np.allclose(J_cd, J, rtol=1e-1), "Failed Jacobian CD"

    J_cd4 = fd.jacobian_func(f, M, N, fd.CD4, h=0.01)(xi)
    assert np.allclose(J_cd4, J, rtol=1e-1), "Failed Jacobian CD4"


def test_Avv_CD():
    Avv_cd = fd.Avv_func(f, fd.AvvCD, h=0.01)(xi, vi)
    assert np.allclose(Avv_cd, Avv, rtol=1e-1), "Failed Avv CD"

    Avv_cd4 = fd.Avv_func(f, fd.AvvCD4, h=0.01)(xi, vi)
    assert np.allclose(Avv_cd4, Avv, rtol=1e-1), "Failed Avv CD4"


if __name__ == "__main__":
    test_jac_FD()
    test_jac_CD()
    test_Avv_CD()
