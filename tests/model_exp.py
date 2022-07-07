import numpy as np

try:
    import jax.numpy as jnp

    jax_avail = True
except ModuleNotFoundError:
    jax_avail = False


np.random.seed(1717)

# Time series
t = np.array([1.0, 2.0, 3.0])
if jax_avail:
    t_ad = jnp.array(t)  # For JAX

# Dimensionality
M = len(t)
N = 2


# Define the model
def f(x):
    """A function that computes the model predictions."""
    return np.exp(-x[0] * t) + np.exp(-x[1] * t)


if jax_avail:

    def f_ad(x):
        """Also a function to compute the predictions, but written for using with
        JAX.
        """
        return jnp.exp(-x[0] * t_ad) + jnp.exp(-x[1] * t_ad)


def jac_an(x):
    """Compute the jacobian analytically. -t*exp(-th*t)"""
    return np.array([-t * np.exp(-x[0] * t), -t * np.exp(-x[1] * t)]).T


def Avv_an(x, v):
    """Compute Avv analytically.t^2 * exp(-th*t)"""
    H = np.array(
        [
            [t ** 2 * np.exp(-x[0] * t), np.zeros(M)],
            [np.zeros(M), t ** 2 * np.exp(-x[1] * t)],
        ]
    )
    H = np.moveaxis(H, (0, 1, 2), (1, 2, 0))
    return H @ v @ v


# Initial conditions
xi = np.exp(np.random.randn(N))
vi = np.random.randn(N)
