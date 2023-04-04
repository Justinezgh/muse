# original code:
# https://github.com/florent-leclercq/correlations_vs_field/blob/main/libLN.py

import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
import numpy as np
from numpy.lib.stride_tricks import as_strided


class ToyLogNormal():

    def _toeplitz(self, c):

        c = jnp.asarray(c).ravel()
        r = c.conjugate()
        vals = np.concatenate((c[::-1], r[1:]))
        out_shp = len(c), len(r)
        n = vals.strides[0]

        return as_strided(vals[len(c)-1:], shape=out_shp, strides=(-n, n))

    def _G_to_LN(self, gaussian, alpha):

        return 1 / alpha * (jnp.exp(alpha * gaussian - 0.5 * alpha ** 2) - 1)

    def _compute_rsquared(self, nside):

        _Di = jnp.tile(self.matrix_t, (nside, nside))

        _Dj = jnp.concatenate([
            jnp.concatenate([
                jnp.tile(jnp.abs(i - j), (nside, nside)) for i in range(nside)
            ], axis=0) for j in range(nside)
        ], axis=1)

        _distance_squared = _Di * _Di + _Dj * _Dj

        return _distance_squared

    def _xi_G(self, rsq, beta):

        return jnp.exp(-0.25 * rsq / (beta ** 2))

    def __init__(
        self,
        Lside,
        PixelNoise,
        alpha_min,
        alpha_max,
        beta_min,
        beta_max
    ):
        self.Lside = Lside
        self.PixelNoise = PixelNoise
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.matrix_t = jnp.array(self._toeplitz(jnp.arange(Lside)))
        self.rsq = self._compute_rsquared(self.Lside)

    def numpyro_model(
        self
    ):
        dim = self.Lside
        alpha = numpyro.sample(
            'alpha',
            dist.Uniform(self.alpha_min, self.alpha_max)
        )

        beta = numpyro.sample(
            'beta',
            dist.Uniform(self.beta_min, self.beta_max)
        )

        xiG = self._xi_G(self.rsq, beta)
        A = jax.numpy.linalg.cholesky(xiG)

        z = numpyro.sample(
            'z',
            dist.MultivariateNormal(
                loc=jnp.zeros(dim * dim),
                covariance_matrix=jnp.eye(dim * dim)
            )
        )

        z = jnp.dot(A, z)
        field = self._G_to_LN(z, alpha).reshape(dim, dim)

        x = numpyro.sample(
            'y',
            dist.Independent(
                dist.Normal(
                    field,
                    self.PixelNoise
                ),
                2
            )
        )

        return x
