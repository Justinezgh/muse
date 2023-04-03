import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import numpyro
import jax_cosmo as jc
import numpyro.distributions as dist


class lensing_log_normal():
    def __init__(
        self,
        N,
        map_size,
        gal_per_arcmin2,
        sigma_e,
        shift
    ):
        self.N = N
        self.map_size = map_size
        self.gal_per_arcmin2 = gal_per_arcmin2
        self.sigma_e = sigma_e

        a = np.linspace(0.2, 0.4, 8)
        b = np.linspace(0.6, 1.0, 8)
        lognormal_params = np.array(np.meshgrid(a, b)).T.reshape([64, 2])
        lognormal_params = np.concatenate([
            lognormal_params,
            shift.reshape([64, 1])
        ], axis=1)

        self.shift_table = lognormal_params.reshape([8, 8, 3])

    def shift_fn(self, omega_m, sigma_8):

        omega_m = jnp.atleast_1d(omega_m)
        sigma_8 = jnp.atleast_1d(sigma_8)
        lambda_shift = map_coordinates(
            self.shift_table[:, :, 2],
            jnp.stack([
                (omega_m - 0.2) / 0.2 * 8 - 0.5,
                (sigma_8 - 0.6) / 0.4 * 8 - 0.5
            ], axis=0).reshape([2, -1]),
            order=1,
            mode='nearest'
        ).squeeze()

        return lambda_shift

    def make_power_map(self, pk_fn, N, map_size, zero_freq_val=0.0):

        k = 2 * jnp.pi * jnp.fft.fftfreq(N, d=map_size / N)
        kcoords = jnp.meshgrid(k, k)
        k = jnp.sqrt(kcoords[0]**2 + kcoords[1]**2)
        ps_map = pk_fn(k)
        ps_map = ps_map.at[0, 0].set(zero_freq_val)
        power_map = ps_map * (N / map_size)**2
        return power_map

    def make_lognormal_power_map(self, power_map, shift, zero_freq_val=0.0):

        power_spectrum_for_lognorm = jnp.fft.ifft2(power_map).real
        power_spectrum_for_lognorm = jnp.log(
            1 + power_spectrum_for_lognorm / shift**2
        )
        power_spectrum_for_lognorm = jnp.abs(
            jnp.fft.fft2(power_spectrum_for_lognorm))
        power_spectrum_for_lognorm = power_spectrum_for_lognorm.at[0, 0].set(0.)
        return power_spectrum_for_lognorm

    def numpyro_model(self):

        pix_area = (self.map_size * 60 / self.N)**2
        map_size = self.map_size / 180 * jnp.pi

        omega_c = numpyro.sample('omega_c', dist.Normal(0.3, 0.05))
        sigma_8 = numpyro.sample('sigma_8', dist.Normal(0.8, 0.05))

        cosmo = jc.Planck15(Omega_c=omega_c, sigma8=sigma_8)
        pz = jc.redshift.smail_nz(0.5, 2., 1.0)
        tracer = jc.probes.WeakLensing([pz])
        ell_tab = jnp.logspace(0, 4.5, 128)
        cell_tab = jc.angular_cl.angular_cl(cosmo, ell_tab, [tracer])[0]
        P = lambda k: jc.scipy.interpolate.interp(
            k.flatten(), ell_tab, cell_tab
        ).reshape(k.shape)

        z = numpyro.sample(
            'z',
            dist.MultivariateNormal(
                loc=jnp.zeros((self.N, self.N)),
                precision_matrix=jnp.eye(self.N)
            )
        )

        power_map = self.make_power_map(P, self.N, map_size)

        shift = self.shift_fn(cosmo.Omega_m, sigma_8)
        power_map = self.make_lognormal_power_map(power_map, shift)
        field = jnp.fft.ifft2(jnp.fft.fft2(z) * jnp.sqrt(power_map)).real
        field = shift * (jnp.exp(field - jnp.var(field) / 2) - 1)
        numpyro.deterministic('field', field)

        x = numpyro.sample(
            'y',
            dist.Independent(
                dist.Normal(
                    field,
                    self.sigma_e / jnp.sqrt(self.gal_per_arcmin2 * pix_area)
                ),
                2
            )
        )

        return x