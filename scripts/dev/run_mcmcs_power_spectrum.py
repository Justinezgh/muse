# -*- coding: utf-8 -*-

import os
import argparse
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import numpyro
from numpyro.handlers import seed, condition
import jax_cosmo as jc
import lenstools as lt
import astropy.units as u
import numpyro.distributions as dist
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from jax.lib import xla_bridge; 
print(xla_bridge.get_backend().platform)


try:
    from azureml.core import Run
    run = Run.get_context()
    ON_AZURE = True
except ImportError:
    ON_AZURE = False

os.makedirs("./outputs", exist_ok=True)


# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=128)
parser.add_argument("--map_size", type=int, default=5)
parser.add_argument("--sigma_e", type=float, default=0.2)
parser.add_argument("--gal_per_arcmin2", type=int, default=30)
parser.add_argument("--non_gaussianity", type=float, default=1)
parser.add_argument("--num_results", type=int, default=10000)
parser.add_argument("--seed", type=int, default=0)
# parser.add_argument("--shift")

args = parser.parse_args()

if ON_AZURE:
    run.log('N', args.N)
    run.log('map_size', args.map_size)
    run.log('sigma_e', args.sigma_e)
    run.log('gal_per_arcmin2', args.gal_per_arcmin2)
    run.log('non_gaussianity', args.non_gaussianity)
    run.log('num_results', args.num_results)
    run.log('seed', args.seed)
    # run.log('shift', args.shift)

else:
    print(args)


lognormal_params = np.loadtxt(
    "lognormal_shift.csv",
    skiprows=1,
    delimiter=','
  )

lognormal_params[:, 2] = jnp.linspace(0.02, 0.1, 64)#args.shift

lognormal_params_ = lognormal_params.reshape([8, 8, 3])


@jax.jit
def shift_fn(omega_m, sigma_8):

    omega_m = jnp.atleast_1d(omega_m)
    sigma_8 = jnp.atleast_1d(sigma_8)
    lambda_shift = map_coordinates(
        lognormal_params_[:, :, 2],
        jnp.stack([
            (omega_m - 0.2) / 0.2 * 8 - 0.5,
            (sigma_8 - 0.6) / 0.4 * 8 - 0.5
        ], axis=0).reshape([2, -1]),
        order=1,
        mode='nearest'
    ).squeeze()

    return lambda_shift


def make_power_map(pk_fn, N, map_size, zero_freq_val=0.0):

    k = 2 * jnp.pi * jnp.fft.fftfreq(N, d=map_size / N)
    kcoords = jnp.meshgrid(k, k)
    k = jnp.sqrt(kcoords[0]**2 + kcoords[1]**2)
    ps_map = pk_fn(k)
    ps_map = ps_map.at[0, 0].set(zero_freq_val)
    power_map = ps_map * (N / map_size)**2
    return power_map


def make_lognormal_power_map(power_map, shift, zero_freq_val=0.0):

    power_spectrum_for_lognorm = jnp.fft.ifft2(power_map).real
    power_spectrum_for_lognorm = jnp.log(
        1 + power_spectrum_for_lognorm / shift**2
    )
    power_spectrum_for_lognorm = jnp.abs(
        jnp.fft.fft2(power_spectrum_for_lognorm))
    power_spectrum_for_lognorm = power_spectrum_for_lognorm.at[0, 0].set(0.)
    return power_spectrum_for_lognorm


def lensingLogNormal(
        N=128,
        map_size=5,
        gal_per_arcmin2=10,
        sigma_e=0.26,
        model_type='lognormal',
        with_noise=True,
        non_gaussianity=1
):

    pix_area = (map_size * 60 / N)**2
    map_size = map_size / 180 * jnp.pi

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
            loc=jnp.zeros((N, N)),
            precision_matrix=jnp.eye(N)
        )
    )

    power_map = make_power_map(P, N, map_size)

    if model_type == 'lognormal':
        shift = shift_fn(cosmo.Omega_m, sigma_8) #non_gaussianity
        power_map = make_lognormal_power_map(power_map, shift)
    field = jnp.fft.ifft2(jnp.fft.fft2(z) * jnp.sqrt(power_map)).real
    if model_type == 'lognormal':
        field = shift * (jnp.exp(field - jnp.var(field) / 2) - 1)

    if with_noise:
        x = numpyro.sample(
            'y',
            dist.Independent(
                dist.Normal(
                    field,
                    sigma_e / jnp.sqrt(gal_per_arcmin2 * pix_area)
                ),
                2
            )
        )
    else:
        x = numpyro.deterministic('y', field)

    return x


# define lensing model
model = partial(lensingLogNormal,
                N=args.N,
                map_size=args.map_size,
                gal_per_arcmin2=args.gal_per_arcmin2,
                sigma_e=args.sigma_e,
                model_type='lognormal',
                non_gaussianity=args.non_gaussianity)

# condition the model on a given set of parameters
fiducial_model = condition(model, {'omega_c': 0.3, 'sigma_8': 0.8})

# sample a mass map
sample_map_fiducial = seed(fiducial_model, jax.random.PRNGKey(args.seed))
m_data = sample_map_fiducial()

plt.imshow(m_data)
plt.savefig('./outputs/m_data.png')

# check power spectrum

cosmo = jc.Planck15(Omega_c=0.3, sigma8=0.8)
# Creating a given redshift distribution
pz = jc.redshift.smail_nz(0.5, 2., 1.0, gals_per_arcmin2=30.)
tracer = jc.probes.WeakLensing([pz], sigma_e=0.2)
f_sky = 5**2/41_253

kmap_lt = lt.ConvergenceMap(m_data, 5*u.deg)
l_edges = np.arange(100.0, 5000.0, 50.0)
l2, Pl2 = kmap_lt.powerSpectrum(l_edges)

cell = jc.angular_cl.angular_cl(cosmo, l2, [tracer])[0]
cell_noise = jc.angular_cl.noise_cl(l2, [tracer])[0]
_, C = jc.angular_cl.gaussian_cl_covariance_and_mean(
   cosmo,
   l2,
   [tracer],
   f_sky=f_sky
)

plt.figure(figsize=[10, 10])
plt.loglog(l2, cell, label='Theory')
plt.loglog(l2, cell + cell_noise, label='Theory+Noise')
plt.loglog(l2, cell_noise, label='Noise')
plt.loglog(l2, Pl2)
plt.legend()
plt.savefig('./outputs/power_spectrum.png')

cosmo = jc.Planck15(Omega_c=0.3, sigma8=0.8)
pz = jc.redshift.smail_nz(
    0.5,
    2.,
    1.0,
    gals_per_arcmin2=args.gal_per_arcmin2
)
tracer = jc.probes.WeakLensing([pz], sigma_e=args.sigma_e)
f_sky = args.map_size**2 / 41_253
kmap_lt = lt.ConvergenceMap(m_data, args.map_size * u.deg)
l_edges = np.arange(100.0, 5000.0, 50.0)
l2, Pl2 = kmap_lt.powerSpectrum(l_edges)
cell_noise = jc.angular_cl.noise_cl(l2, [tracer])[0]
_, C = jc.angular_cl.gaussian_cl_covariance_and_mean(
    cosmo,
    l2,
    [tracer],
    f_sky=f_sky
)


@jax.jit
@jax.vmap
def log_prob_fn(params):
    cosmo = jc.Planck15(Omega_c=params[0] * 0.05 + 0.3,
                        sigma8=params[1] * 0.05 + 0.8)
    cell = jc.angular_cl.angular_cl(cosmo, l2, [tracer])[0]
    prior = tfd.MultivariateNormalDiag(
        loc=jnp.zeros(2),
        scale_identity_multiplier=1.
    )
    likelihood = tfd.MultivariateNormalDiag(
        cell,
        scale_diag=jnp.sqrt(np.diag(C))
    )
    logp = prior.log_prob(params) + likelihood.log_prob(Pl2 - cell_noise)
    return logp


# Initialize the HMC transition kernel.
num_burnin_steps = int(3e2)
adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
    tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob_fn,
        num_leapfrog_steps=3,
        step_size=0.07
    ),
    num_adaptation_steps=int(num_burnin_steps * 0.8))

# Run the chain (with burn-in).
nb_parallel = 50
num_results = args.num_results // nb_parallel
samples, is_accepted = tfp.mcmc.sample_chain(
    num_results=num_results,
    num_burnin_steps=num_burnin_steps,
    current_state=np.random.randn(nb_parallel, 2),
    kernel=adaptive_hmc,
    trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
    seed=jax.random.PRNGKey(0))

theta = samples[is_accepted] * 0.05 + jnp.array([0.3, 0.8])


plt.figure(figsize=[10, 10])
plt.scatter(theta[:, 0], theta[:, 1], c=np.arange(len(theta)))
plt.axvline(0.3)
plt.axhline(0.8)
plt.xlabel('Omega_c')
plt.ylabel('sigma_8')
plt.xlim(0.15, 0.5)
plt.ylim(0.6, 1.)
plt.savefig('./outputs/contour.png')

jnp.save('./outputs/m_data.npy', m_data)
jnp.save('./outputs/posterior.npy', theta)
