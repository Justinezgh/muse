# -*- coding: utf-8 -*-

import os
import argparse
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpyro
from numpyro.handlers import seed, condition, trace
import numpyro.distributions as dist
from muse_inference.jax import JaxMuseProblem
from muse_inference import MuseResult
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
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--maxsetp", type=int, default=100)
parser.add_argument("--nsims", type=int, default=100)

args = parser.parse_args()

if ON_AZURE:
    run.log('N', args.N)
    run.log('map_size', args.map_size)
    run.log('sigma_e', args.sigma_e)
    run.log('gal_per_arcmin2', args.gal_per_arcmin2)
    run.log('non_gaussianity', args.non_gaussianity)
    run.log('seed', args.seed)
    run.log('maxsetp', args.maxsetp)
    run.log('nsims', args.nsims)


else:
    print(args)


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
        shift = non_gaussianity
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


@jax.jit
def log_likelihood(x, z, theta):
    cond_model = condition(
        model,
        {'omega_c': theta[0], 'sigma_8': theta[1], 'y': x, 'z': z}
    )

    model_trace = trace(cond_model).get_trace()
    log_joint_like = model_trace['y']['fn'].log_prob(
        model_trace['y']['value']
    ).sum()

    log_joint_like += model_trace['z']['fn'].log_prob(
        model_trace['z']['value']
    ).sum()

    return log_joint_like


@jax.jit
def log_prior(theta):
    log_prob = dist.Normal(0.3, 0.05).log_prob(theta[0])
    log_prob += dist.Normal(0.8, 0.05).log_prob(theta[1])

    return log_prob


@jax.jit
def sample_x_and_latent_var(theta, key):
    cond_model = condition(
        seed(model, key),
        {'omega_c': theta[0], 'sigma_8': theta[1]}
    )
    model_trace = trace(cond_model).get_trace()

    return model_trace['y']['value'], model_trace['z']['value']


class LensingProblem(JaxMuseProblem):

    def sample_x_z(self, key, theta):
        return sample_x_and_latent_var(jnp.array(theta), key)

    def logLike(self, x, z, theta):
        return log_likelihood(x, z, jnp.array(theta))

    def logPrior(self, theta):
        return log_prior(theta)


prob = LensingProblem(implicit_diff=False)
prob.set_x(m_data)
result = MuseResult()

theta_start = jnp.array([0.3, 0.8])

prob.solve(
    result=result,
    α=0.2,
    θ_start=theta_start,
    θ_rtol=0,
    z_tol=1e-2,
    progress=True,
    maxsteps=args.maxsetp,
    nsims=args.nsims,
    rng=jax.random.PRNGKey(1)
)

plt.figure(figsize=(10, 8))
plt.plot([np.linalg.norm(h["s̃_post"]) for h in result.history][10:], ".-")
plt.ylabel(r"$|s^{\rm MUSE}|$")
plt.xlabel("step")
plt.ylim(0)
plt.savefig('./outputs/loss.png')


plt.figure(figsize=(8, 8))
x, y = np.transpose([h["θ"] for h in result.history])
plt.plot(x, y, "o")
ds = 0.003
Ns = np.round(
    np.sqrt((x[1:]-x[:-1])**2 + (y[1:]-y[:-1]) ** 2) / ds
).astype(int)
subdiv = lambda x, Ns=Ns: np.concatenate([
    np.linspace(x[ii], x[ii+1], Ns[ii]) for ii, _ in enumerate(x[: -1])
])
x, y = subdiv(x), subdiv(y)
plt.quiver(
    x[:-1],
    y[:-1],
    x[1:]-x[:-1],
    y[1:]-y[:-1],
    scale_units='xy',
    angles='xy',
    scale=1,
    width=.004,
    headlength=6,
    headwidth=6
)

plt.xlabel(r"$\omega_c$")
plt.ylabel(r"$\sigma_8$")
plt.legend()
plt.title('MAP')
plt.savefig('./outputs/map_cvg.png')

theta = result.dist.rvs(1000000)
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
