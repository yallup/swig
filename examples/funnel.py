"""
10D Funnel: NS-SwiG Minimum Working Example.

Neal's funnel distribution mapped to SwiG's hierarchical structure:
  psi = theta (scalar, controls funnel width)
  theta_j = z_j (D=10 local parameters)
  likelihood per group = log N(z_j | 0, exp(theta/2))

With uniform(-100, 100) priors on z_j, the evidence depends on the prior volume.
"""

import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from swig import swg
from utils import posterior_weights, resample, run_swig

NUM_GROUPS = 10
PRIOR_EXTENT = 100.0


def logprior_psi_fn(psi):
    """N(0, 3) on psi[0]."""
    return -0.5 * (psi[0] ** 2 / 9.0 + jnp.log(2 * jnp.pi * 9.0))


def logprior_theta_given_psi_fn(theta_j, psi):
    """Uniform(-extent, extent) on theta_j[0]."""
    in_bounds = (theta_j[0] >= -PRIOR_EXTENT) & (theta_j[0] <= PRIOR_EXTENT)
    return jnp.where(in_bounds, -jnp.log(2.0 * PRIOR_EXTENT), -jnp.inf)


def loglikelihood_per_group_fn(theta_j, psi, data_j):
    """log N(z_j | 0, exp(psi/2)) — funnel coupling."""
    var = jnp.exp(psi[0])
    return -0.5 * (theta_j[0] ** 2 / var + psi[0] + jnp.log(2.0 * jnp.pi))


def sample_prior(key, num_particles):
    """Sample from the prior: psi ~ N(0,3), theta_j ~ U(-PRIOR_EXTENT, PRIOR_EXTENT)."""
    k1, k2 = jax.random.split(key)
    psi = jax.random.normal(k1, shape=(num_particles, 1)) * 3.0
    theta = jax.random.uniform(
        k2, shape=(num_particles, NUM_GROUPS, 1),
        minval=-PRIOR_EXTENT, maxval=PRIOR_EXTENT,
    )
    return {"psi": psi, "theta": theta}


def main():
    num_particles = 1000
    num_delete = 50
    termination = -3.0
    seed = 42

    data = jnp.zeros((NUM_GROUPS, 1))

    algorithm = swg.as_top_level_api(
        logprior_psi_fn=logprior_psi_fn,
        logprior_theta_given_psi_fn=logprior_theta_given_psi_fn,
        loglikelihood_per_group_fn=loglikelihood_per_group_fn,
        data=data,
        num_groups=NUM_GROUPS,
        num_gibbs_sweeps=5,
        num_inner_steps_theta=1,
        num_delete=num_delete,
        likelihood_depends_on_psi=True,
    )

    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    initial_positions = sample_prior(init_key, num_particles)

    key, final_state, logw, elapsed, num_dead = run_swig(
        algorithm, key, initial_positions,
        num_delete=num_delete, termination=termination,
    )
    logzs = jax.scipy.special.logsumexp(logw, axis=0)
    logz_mean = float(logzs.mean())
    logz_std = float(logzs.std())

    analytic_logz = float(-NUM_GROUPS * jnp.log(2.0 * PRIOR_EXTENT))
    print("10D Funnel — NS-SwiG results")
    print(f"  Analytic log Z  = {analytic_logz:.2f}")
    print(f"  Estimated log Z = {logz_mean:.2f} +/- {logz_std:.2f}")
    print(f"  Difference      = {logz_mean - analytic_logz:.2f}")
    print(f"  Dead points     = {num_dead}")
    print(f"  Wall time       = {elapsed:.1f}s")

    positions = final_state.particles.position
    psi_all = positions["psi"][:, 0]
    theta0_all = positions["theta"][:, 0, 0]
    weights = posterior_weights(logw)
    key, rk = jax.random.split(key)
    idx = resample(rk, weights, n=2000)

    psi_grid = np.linspace(-8, 8, 200)
    z_grid = np.linspace(-30, 30, 200)
    Psi, Z = np.meshgrid(psi_grid, z_grid)
    log_p = -0.5 * (Psi**2 / 9.0 + np.log(2 * np.pi * 9.0)) \
            -0.5 * (Z**2 / np.exp(Psi) + Psi + np.log(2 * np.pi))
    log_p -= log_p.max()

    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots()
    ax.scatter(psi_all[idx], theta0_all[idx], s=1, alpha=0.3, zorder=1, label="NS-SwiG")
    ax.contour(Psi, Z, np.exp(log_p), levels=[np.exp(-0.5*4), np.exp(-0.5*1)],
               colors="k", linewidths=0.8, zorder=2)
    ax.plot([], [], color="k", linewidth=0.8, label=r"Analytic $1\sigma$, $2\sigma$")
    ax.legend(markerscale=5)
    ax.set_xlabel(r"$\psi$ (log-variance)")
    ax.set_ylabel(r"$\theta_1$")
    ax.set_title("Funnel posterior")
    fig.savefig("plots/funnel.pdf", bbox_inches="tight")
    print(f"  Plot saved to plots/funnel.pdf")


if __name__ == "__main__":
    main()
