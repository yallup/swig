"""
Hierarchical Gaussian (GLM): NS-SwiG Minimum Working Example.

Model:
    mu ~ N(mu_0, sigma_mu^2)
    theta_j | mu ~ N(mu, sigma_theta^2)   for j = 1..J
    y_j | theta_j ~ N(theta_j, sigma_obs^2)

The likelihood does NOT depend on psi (= mu), so psi updates are free.
The evidence is analytic via conjugacy, enabling validation.
"""

import os

import distrax
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.linalg import solve_triangular, cho_solve

from swig import swg
from utils import run_swig, posterior_weights, resample

# --- Configuration ---
MU_0 = 0.0
SIGMA_MU = 10.0
SIGMA_THETA = 2.0
SIGMA_OBS = 1.0
TAU_SQ = SIGMA_THETA**2 + SIGMA_OBS**2


# --- Analytic evidence ---

def analytic_log_evidence(y):
    """Compute analytic log Z by marginalising theta_j then mu."""
    J = len(y)
    mu_pi = jnp.array([MU_0])
    Sigma_pi = jnp.array([[SIGMA_MU**2]])

    y_mean = jnp.mean(y)
    mu_L = jnp.array([y_mean])
    Sigma_L = jnp.array([[TAU_SQ / J]])

    logLmax = -0.5 * J * jnp.log(2 * jnp.pi * TAU_SQ) - 0.5 * jnp.sum((y - y_mean)**2) / TAU_SQ

    # Cholesky-based log evidence
    L_pi = jnp.linalg.cholesky(Sigma_pi)
    L_L = jnp.linalg.cholesky(Sigma_L)
    prec_pi = cho_solve((L_pi, True), jnp.eye(1))
    prec_L = cho_solve((L_L, True), jnp.eye(1))
    prec_P = prec_pi + prec_L
    L_P = jnp.linalg.cholesky(prec_P)

    b = cho_solve((L_pi, True), mu_pi) + cho_solve((L_L, True), mu_L)
    mu_P = cho_solve((L_P, True), b)

    logdet_Sigma_P = -2 * jnp.sum(jnp.log(jnp.diag(L_P)))
    logdet_Sigma_pi = 2 * jnp.sum(jnp.log(jnp.diag(L_pi)))

    diff_pi = mu_P - mu_pi
    diff_L = mu_P - mu_L
    quad_pi = jnp.sum(jnp.square(solve_triangular(L_pi, diff_pi, lower=True)))
    quad_L = jnp.sum(jnp.square(solve_triangular(L_L, diff_L, lower=True)))

    return float(logLmax + logdet_Sigma_P / 2 - logdet_Sigma_pi / 2 - quad_pi / 2 - quad_L / 2)


# --- Data generation ---

def generate_data(key, J, true_mu=1.5):
    key, k1, k2 = jax.random.split(key, 3)
    true_theta = true_mu + SIGMA_THETA * jax.random.normal(k1, (J,))
    y = true_theta + SIGMA_OBS * jax.random.normal(k2, (J,))
    data = [y[j:j+1] for j in range(J)]
    return data, y


# --- SwiG model functions ---

def logprior_psi_fn(psi):
    return stats.norm.logpdf(psi[0], MU_0, SIGMA_MU)


def logprior_theta_given_psi_fn(theta_j, psi):
    return stats.norm.logpdf(theta_j, psi[0], SIGMA_THETA).sum()


def loglikelihood_per_group_fn(theta_j, psi, data_j):
    return stats.norm.logpdf(data_j, theta_j, SIGMA_OBS).sum()


# --- Prior sampler ---

def sample_prior(key, num_particles, J):
    k1, k2 = jax.random.split(key)
    mu_prior = distrax.Normal(loc=MU_0, scale=SIGMA_MU)
    psi = mu_prior.sample(seed=k1, sample_shape=(num_particles, 1))

    def sample_theta(subkey, mu):
        return distrax.Normal(loc=mu, scale=SIGMA_THETA).sample(
            seed=subkey, sample_shape=(J, 1)
        )

    theta_keys = jax.random.split(k2, num_particles)
    theta = jax.vmap(sample_theta)(theta_keys, psi[:, 0])
    return {"psi": psi, "theta": theta}


# --- Run ---

def main():
    J = 100
    num_particles = 1000
    num_delete = 500
    termination = -3.0
    seed = 123

    key = jax.random.key(seed)
    key, data_key = jax.random.split(key)
    data, y = generate_data(data_key, J)

    true_logZ = analytic_log_evidence(y)

    algorithm = swg.as_top_level_api(
        logprior_psi_fn=logprior_psi_fn,
        logprior_theta_given_psi_fn=logprior_theta_given_psi_fn,
        loglikelihood_per_group_fn=loglikelihood_per_group_fn,
        data=data,
        num_groups=J,
        num_gibbs_sweeps=1,
        num_inner_steps_theta=1,
        num_delete=num_delete,
        likelihood_depends_on_psi=False,
    )

    key, init_key = jax.random.split(key)
    initial_positions = sample_prior(init_key, num_particles, J)

    key, final_state, logw, elapsed, num_dead = run_swig(
        algorithm, key, initial_positions,
        num_delete=num_delete, termination=termination,
    )
    logzs = jax.scipy.special.logsumexp(logw, axis=0)
    logZ_mean = float(logzs.mean())
    logZ_std = float(logzs.std())

    print(f"Hierarchical Gaussian (J={J}) — NS-SwiG results")
    print(f"  Analytic log Z  = {true_logZ:.2f}")
    print(f"  Estimated log Z = {logZ_mean:.2f} +/- {logZ_std:.2f}")
    print(f"  Difference      = {logZ_mean - true_logZ:.2f}")
    print(f"  Dead points     = {num_dead}")
    print(f"  Wall time       = {elapsed:.1f}s")

    # Weighted posterior scatter: mu vs theta_0
    positions = final_state.particles.position
    mu_all = positions["psi"][:, 0]
    theta0_all = positions["theta"][:, 0, 0]
    weights = posterior_weights(logw)
    key, rk = jax.random.split(key)
    idx = resample(rk, weights, n=2000)

    # Analytic contours: p(mu, theta_1 | y) = p(mu | y) * p(theta_1 | mu, y_1)
    prec_prior = 1.0 / SIGMA_MU**2
    prec_lik = J / TAU_SQ
    prec_post = prec_prior + prec_lik
    sigma_mu_post = np.sqrt(1.0 / prec_post)
    mu_post = (np.sum(np.array(y)) / TAU_SQ + MU_0 / SIGMA_MU**2) / prec_post

    prec_th_prior = 1.0 / SIGMA_THETA**2
    prec_th_lik = 1.0 / SIGMA_OBS**2
    prec_th_post = prec_th_prior + prec_th_lik
    sigma_th_post = np.sqrt(1.0 / prec_th_post)
    y1 = float(y[0])

    mu_grid = np.linspace(mu_post - 4*sigma_mu_post, mu_post + 4*sigma_mu_post, 200)
    def th1_mean_fn(m):
        return (m / SIGMA_THETA**2 + y1 / SIGMA_OBS**2) / prec_th_post

    th_lo = th1_mean_fn(mu_post) - 4*sigma_th_post
    th_hi = th1_mean_fn(mu_post) + 4*sigma_th_post
    th_grid = np.linspace(th_lo, th_hi, 200)
    Mu, Th = np.meshgrid(mu_grid, th_grid)

    log_p_mu = -0.5 * ((Mu - mu_post)**2 / sigma_mu_post**2)
    th1_mean = (Mu / SIGMA_THETA**2 + y1 / SIGMA_OBS**2) / prec_th_post
    log_p_th = -0.5 * ((Th - th1_mean)**2 / sigma_th_post**2)
    log_p = log_p_mu + log_p_th
    log_p -= log_p.max()

    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots()
    ax.scatter(mu_all[idx], theta0_all[idx], s=1, alpha=0.3, zorder=1, label="NS-SwiG")
    ax.contour(Mu, Th, np.exp(log_p), levels=[np.exp(-0.5*4), np.exp(-0.5*1)],
               colors="k", linewidths=0.8, zorder=2)
    ax.plot([], [], color="k", linewidth=0.8, label=r"Analytic $1\sigma$, $2\sigma$")
    ax.legend(markerscale=5)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\theta_1$")
    ax.set_title(f"GLM posterior (J={J})")
    fig.savefig("plots/glm.pdf", bbox_inches="tight")
    print(f"  Plot saved to plots/glm.pdf")


if __name__ == "__main__":
    main()
