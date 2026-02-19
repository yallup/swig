"""
Stochastic Volatility (SP500 Small, 103 dims): NS-SwiG Markov MWE.

Model (from inference_gym):
  beta ~ 2*Beta(20, 1.5) - 1      (persistence)
  mu ~ Cauchy(0, 5)                (mean log-volatility)
  sigma ~ HalfCauchy(0, 2)         (shock scale)
  log_vol[0] ~ N(mu, sigma/sqrt(1-beta^2))
  log_vol[t] ~ N(mu + beta*(log_vol[t-1]-mu), sigma)
  y[t] ~ N(0, exp(log_vol[t]/2))

T=100 centered returns from S&P 500.
SwiG Markov: psi = (beta_unc, mu, sigma_unc) in R^3, theta_t = log_vol[t].
likelihood_depends_on_psi = False.

Model and data originate from the TensorFlow Probability inference_gym
(https://pypi.org/project/inference-gym/). The model has been rewritten to
expose the SwiG Markov structure and the data extracted into a standalone
.npz file. Original model references and specification are available at:
  inference_gym.targets.stochastic_volatility_sp500_small
"""

import os
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from swig import swg_markov
from utils import posterior_weights, resample, run_swig

# --- Data (bundled .npz, extracted from inference_gym) ---
_data = np.load(Path(__file__).parent / "sv_sp500_data.npz")
CENTERED_RETURNS = jnp.array(_data["centered_returns"], dtype=jnp.float64)
T = len(CENTERED_RETURNS)  # 100


# --- Transforms ---
sigmoid = jax.nn.sigmoid

def softplus(x):
    return jnp.log1p(jnp.exp(x))

def inv_sigmoid(y):
    return jnp.log(y) - jnp.log(1.0 - y)

def inv_softplus(y):
    return jnp.log(jnp.expm1(y))

def psi_to_constrained(psi):
    beta = 2.0 * sigmoid(psi[..., 0]) - 1.0
    mu = psi[..., 1]
    sigma = softplus(psi[..., 2])
    return beta, mu, sigma


# --- Model functions ---
def _log_normal(x, mean, std):
    return -0.5 * ((x - mean) ** 2 / std ** 2 + jnp.log(2.0 * jnp.pi * std ** 2))

def logprior_psi_fn(psi):
    s = sigmoid(psi[0])
    lp_beta = 19.0 * jnp.log(s) + 0.5 * jnp.log(1.0 - s) - jax.scipy.special.betaln(20.0, 1.5)
    lp_beta += jnp.log(s) + jnp.log(1.0 - s)  # Jacobian
    lp_mu = -jnp.log(jnp.pi * 5.0) - jnp.log(1.0 + (psi[1] / 5.0) ** 2)
    sigma = softplus(psi[2])
    lp_sigma = jnp.log(2.0) - jnp.log(jnp.pi * 2.0) - jnp.log(1.0 + (sigma / 2.0) ** 2)
    lp_sigma += jnp.log(sigmoid(psi[2]))  # Jacobian
    return lp_beta + lp_mu + lp_sigma

def logprior_transition_fn(theta_t, theta_prev, psi, t, T_):
    beta, mu, sigma = psi_to_constrained(psi)
    std0 = sigma / jnp.sqrt(jnp.maximum(1.0 - beta ** 2, 1e-6))
    return jnp.where(
        t == 0,
        jnp.sum(_log_normal(theta_t, mu, std0)),
        jnp.sum(_log_normal(theta_t, mu + beta * (theta_prev - mu), sigma)),
    )

def logprior_site_fn(theta_t, psi, theta_prev, theta_next, t, T_):
    beta, mu, sigma = psi_to_constrained(psi)
    std0 = sigma / jnp.sqrt(jnp.maximum(1.0 - beta ** 2, 1e-6))
    lp_back = jnp.where(
        t == 0,
        jnp.sum(_log_normal(theta_t, mu, std0)),
        jnp.sum(_log_normal(theta_t, mu + beta * (theta_prev - mu), sigma)),
    )
    lp_fwd = jnp.sum(_log_normal(theta_next, mu + beta * (theta_t - mu), sigma))
    return jnp.where(t < T_ - 1, lp_back + lp_fwd, lp_back)

def loglikelihood_per_site_fn(theta_t, psi, data_t):
    return jnp.sum(-0.5 * (theta_t + jnp.log(2.0 * jnp.pi) + data_t ** 2 * jnp.exp(-theta_t)))


# --- Prior sampler ---
def sample_prior(key, num_particles):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    beta_01 = jax.random.beta(k1, 20.0, 1.5, shape=(num_particles,))
    beta = 2.0 * beta_01 - 1.0
    mu = jnp.clip(5.0 * jnp.tan(jnp.pi * (jax.random.uniform(k2, (num_particles,)) - 0.5)), -50.0, 50.0)
    sigma = jnp.clip(2.0 * jnp.abs(jnp.tan(jnp.pi * (jax.random.uniform(k3, (num_particles,)) - 0.5))), 1e-4, 50.0)
    psi = jnp.stack([inv_sigmoid((beta + 1) / 2), mu, inv_softplus(sigma)], axis=-1)

    def sample_chain(key, b, m, s):
        std0 = s / jnp.sqrt(jnp.maximum(1.0 - b ** 2, 1e-6))
        k0, ks = jax.random.split(key)
        lv0 = m + std0 * jax.random.normal(k0)
        def step(carry, k):
            lv = m + b * (carry - m) + s * jax.random.normal(k)
            return lv, lv
        _, lv_rest = jax.lax.scan(step, lv0, jax.random.split(ks, T - 1))
        return jnp.concatenate([lv0[None], lv_rest])

    keys = jax.random.split(k4, num_particles)
    theta = jax.vmap(sample_chain)(keys, beta, mu, sigma)[..., None]
    return {"psi": psi, "theta": theta}


# --- Run ---
def main():
    num_particles = 1000
    num_delete = 50
    termination = -3.0
    seed = 42

    data = CENTERED_RETURNS[:, None]  # (T, 1)

    algorithm = swg_markov.as_top_level_api(
        logprior_psi_fn=logprior_psi_fn,
        logprior_site_fn=logprior_site_fn,
        logprior_transition_fn=logprior_transition_fn,
        loglikelihood_per_group_fn=loglikelihood_per_site_fn,
        data=data,
        num_groups=T,
        num_gibbs_sweeps=5,
        num_inner_steps_psi=3,
        num_inner_steps_theta=1,
        num_delete=num_delete,
        likelihood_depends_on_psi=False,
    )

    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)

    key, final_state, logw, elapsed, num_dead = run_swig(
        algorithm, key, sample_prior(init_key, num_particles),
        num_delete=num_delete, termination=termination,
    )
    logzs = jax.scipy.special.logsumexp(logw, axis=0)

    print(f"\nSV SP500 Small (T={T}) — NS-SwiG Markov results")
    print(f"  Estimated log Z = {float(logzs.mean()):.2f} +/- {float(logzs.std()):.2f}")
    print(f"  Dead points     = {num_dead}")
    print(f"  Wall time       = {elapsed:.1f}s")

    # Plot 1D hyperparameter marginals
    weights = posterior_weights(logw)
    key, rk = jax.random.split(key)
    idx = resample(rk, weights)

    psi = final_state.particles.position["psi"][idx]
    beta, mu, sigma = psi_to_constrained(psi)

    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, samples, label in zip(
        axes,
        [np.array(beta), np.array(mu), np.array(sigma)],
        [r"$\beta$", r"$\mu$", r"$\sigma$"],
    ):
        ax.hist(samples, bins=50, density=True, alpha=0.6)
        ax.set_xlabel(label)
    fig.suptitle("SV hyperparameter posteriors")
    fig.tight_layout()
    fig.savefig("plots/sv.pdf", bbox_inches="tight")
    print(f"  Plot saved to plots/sv.pdf")


if __name__ == "__main__":
    main()
