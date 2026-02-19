"""
Radon Contextual Effects (Minnesota, 91 dims): NS-SwiG MWE.

Model (from inference_gym):
  county_effect_mean ~ N(0, 1)
  county_effect_scale ~ Uniform(0, 100)
  county_effect[j] ~ N(county_effect_mean, county_effect_scale)  j=0..84
  weight[k] ~ N(0, 1)  k=0..2
  log_radon_scale ~ Uniform(0, 100)

  log_radon[n] ~ N(
      log_uranium[n]*weight[0] + floor[n]*weight[1] +
      floor_by_county[n]*weight[2] + county_effect[county[n]],
      log_radon_scale
  )

946 observations across J=85 counties.
SwiG: psi = (cem, ces_unc, w0, w1, w2, lrs_unc) in R^6, theta_j = county_effect[j].
likelihood_depends_on_psi = True (weights and log_radon_scale appear in likelihood).

Model and data originate from the TensorFlow Probability inference_gym
(https://pypi.org/project/inference-gym/). The model has been rewritten to
expose the SwiG hierarchical structure and the data extracted into a standalone
.npz file. Original model references and specification are available at:
  inference_gym.targets.radon_contextual_effects
"""

import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from swig import swg
from utils import posterior_weights, resample, run_swig

# --- Data (bundled .npz, extracted from inference_gym) ---
_data = np.load(os.path.join(os.path.dirname(__file__), "radon_minnesota_data.npz"))
LOG_RADON = np.asarray(_data["log_radon"], dtype=np.float64)
LOG_URANIUM = np.asarray(_data["log_uranium"], dtype=np.float64)
FLOOR = np.asarray(_data["floor"], dtype=np.float64)
COUNTY = np.asarray(_data["county"], dtype=np.int32)
FLOOR_BY_COUNTY = np.asarray(_data["floor_by_county"], dtype=np.float64)
J = int(_data["num_counties"])  # 85

# Group data by county: padded arrays (J, max_obs, 5) with mask
_obs_per_county = [int((COUNTY == j).sum()) for j in range(J)]
MAX_OBS = max(_obs_per_county)
_dp = np.zeros((J, MAX_OBS, 4), dtype=np.float64)
_mask = np.zeros((J, MAX_OBS), dtype=np.float64)
for j in range(J):
    idx = np.where(COUNTY == j)[0]
    n_j = len(idx)
    _dp[j, :n_j, 0] = LOG_RADON[idx]
    _dp[j, :n_j, 1] = LOG_URANIUM[idx]
    _dp[j, :n_j, 2] = FLOOR[idx]
    _dp[j, :n_j, 3] = FLOOR_BY_COUNTY[idx]
    _mask[j, :n_j] = 1.0
DATA_PER_GROUP = jnp.concatenate([jnp.array(_dp), jnp.array(_mask)[..., None]], axis=-1)


# --- Transforms ---
def scale_to_constrained(unc):
    return 100.0 * jax.nn.sigmoid(unc)

def scale_log_jacobian(unc):
    s = jax.nn.sigmoid(unc)
    return jnp.log(100.0) + jnp.log(s) + jnp.log(1.0 - s)

def inv_scale(y):
    return jnp.log(y / 100.0) - jnp.log(1.0 - y / 100.0)


# --- Model functions ---
def _log_normal(x, mean, std):
    return -0.5 * ((x - mean) ** 2 / std ** 2 + jnp.log(2.0 * jnp.pi * std ** 2))

def logprior_psi_fn(psi):
    cem, ces_unc, w0, w1, w2, lrs_unc = psi[0], psi[1], psi[2], psi[3], psi[4], psi[5]
    uniform_logjac = jnp.log(1.0 / 100.0)
    lp = _log_normal(cem, 0.0, 1.0)
    lp += uniform_logjac + scale_log_jacobian(ces_unc)
    lp += _log_normal(w0, 0.0, 1.0) + _log_normal(w1, 0.0, 1.0) + _log_normal(w2, 0.0, 1.0)
    lp += uniform_logjac + scale_log_jacobian(lrs_unc)
    return lp

def logprior_theta_given_psi_fn(theta_j, psi):
    return jnp.sum(_log_normal(theta_j, psi[0], scale_to_constrained(psi[1])))

def loglikelihood_per_group_fn(theta_j, psi, data_j):
    ce_j = theta_j[0]
    w0, w1, w2 = psi[2], psi[3], psi[4]
    lrs = scale_to_constrained(psi[5])
    mean = data_j[:, 1] * w0 + data_j[:, 2] * w1 + data_j[:, 3] * w2 + ce_j
    return jnp.sum(_log_normal(data_j[:, 0], mean, lrs) * data_j[:, 4])


# --- Prior sampler ---
def sample_prior(key, num_particles):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    cem = jax.random.normal(k1, (num_particles,))
    ces = jax.random.uniform(k2, (num_particles,), minval=1e-4, maxval=100.0)
    weights = jax.random.normal(k3, (num_particles, 3))
    lrs = jax.random.uniform(k4, (num_particles,), minval=1e-4, maxval=100.0)
    psi = jnp.stack([cem, inv_scale(ces), weights[:, 0], weights[:, 1], weights[:, 2], inv_scale(lrs)], axis=-1)
    eps = jax.random.normal(k5, (num_particles, J))
    theta = (cem[:, None] + ces[:, None] * eps)[..., None]
    return {"psi": psi, "theta": theta}


# --- Run ---
def main():
    num_particles = 1000
    num_delete = 50
    termination = -3.0
    seed = 42

    algorithm = swg.as_top_level_api(
        logprior_psi_fn=logprior_psi_fn,
        logprior_theta_given_psi_fn=logprior_theta_given_psi_fn,
        loglikelihood_per_group_fn=loglikelihood_per_group_fn,
        data=DATA_PER_GROUP,
        num_groups=J,
        num_gibbs_sweeps=5,
        num_inner_steps_theta=1,
        num_inner_steps_psi=6,
        num_delete=num_delete,
        likelihood_depends_on_psi=True,
    )

    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)

    key, final_state, logw, elapsed, num_dead = run_swig(
        algorithm, key, sample_prior(init_key, num_particles),
        num_delete=num_delete, termination=termination,
    )
    logzs = jax.scipy.special.logsumexp(logw, axis=0)

    print(f"\nRadon (J={J}) — NS-SwiG results")
    print(f"  Estimated log Z = {float(logzs.mean()):.2f} +/- {float(logzs.std()):.2f}")
    print(f"  Dead points     = {num_dead}")
    print(f"  Wall time       = {elapsed:.1f}s")

    # Plot 1D hyperparameter marginals
    weights = posterior_weights(logw)
    key, rk = jax.random.split(key)
    idx = resample(rk, weights)

    psi = final_state.particles.position["psi"][idx]
    constrained = [
        psi[:, 0],
        scale_to_constrained(psi[:, 1]),
        psi[:, 2], psi[:, 3], psi[:, 4],
        scale_to_constrained(psi[:, 5]),
    ]
    labels = [
        r"county\_effect\_mean", r"county\_effect\_scale",
        r"weight[0]", r"weight[1]", r"weight[2]", r"log\_radon\_scale",
    ]

    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        samples = np.array(constrained[i])
        ax.hist(samples, bins=50, density=True, alpha=0.6)
        ax.set_xlabel(labels[i], fontsize=8)
    fig.suptitle("Radon hyperparameter posteriors")
    fig.tight_layout()
    fig.savefig("plots/radon.pdf", bbox_inches="tight")
    print(f"  Plot saved to plots/radon.pdf")


if __name__ == "__main__":
    main()
