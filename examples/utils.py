"""Shared utilities for NS-SwiG examples."""

import time

import jax
import jax.numpy as jnp
from blackjax.ns import utils as ns_utils


def run_swig(algorithm, key, initial_positions, *, num_delete,
             termination=-3.0, log_every=100):
    """Run NS-SwiG: compile, iterate to convergence, finalise, return results.

    Returns (key, final_state, logw, elapsed, num_dead).
    """
    state = algorithm.init(initial_positions)

    @jax.jit
    def one_step(carry, _xs):
        state, k = carry
        k, subk = jax.random.split(k)
        state, dead_info = algorithm.step(subk, state)
        # dead_info retains the full joint (psi + all theta) for every deleted
        # particle.  For large models this can consume significant memory.
        # Consider saving dead_info to disk per iteration, or stripping theta
        # and keeping only psi if the local parameters are not needed.
        return (state, k), dead_info

    # JIT warmup
    key, warmup_key = jax.random.split(key)
    print("Compiling...")
    (state, key), _ = jax.block_until_ready(one_step((state, warmup_key), None))

    # Run until convergence
    print("Running...")
    t0 = time.time()
    dead = []
    n = 0
    while state.integrator.logZ_live - state.integrator.logZ >= termination:
        (state, key), dead_info = one_step((state, key), None)
        dead.append(dead_info)
        n += 1
        if log_every and n % log_every == 0:
            gap = float(state.integrator.logZ_live - state.integrator.logZ)
            print(f"  iter {n}, logZ_live-logZ={gap:.2f}")
    elapsed = time.time() - t0

    # Finalise and compute weights
    final_state = ns_utils.finalise(state, dead)
    key, logw_key = jax.random.split(key)
    logw = ns_utils.log_weights(logw_key, final_state)

    return key, final_state, logw, elapsed, n * num_delete


def posterior_weights(logw):
    """Convert log-weights to normalised weights, handling NaNs."""
    logw_mean = logw.mean(axis=1)
    logw_safe = jnp.nan_to_num(logw_mean, nan=jnp.nan_to_num(logw_mean).min())
    return jnp.exp(logw_safe - jax.scipy.special.logsumexp(logw_safe))


def resample(key, weights, n=5000):
    """Weighted resampling, returns indices."""
    return jax.random.choice(key, len(weights), shape=(n,), p=weights, replace=True)
