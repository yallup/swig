"""NS-SwiG for Markov chain (non-iid) latent priors.

Extends SwiG to models where the group-level prior is a Markov chain
(e.g. AR(1) in stochastic volatility) rather than iid given psi.

The prior factorizes as:
    p(theta | psi) = p(theta_0 | psi) * prod_{t=1}^{T-1} p(theta_t | theta_{t-1}, psi)

Each site update uses the Markov blanket (both neighbors) as the
conditional prior, while the per-site likelihood caching and budget
trick from SwiG still apply.

The user provides a `logprior_site_fn(theta_t, psi, theta_prev, theta_next, t, T)`
that returns the sum of prior factors involving theta_t.
"""

from functools import partial
from typing import Callable, Dict, Optional, Union, List

import jax
import jax.numpy as jnp

from blackjax import SamplingAlgorithm
from blackjax.mcmc.ss import SliceState
from blackjax.mcmc.ss import build_kernel as build_slice_kernel
from blackjax.mcmc.ss import sample_direction_from_covariance
from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from blackjax.ns.adaptive import init
from blackjax.ns.base import delete_fn as default_delete_fn
from blackjax.ns.from_mcmc import update_with_mcmc_take_last
from blackjax.types import Array, ArrayTree, PRNGKey

from swig.swg import (
    SwGStateWithLogLikelihood,
    SwGKernelParams,
    SwGInfo,
    _regularize_cov,
    update_inner_kernel_params,
    default_stepper_fn,
)

__all__ = [
    "as_top_level_api",
    "build_kernel",
    "init_markov_state_strategy",
]


def init_markov_state_strategy(
    position: ArrayTree,
    logprior_fn: Callable,
    loglikelihood_per_group_fn: Callable,
    data_array: Array,
    loglikelihood_birth: float = jnp.nan,
) -> SwGStateWithLogLikelihood:
    """Initialize state — identical to iid SwiG init."""
    logprior_values = logprior_fn(position)
    psi = position["psi"]
    theta = position["theta"]

    ll_per_group = jax.vmap(
        lambda theta_j, data_j: loglikelihood_per_group_fn(theta_j, psi, data_j)
    )(theta, data_array)
    loglikelihood_values = ll_per_group.sum()
    loglikelihood_birth_values = loglikelihood_birth * jnp.ones_like(loglikelihood_values)

    return SwGStateWithLogLikelihood(
        position=position,
        logdensity=logprior_values,
        loglikelihood=loglikelihood_values,
        loglikelihood_birth=loglikelihood_birth_values,
        loglikelihood_per_group=ll_per_group,
    )



def build_kernel(
    logprior_psi_fn: Callable,
    logprior_site_fn: Callable,
    logprior_transition_fn: Callable,
    loglikelihood_per_group_fn: Callable,
    data_array: Array,
    num_groups: int,
    num_gibbs_sweeps: int,
    num_inner_steps_psi: int = 1,
    num_inner_steps_theta: int = 1,
    num_delete: int = 1,
    likelihood_depends_on_psi: bool = False,
    update_inner_kernel_params_fn: Callable = update_inner_kernel_params,
    delete_fn: Callable = default_delete_fn,
    max_steps: int = 10,
    max_shrinkage: int = 100,
) -> Callable:
    """Build NS-SwiG kernel for Markov chain latent priors.

    Parameters
    ----------
    logprior_psi_fn
        log p(psi).
    logprior_site_fn
        Markov blanket prior: (theta_t, psi, theta_prev, theta_next, t, T) -> scalar.
        Returns sum of prior factors involving theta_t.
    logprior_transition_fn
        Forward transition: (theta_t, theta_prev, psi, t, T) -> scalar.
        Returns log p(theta_t | theta_{t-1}, psi) for t>0, log p(theta_0 | psi) for t=0.
        Used for computing total log prior (avoids double counting).
    loglikelihood_per_group_fn
        (theta_t, psi, data_t) -> scalar.
    data_array
        Shape (T, ...).
    num_groups
        T, the chain length.
    num_gibbs_sweeps
        Gibbs sweeps per NS iteration.
    num_inner_steps_theta
        Slice steps per site per sweep.
    """
    T = num_groups

    def compute_per_group_likelihoods(theta: Array, psi: Array) -> Array:
        return jax.vmap(
            lambda theta_j, data_j: loglikelihood_per_group_fn(theta_j, psi, data_j)
        )(theta, data_array)

    def compute_total_logprior(position):
        psi = position["psi"]
        theta = position["theta"]
        logprior_psi = logprior_psi_fn(psi)
        # Forward chain: sum_t logprior_transition_fn(theta_t, theta_{t-1}, psi, t, T)
        # All terms are independent (no carry dependency), so vmap for full parallelism.
        dummy = jnp.zeros_like(theta[0])
        theta_prev_arr = jnp.concatenate([dummy[None], theta[:-1]], axis=0)
        idxs = jnp.arange(T)
        lps = jax.vmap(
            lambda th, th_prev, t: logprior_transition_fn(th, th_prev, psi, t, T)
        )(theta, theta_prev_arr, idxs)
        return logprior_psi + lps.sum()

    def constrained_mcmc_fn(
        rng_key: PRNGKey,
        state: SwGStateWithLogLikelihood,
        loglikelihood_0: float,
        cov_psi: Array,
        cov_theta: Array,
    ):
        position = state.position
        psi = position["psi"]
        theta = position["theta"]  # (T, d_theta)
        ll_cache = state.loglikelihood_per_group  # (T,)

        # =================================================================
        # Psi slice step (same as iid SwiG)
        # =================================================================
        def psi_slice_step(rng_key, current_psi, current_theta, current_ll_cache):
            rng_key, prop_key = jax.random.split(rng_key)
            d_psi = sample_direction_from_covariance(prop_key, current_psi, cov_psi)

            current_total_logprior = compute_total_logprior(
                {"psi": current_psi, "theta": current_theta}
            )

            if likelihood_depends_on_psi:
                def slice_fn(t):
                    new_psi = current_psi + t * d_psi
                    new_lp = compute_total_logprior(
                        {"psi": new_psi, "theta": current_theta}
                    )
                    new_ll_cache = compute_per_group_likelihoods(current_theta, new_psi)
                    new_total_ll = new_ll_cache.sum()
                    in_contour = new_total_ll > loglikelihood_0
                    is_valid = jnp.isfinite(new_lp) & in_contour
                    return SliceState(position=new_psi, logdensity=new_lp), is_valid
            else:
                def slice_fn(t):
                    new_psi = current_psi + t * d_psi
                    new_lp = compute_total_logprior(
                        {"psi": new_psi, "theta": current_theta}
                    )
                    is_valid = jnp.isfinite(new_lp)
                    return SliceState(position=new_psi, logdensity=new_lp), is_valid

            slice_kernel = build_slice_kernel(
                slice_fn, max_steps=max_steps, max_shrinkage=max_shrinkage
            )
            init_ss = SliceState(position=current_psi, logdensity=current_total_logprior)
            final_ss, slice_info = slice_kernel(rng_key, init_ss)

            new_psi = final_ss.position
            if likelihood_depends_on_psi:
                new_ll_cache = compute_per_group_likelihoods(current_theta, new_psi)
            else:
                new_ll_cache = current_ll_cache

            return new_psi, new_ll_cache, slice_info

        def psi_multi_step(rng_key, current_psi, current_theta, current_ll_cache):
            """Run num_inner_steps_psi slice steps on psi."""
            def inner_body(carry, rng_key):
                psi_val, ll_cache_val = carry
                new_psi, new_ll_cache, info = psi_slice_step(
                    rng_key, psi_val, current_theta, ll_cache_val
                )
                return (new_psi, new_ll_cache), info

            keys = jax.random.split(rng_key, num_inner_steps_psi)
            (final_psi, final_ll_cache), infos = jax.lax.scan(
                inner_body, (current_psi, current_ll_cache), keys
            )
            return final_psi, final_ll_cache, infos

        # =================================================================
        # Theta site slice step with Markov blanket prior + budget constraint
        # =================================================================
        def theta_site_slice_step(
            rng_key, current_theta_t, current_psi, data_t,
            ll_t_current, ll_sum, cov_theta_t,
            theta_prev, theta_next, t,
        ):
            rng_key, prop_key = jax.random.split(rng_key)
            d_theta_t = sample_direction_from_covariance(
                prop_key, current_theta_t, cov_theta_t
            )

            ll_others = ll_sum - ll_t_current
            budget_t = loglikelihood_0 - ll_others

            current_logprior_site = logprior_site_fn(
                current_theta_t, current_psi, theta_prev, theta_next, t, T
            )

            def slice_fn(step_scale):
                new_theta_t = current_theta_t + step_scale * d_theta_t
                new_logprior_site = logprior_site_fn(
                    new_theta_t, current_psi, theta_prev, theta_next, t, T
                )
                new_ll_t = loglikelihood_per_group_fn(new_theta_t, current_psi, data_t)
                in_contour = new_ll_t > budget_t
                is_valid = jnp.isfinite(new_logprior_site) & in_contour
                return SliceState(position=new_theta_t, logdensity=new_logprior_site), is_valid

            slice_kernel = build_slice_kernel(
                slice_fn, max_steps=max_steps, max_shrinkage=max_shrinkage
            )
            init_ss = SliceState(position=current_theta_t, logdensity=current_logprior_site)
            final_ss, slice_info = slice_kernel(rng_key, init_ss)

            new_theta_t = final_ss.position
            new_ll_t = loglikelihood_per_group_fn(new_theta_t, current_psi, data_t)
            return new_theta_t, new_ll_t, slice_info

        def theta_site_multi_step(
            rng_key, current_theta_t, current_psi, data_t,
            ll_t, ll_sum, cov_theta_t,
            theta_prev, theta_next, t,
        ):
            def inner_body(carry, rng_key):
                theta_t, ll_t_inner, ll_sum_inner = carry
                new_theta_t, new_ll_t, info = theta_site_slice_step(
                    rng_key, theta_t, current_psi, data_t,
                    ll_t_inner, ll_sum_inner, cov_theta_t,
                    theta_prev, theta_next, t,
                )
                new_ll_sum = ll_sum_inner - ll_t_inner + new_ll_t
                return (new_theta_t, new_ll_t, new_ll_sum), info

            keys = jax.random.split(rng_key, num_inner_steps_theta)
            (final_theta_t, final_ll_t, final_ll_sum), infos = jax.lax.scan(
                inner_body, (current_theta_t, ll_t, ll_sum), keys
            )
            return final_theta_t, final_ll_t, final_ll_sum, infos

        # =================================================================
        # One Gibbs sweep: 1 psi step + sequential theta sweep with neighbors
        # =================================================================
        def one_gibbs_sweep(carry, rng_key):
            current_psi, current_theta, current_ll_cache = carry
            rng_key, psi_key, theta_key = jax.random.split(rng_key, 3)

            # --- Psi steps ---
            new_psi, new_ll_cache, psi_info = psi_multi_step(
                psi_key, current_psi, current_theta, current_ll_cache
            )

            # --- Theta sweep over sites with neighbor access ---
            # Carry theta_prev through scan; theta_next comes from shifted input.
            # This avoids .at[t].set() scatter ops.
            ll_sum = new_ll_cache.sum()
            dummy = jnp.zeros_like(current_theta[0])

            # theta_next for site t is current_theta[t+1] (not yet updated).
            # Pad with dummy at end.
            theta_next_arr = jnp.concatenate(
                [current_theta[1:], dummy[None]], axis=0
            )  # (T, d_theta)

            # Site index array for boundary logic
            site_idxs = jnp.arange(T)

            def group_update_body(carry, inputs):
                ll_sum_val, psi_val, key, theta_prev = carry
                theta_t, ll_t, data_t, cov_t, theta_next, t = inputs

                key, subkey = jax.random.split(key)
                new_theta_t, new_ll_t, new_ll_sum, infos = theta_site_multi_step(
                    subkey, theta_t, psi_val, data_t,
                    ll_t, ll_sum_val, cov_t,
                    theta_prev, theta_next, t,
                )

                # Updated theta_t becomes theta_prev for next site
                return (new_ll_sum, psi_val, key, new_theta_t), (new_theta_t, new_ll_t, infos)

            init_carry = (ll_sum, new_psi, theta_key, dummy)
            scan_inputs = (
                current_theta, new_ll_cache, data_array,
                cov_theta, theta_next_arr, site_idxs,
            )
            (final_ll_sum, _, _, _), (final_theta, final_ll_cache, theta_infos) = jax.lax.scan(
                group_update_body, init_carry, scan_inputs
            )

            sweep_info = SwGInfo(psi_info=psi_info, theta_info=theta_infos)
            return (new_psi, final_theta, final_ll_cache), sweep_info

        # =================================================================
        # Run sweeps
        # =================================================================
        sweep_keys = jax.random.split(rng_key, num_gibbs_sweeps)
        (final_psi, final_theta, final_ll_cache), all_infos = jax.lax.scan(
            one_gibbs_sweep, (psi, theta, ll_cache), sweep_keys
        )

        final_position = {"psi": final_psi, "theta": final_theta}
        final_logprior = compute_total_logprior(final_position)
        final_loglikelihood = final_ll_cache.sum()

        final_state = SwGStateWithLogLikelihood(
            position=final_position,
            logdensity=final_logprior,
            loglikelihood=final_loglikelihood,
            loglikelihood_birth=jnp.array(loglikelihood_0),
            loglikelihood_per_group=final_ll_cache,
        )
        return final_state, all_infos

    inner_kernel = update_with_mcmc_take_last(constrained_mcmc_fn, num_mcmc_steps=1)
    delete_fn_partial = partial(delete_fn, num_delete=num_delete)
    kernel = build_adaptive_kernel(
        delete_fn_partial,
        inner_kernel,
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
    )
    return kernel


def as_top_level_api(
    logprior_psi_fn: Callable,
    logprior_site_fn: Callable,
    logprior_transition_fn: Callable,
    loglikelihood_per_group_fn: Callable,
    data: Union[List, Array],
    num_groups: int,
    num_gibbs_sweeps: int = 1,
    num_inner_steps_psi: int = 1,
    num_inner_steps_theta: int = 1,
    num_delete: int = 1,
    likelihood_depends_on_psi: bool = False,
    update_inner_kernel_params_fn: Callable = update_inner_kernel_params,
    delete_fn: Callable = default_delete_fn,
    max_steps: int = 10,
    max_shrinkage: int = 100,
) -> SamplingAlgorithm:
    """NS-SwiG for Markov chain latent priors."""
    if isinstance(data, list):
        data_array = jnp.stack(data)
    else:
        data_array = jnp.asarray(data)

    T = num_groups

    def compute_total_logprior(position):
        psi = position["psi"]
        theta = position["theta"]
        logprior_psi = logprior_psi_fn(psi)
        dummy = jnp.zeros_like(theta[0])
        theta_prev_arr = jnp.concatenate([dummy[None], theta[:-1]], axis=0)
        idxs = jnp.arange(T)
        lps = jax.vmap(
            lambda th, th_prev, t: logprior_transition_fn(th, th_prev, psi, t, T)
        )(theta, theta_prev_arr, idxs)
        return logprior_psi + lps.sum()

    def init_state_fn(position, loglikelihood_birth=jnp.nan):
        return init_markov_state_strategy(
            position=position,
            logprior_fn=compute_total_logprior,
            loglikelihood_per_group_fn=loglikelihood_per_group_fn,
            data_array=data_array,
            loglikelihood_birth=loglikelihood_birth,
        )

    kernel = build_kernel(
        logprior_psi_fn,
        logprior_site_fn,
        logprior_transition_fn,
        loglikelihood_per_group_fn,
        data_array,
        T,
        num_gibbs_sweeps,
        num_inner_steps_psi=num_inner_steps_psi,
        num_inner_steps_theta=num_inner_steps_theta,
        num_delete=num_delete,
        likelihood_depends_on_psi=likelihood_depends_on_psi,
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
        delete_fn=delete_fn,
        max_steps=max_steps,
        max_shrinkage=max_shrinkage,
    )

    def init_fn(position, rng_key=None):
        return init(
            position,
            init_state_fn=jax.vmap(init_state_fn),
            update_inner_kernel_params_fn=update_inner_kernel_params_fn,
        )

    def step_fn(rng_key, state):
        return kernel(rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)
