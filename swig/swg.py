# Copyright 2024 David Yallup
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This code builds on the BlackJAX library (https://github.com/blackjax-devs/blackjax)
"""Nested Sampling with Slice-within-Gibbs (NS-SwG) algorithm.

This module implements nested sampling using a slice-within-Gibbs inner kernel
for hierarchical Bayesian models. The key innovation is exploiting conditional
independence structure with per-group likelihood caching for O(1) constraint
checks per theta_j update (instead of O(J)).

The hierarchical model structure is:
    Y_j | theta_j, psi ~ f(Y_j | theta_j, psi)  # Likelihood per group (may depend on psi)
    theta_j | psi ~ p(theta_j | psi)            # Local params given hyperparams (iid)
    psi ~ p_0(psi)                              # Hyperparameter prior

The NS constraint couples all theta_j through the total likelihood, but we
exploit the additive structure: when updating theta_j, the constraint
    Σ_k log f_k > threshold
becomes
    log f_j > threshold - Σ_{k≠j} log f_k  (the "budget" for group j)

By caching per-group likelihoods, each theta_j update only needs to evaluate
log f_j (not the full sum).

When the likelihood depends on psi:
- psi updates require O(J) likelihood evaluations per candidate (all groups affected)
- theta_j updates still benefit from O(1) constraint checking via the budget trick

When the likelihood is independent of psi:
- psi updates require NO likelihood evaluations (constraint automatically satisfied)
- theta_j updates benefit from O(1) constraint checking

References
----------
.. [1] https://arxiv.org/abs/2403.09416 (dimension-free mixing for hierarchical Gibbs)
"""

from functools import partial
from typing import Callable, Dict, List, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp

from blackjax import SamplingAlgorithm
from blackjax.mcmc.ss import SliceInfo, SliceState
from blackjax.mcmc.ss import build_kernel as build_slice_kernel
from blackjax.mcmc.ss import sample_direction_from_covariance
from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from blackjax.ns.adaptive import init
from blackjax.ns.base import NSInfo, NSState, StateWithLogLikelihood
from blackjax.ns.base import delete_fn as default_delete_fn
from blackjax.ns.base import init_state_strategy
from blackjax.ns.from_mcmc import update_with_mcmc_take_last
from blackjax.types import Array, ArrayTree, PRNGKey

__all__ = [
    "as_top_level_api",
    "build_kernel",
    "init",
    "init_swg_state_strategy",
    "update_inner_kernel_params",
    "SwGKernelParams",
    "SwGInfo",
    "SwGStateWithLogLikelihood",
]


class SwGStateWithLogLikelihood(NamedTuple):
    """State of a particle in NS-SwG with per-group likelihood caching.

    Extends StateWithLogLikelihood with cached per-group likelihoods to enable
    O(1) constraint checking during theta_j updates.

    Attributes
    ----------
    position
        The position of the particle (PyTree with 'psi' and 'theta' keys).
    logdensity
        The log-density of the particle under the prior (Array).
    loglikelihood
        The total log-likelihood of the particle (Array).
    loglikelihood_birth
        The log-likelihood birth threshold for the particle (Array).
    loglikelihood_per_group
        Cached per-group log-likelihoods, shape (J,). Enables O(1) budget
        constraint checking during theta_j updates.
    """

    position: ArrayTree
    logdensity: Array
    loglikelihood: Array
    loglikelihood_birth: Array
    loglikelihood_per_group: Array


class SwGKernelParams(NamedTuple):
    """Covariance blocks for each Gibbs component.

    Attributes
    ----------
    cov_psi
        Covariance matrix for hyperparameters (d_psi, d_psi).
    cov_theta
        Per-group covariance matrices (J, d_theta, d_theta).
    """

    cov_psi: Array
    cov_theta: Array  # Shape: (J, d_theta, d_theta)


class SwGInfo(NamedTuple):
    """Info from Gibbs update step.

    Attributes
    ----------
    psi_info
        SliceInfo from the psi (hyperparameter) update.
    theta_info
        SliceInfo from the theta (local parameters) updates, batched over J groups.
    """

    psi_info: NamedTuple
    theta_info: NamedTuple


def default_stepper_fn(
    x: ArrayTree, d: ArrayTree, t: float
) -> tuple[ArrayTree, bool]:
    """A simple stepper function that moves from `x` along direction `d` by `t` units.

    Implements the operation: `x_new = x + t * d`.
    """
    return jax.tree.map(lambda x, d: x + t * d, x, d), True


def _regularize_cov(cov: Array, eps: float = 1e-6) -> Array:
    """Add small diagonal regularization to prevent singular covariances."""
    return cov + eps * jnp.eye(cov.shape[0])


def init_swg_state_strategy(
    position: ArrayTree,
    logprior_fn: Callable,
    loglikelihood_per_group_fn: Callable,
    data_array: Array,
    loglikelihood_birth: float = jnp.nan,
) -> SwGStateWithLogLikelihood:
    """Initialize a SwG state with per-group likelihood caching.

    This is the SwG-specific version of init_state_strategy that computes
    and caches per-group likelihoods for efficient Gibbs updates.

    Parameters
    ----------
    position
        A PyTree with 'psi' and 'theta' keys representing the particle position.
    logprior_fn
        Function computing the total log-prior: log p_0(psi) + Σ_j log p(theta_j | psi).
    loglikelihood_per_group_fn
        Function computing log f(Y_j | theta_j, psi) for a single group.
        Signature: (theta_j, psi, data_j) -> scalar.
    data_array
        Stacked data array of shape (J, ...) for vectorized access.
    loglikelihood_birth
        The log-likelihood threshold that the particle must exceed. Defaults to NaN.

    Returns
    -------
    SwGStateWithLogLikelihood
        The initialized state with cached per-group likelihoods.
    """
    logprior_values = logprior_fn(position)

    # Compute per-group likelihoods using vmap (no Python loop!)
    psi = position["psi"]
    theta = position["theta"]

    # vmap over (theta_j, data_j) pairs
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


def update_inner_kernel_params(
    rng_key: PRNGKey,
    state: NSState,
    info: NSInfo,
    inner_kernel_params: Optional[Dict[str, ArrayTree]] = None,
) -> Dict[str, ArrayTree]:
    """Update inner kernel parameters from current particles.

    Computes separate empirical covariance matrices for psi (hyperparameters)
    and per-group covariances for theta (local parameters) from the live particles.
    """
    positions = state.particles.position

    # Psi covariance: shape (num_particles, d_psi) -> (d_psi, d_psi)
    psi_all = positions["psi"]
    cov_psi = jnp.atleast_2d(jnp.cov(psi_all, rowvar=False, ddof=0))
    cov_psi = _regularize_cov(cov_psi)

    # Per-group theta covariance: (num_particles, J, d_theta) -> (J, d_theta, d_theta)
    # Each group j gets its own covariance from the num_particles samples
    theta_all = positions["theta"]  # Shape: (num_particles, J, d_theta)

    def compute_group_cov(theta_j_all):
        """Compute covariance for a single group from all particles."""
        # theta_j_all shape: (num_particles, d_theta)
        cov_j = jnp.atleast_2d(jnp.cov(theta_j_all, rowvar=False, ddof=0))
        return _regularize_cov(cov_j)

    # Transpose to (J, num_particles, d_theta) then vmap over groups
    theta_by_group = jnp.transpose(theta_all, (1, 0, 2))
    cov_theta = jax.vmap(compute_group_cov)(theta_by_group)  # Shape: (J, d_theta, d_theta)

    return {"cov_psi": cov_psi, "cov_theta": cov_theta}


def build_kernel(
    logprior_psi_fn: Callable,
    logprior_theta_given_psi_fn: Callable,
    loglikelihood_per_group_fn: Callable,
    data_array: Array,
    num_groups: int,
    num_gibbs_sweeps: int,
    num_inner_steps_theta: int,
    num_inner_steps_psi: int = 1,
    num_delete: int = 1,
    likelihood_depends_on_psi: bool = False,
    stepper_fn: Callable = default_stepper_fn,
    update_inner_kernel_params_fn: Callable = update_inner_kernel_params,
    delete_fn: Callable = default_delete_fn,
    max_steps: int = 10,
    max_shrinkage: int = 100,
) -> Callable:
    """Builds the Nested Sampling Slice-within-Gibbs kernel with likelihood caching.

    This kernel implements a two-block Gibbs structure with O(1) constraint
    checks per theta_j update (instead of O(J)) via per-group likelihood caching.

    Each Gibbs sweep consists of:
    1. Update psi | theta: One slice sampling step on hyperparameters
       - If likelihood_depends_on_psi=False: No constraint checking needed (O(1))
       - If likelihood_depends_on_psi=True: Must recompute all J likelihoods (O(J))
    2. Update theta_j | psi, theta_{-j}: Slice sampling with budget constraint (O(1))

    The kernel runs num_gibbs_sweeps such sweeps, interleaving psi and theta updates.

    Parameters
    ----------
    logprior_psi_fn
        Function computing log p_0(psi).
    logprior_theta_given_psi_fn
        Function computing log p(theta_j | psi). Signature: (theta_j, psi) -> scalar.
    loglikelihood_per_group_fn
        Function computing log f(Y_j | theta_j, psi, data_j).
        Signature: (theta_j, psi, data_j) -> scalar.
    data_array
        Stacked data array of shape (J, ...) for vectorized access.
    num_groups
        Number of groups J.
    num_gibbs_sweeps
        Number of full Gibbs sweeps (1 psi step + 1 theta sweep) per NS iteration.
    num_inner_steps_theta
        Number of slice sampling steps for each theta_j per sweep.
    num_inner_steps_psi
        Number of slice sampling steps for psi per sweep.
    num_delete
        Number of particles to delete and replace at each NS step.
    likelihood_depends_on_psi
        If True, the likelihood depends on psi and constraint checking is needed
        during psi updates (O(J) per candidate). If False, psi updates don't affect
        the likelihood constraint (O(1) per candidate). Default: False.
    stepper_fn
        The stepper function for slice sampling.
    update_inner_kernel_params_fn
        Function to update kernel parameters (covariances).
    delete_fn
        Function to select particles for deletion.
    max_steps
        Maximum steps for slice interval expansion.
    max_shrinkage
        Maximum shrinking steps for slice sampling.

    Returns
    -------
    Callable
        The NS-SwG kernel function.
    """

    def compute_per_group_likelihoods(theta: Array, psi: Array) -> Array:
        """Compute log f_j(theta_j, psi) for all j. Returns shape (J,).

        Uses vmap for efficient vectorization (no Python loop).
        """
        return jax.vmap(
            lambda theta_j, data_j: loglikelihood_per_group_fn(theta_j, psi, data_j)
        )(theta, data_array)

    def compute_total_logprior(position: Dict[str, ArrayTree]) -> Array:
        """Compute log p_0(psi) + Σ_j log p(theta_j | psi)."""
        psi = position["psi"]
        theta = position["theta"]
        logprior_psi = logprior_psi_fn(psi)
        logprior_theta = jax.vmap(lambda th: logprior_theta_given_psi_fn(th, psi))(theta).sum()
        return logprior_psi + logprior_theta

    def constrained_mcmc_swg_fn(
        rng_key: PRNGKey,
        state: SwGStateWithLogLikelihood,
        loglikelihood_0: float,
        cov_psi: Array,
        cov_theta: Array,
    ) -> tuple[SwGStateWithLogLikelihood, SwGInfo]:
        """One Gibbs cycle with per-group likelihood caching."""

        position = state.position
        psi = position["psi"]
        theta = position["theta"]

        # Use cached per-group likelihoods from state (no recomputation needed!)
        ll_cache = state.loglikelihood_per_group  # Shape: (J,)

        # =====================================================================
        # STEP 1: Update psi | theta
        # =====================================================================
        # If likelihood_depends_on_psi=False: constraint L > L* is automatically
        # satisfied throughout psi updates (no constraint checking needed).
        # If likelihood_depends_on_psi=True: must check constraint, requiring
        # O(J) likelihood evaluations per candidate.

        def psi_slice_step(rng_key, current_psi, current_theta, current_ll_cache):
            """Slice update for psi. Returns new psi, updated ll_cache, and slice info."""
            rng_key, prop_key = jax.random.split(rng_key)

            # Sample direction for psi
            d_psi = sample_direction_from_covariance(prop_key, current_psi, cov_psi)

            # Current log-prior for psi block (for slice level)
            current_logprior_psi = logprior_psi_fn(current_psi)
            current_logprior_theta = jax.vmap(
                lambda th: logprior_theta_given_psi_fn(th, current_psi)
            )(current_theta).sum()
            current_total_logprior = current_logprior_psi + current_logprior_theta

            if likelihood_depends_on_psi:
                # Likelihood depends on psi: need to check constraint
                def slice_fn(t):
                    new_psi = current_psi + t * d_psi
                    new_logprior_psi = logprior_psi_fn(new_psi)
                    new_logprior_theta = jax.vmap(
                        lambda th: logprior_theta_given_psi_fn(th, new_psi)
                    )(current_theta).sum()
                    new_total_logprior = new_logprior_psi + new_logprior_theta

                    # Recompute all per-group likelihoods with new psi (O(J))
                    new_ll_cache = compute_per_group_likelihoods(current_theta, new_psi)
                    new_total_ll = new_ll_cache.sum()

                    # Check constraint
                    in_contour = new_total_ll > loglikelihood_0
                    is_valid = jnp.isfinite(new_total_logprior) & in_contour

                    new_state = SliceState(position=new_psi, logdensity=new_total_logprior)
                    return new_state, is_valid
            else:
                # Likelihood independent of psi: constraint automatically satisfied
                def slice_fn(t):
                    new_psi = current_psi + t * d_psi
                    new_logprior_psi = logprior_psi_fn(new_psi)
                    new_logprior_theta = jax.vmap(
                        lambda th: logprior_theta_given_psi_fn(th, new_psi)
                    )(current_theta).sum()
                    new_total_logprior = new_logprior_psi + new_logprior_theta

                    # No likelihood evaluation needed!
                    is_valid = jnp.isfinite(new_total_logprior)

                    new_state = SliceState(position=new_psi, logdensity=new_total_logprior)
                    return new_state, is_valid

            slice_kernel = build_slice_kernel(
                slice_fn, max_steps=max_steps, max_shrinkage=max_shrinkage
            )
            init_slice_state = SliceState(position=current_psi, logdensity=current_total_logprior)
            final_slice_state, slice_info = slice_kernel(rng_key, init_slice_state)

            new_psi = final_slice_state.position

            # Update ll_cache if likelihood depends on psi
            if likelihood_depends_on_psi:
                new_ll_cache = compute_per_group_likelihoods(current_theta, new_psi)
            else:
                new_ll_cache = current_ll_cache  # Unchanged

            return new_psi, new_ll_cache, slice_info

        def psi_multi_step(rng_key, current_psi, current_theta, current_ll_cache):
            """Run num_inner_steps_psi slice steps for psi."""

            def inner_body(carry, rng_key):
                psi, ll_cache = carry
                new_psi, new_ll_cache, info = psi_slice_step(
                    rng_key, psi, current_theta, ll_cache
                )
                return (new_psi, new_ll_cache), info

            keys = jax.random.split(rng_key, num_inner_steps_psi)
            (final_psi, final_ll_cache), infos = jax.lax.scan(
                inner_body, (current_psi, current_ll_cache), keys
            )

            return final_psi, final_ll_cache, infos

        # =====================================================================
        # Helper: theta_j slice step with budget constraint
        # =====================================================================
        def theta_j_slice_step(rng_key, current_theta_j, current_psi, data_j, ll_j_current, ll_sum, cov_theta_j):
            """Slice update for theta_j using budget constraint. O(1) likelihood evals."""
            rng_key, prop_key = jax.random.split(rng_key)

            d_theta_j = sample_direction_from_covariance(prop_key, current_theta_j, cov_theta_j)

            ll_others = ll_sum - ll_j_current
            budget_j = loglikelihood_0 - ll_others

            current_logprior_j = logprior_theta_given_psi_fn(current_theta_j, current_psi)

            def slice_fn(t):
                new_theta_j = current_theta_j + t * d_theta_j
                new_logprior_j = logprior_theta_given_psi_fn(new_theta_j, current_psi)
                new_ll_j = loglikelihood_per_group_fn(new_theta_j, current_psi, data_j)

                in_contour = new_ll_j > budget_j
                is_valid = jnp.isfinite(new_logprior_j) & in_contour

                new_state = SliceState(position=new_theta_j, logdensity=new_logprior_j)
                return new_state, is_valid

            slice_kernel = build_slice_kernel(
                slice_fn, max_steps=max_steps, max_shrinkage=max_shrinkage
            )
            init_slice_state = SliceState(position=current_theta_j, logdensity=current_logprior_j)
            final_slice_state, slice_info = slice_kernel(rng_key, init_slice_state)

            new_theta_j = final_slice_state.position
            new_ll_j = loglikelihood_per_group_fn(new_theta_j, current_psi, data_j)

            return new_theta_j, new_ll_j, slice_info

        def theta_j_multi_step(rng_key, current_theta_j, current_psi, data_j, ll_j, ll_sum, cov_theta_j):
            """Run num_inner_steps_theta slice steps for a single theta_j."""

            def inner_body(carry, rng_key):
                theta_j, ll_j_inner, ll_sum_inner = carry
                new_theta_j, new_ll_j, info = theta_j_slice_step(
                    rng_key, theta_j, current_psi, data_j, ll_j_inner, ll_sum_inner, cov_theta_j
                )
                new_ll_sum = ll_sum_inner - ll_j_inner + new_ll_j
                return (new_theta_j, new_ll_j, new_ll_sum), info

            keys = jax.random.split(rng_key, num_inner_steps_theta)
            (final_theta_j, final_ll_j, final_ll_sum), infos = jax.lax.scan(
                inner_body, (current_theta_j, ll_j, ll_sum), keys
            )

            return final_theta_j, final_ll_j, final_ll_sum, infos

        # =====================================================================
        # One full Gibbs sweep: 1 psi step + 1 theta sweep (over all J groups)
        # =====================================================================
        def one_gibbs_sweep(carry, rng_key):
            """One Gibbs sweep: 1 psi step followed by 1 theta sweep."""
            current_psi, current_theta, current_ll_cache = carry

            # Split keys for psi and theta
            rng_key, psi_key, theta_key = jax.random.split(rng_key, 3)

            # --- Step 1: Psi slice step(s) ---
            new_psi, new_ll_cache, psi_info = psi_multi_step(
                psi_key, current_psi, current_theta, current_ll_cache
            )

            # --- Step 2: Theta sweep over all J groups ---
            # Optimized scan: iterate over arrays directly, not indices
            # This avoids .at[j].set() scatter ops inside the loop
            ll_sum = new_ll_cache.sum()

            def group_update_body(carry, inputs):
                """Update theta_j for group j (optimized: no scatter/gather)."""
                ll_sum_val, psi_val, key = carry
                theta_j, ll_j, data_j, cov_theta_j = inputs

                key, subkey = jax.random.split(key)
                new_theta_j, new_ll_j, new_ll_sum, infos = theta_j_multi_step(
                    subkey, theta_j, psi_val, data_j, ll_j, ll_sum_val, cov_theta_j
                )

                return (new_ll_sum, psi_val, key), (new_theta_j, new_ll_j, infos)

            # Scan over arrays directly - scan stacks outputs into arrays
            (final_ll_sum, _, _), (final_theta, final_ll_cache, theta_infos) = jax.lax.scan(
                group_update_body,
                (ll_sum, new_psi, theta_key),
                (current_theta, new_ll_cache, data_array, cov_theta)
            )

            sweep_info = SwGInfo(psi_info=psi_info, theta_info=theta_infos)
            return (new_psi, final_theta, final_ll_cache), sweep_info

        # =====================================================================
        # Run one Gibbs sweep
        # =====================================================================
        (final_psi, final_theta, final_ll_cache), sweep_info = one_gibbs_sweep(
            (psi, theta, ll_cache), rng_key
        )

        # =====================================================================
        # Construct final state with cached per-group likelihoods
        # =====================================================================
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

        return final_state, sweep_info

    inner_kernel = update_with_mcmc_take_last(constrained_mcmc_swg_fn, num_mcmc_steps=num_gibbs_sweeps, num_delete=num_delete)

    delete_fn_partial = partial(delete_fn, num_delete=num_delete)

    kernel = build_adaptive_kernel(
        delete_fn_partial,
        inner_kernel,
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
    )
    return kernel


def as_top_level_api(
    logprior_psi_fn: Callable,
    logprior_theta_given_psi_fn: Callable,
    loglikelihood_per_group_fn: Callable,
    data: Union[List, Array],
    num_groups: int,
    num_gibbs_sweeps: int = 1,
    num_inner_steps_theta: int = 1,
    num_inner_steps_psi: int = 1,
    num_delete: int = 1,
    likelihood_depends_on_psi: bool = False,
    stepper_fn: Callable = default_stepper_fn,
    update_inner_kernel_params_fn: Callable = update_inner_kernel_params,
    delete_fn: Callable = default_delete_fn,
    max_steps: int = 10,
    max_shrinkage: int = 100,
) -> SamplingAlgorithm:
    """Creates a Nested Sampling Slice-within-Gibbs (NS-SwG) algorithm.

    This algorithm is designed for hierarchical Bayesian models with the structure:
        Y_j | theta_j, psi ~ f(Y_j | theta_j, psi)  # Likelihood per group (may depend on psi)
        theta_j | psi ~ p(theta_j | psi)            # Local params given hyperparams (iid)
        psi ~ p_0(psi)                              # Hyperparameter prior

    Uses per-group likelihood caching for O(1) constraint checks per theta_j
    update (instead of O(J) for naive implementation).

    Each Gibbs sweep consists of: num_inner_steps_psi psi steps + 1 theta sweep
    (with num_inner_steps_theta slice steps per group). The kernel runs
    num_gibbs_sweeps such sweeps.

    Parameters
    ----------
    logprior_psi_fn
        Function computing log p_0(psi) - the hyperparameter prior.
    logprior_theta_given_psi_fn
        Function computing log p(theta_j | psi) - the conditional prior for
        local parameters given hyperparameters. Signature: (theta_j, psi) -> scalar.
    loglikelihood_per_group_fn
        Function computing log f(Y_j | theta_j, psi) - the likelihood for each group.
        Signature: (theta_j, psi, data_j) -> scalar.
    data
        List or array of data for each group [Y_1, ..., Y_J]. Will be stacked
        into an array for efficient vectorized access.
    num_groups
        Number of groups J in the hierarchical model.
    num_gibbs_sweeps
        Number of full Gibbs sweeps per NS iteration.
    num_inner_steps_theta
        Number of slice sampling steps for each theta_j per sweep.
    num_inner_steps_psi
        Number of slice sampling steps for psi per sweep. Should typically
        match the dimensionality of psi for efficient exploration.
    num_delete
        Number of particles to delete and replace at each NS step.
    likelihood_depends_on_psi
        If True, the likelihood depends on psi. This means:
        - psi updates require O(J) likelihood evaluations per candidate
        - theta_j updates still benefit from O(1) constraint checking
        If False (default), psi updates are "free" (no likelihood evaluation).
    stepper_fn
        The stepper function for slice sampling (currently unused in cached version).
    update_inner_kernel_params_fn
        Function to update kernel parameters (covariances).
    delete_fn
        Function to select particles for deletion.
    max_steps
        Maximum steps for slice interval expansion.
    max_shrinkage
        Maximum shrinking steps for slice sampling.

    Returns
    -------
    SamplingAlgorithm
        A SamplingAlgorithm tuple with init and step functions.

    Example
    -------
    >>> # Hierarchical normal model (likelihood independent of psi)
    >>> J = 50   # Number of groups
    >>> d_theta = 1
    >>> data = [Y[j] for j in range(J)]
    >>>
    >>> # Create likelihood function (takes data_j directly, not index j)
    >>> def ll_fn(theta_j, psi, data_j):
    ...     return jax.scipy.stats.norm.logpdf(data_j, theta_j, 1.0).sum()
    >>>
    >>> algorithm = swg.as_top_level_api(
    ...     logprior_psi_fn=lambda psi: (
    ...         jax.scipy.stats.norm.logpdf(psi[0], 0, 10) +  # mu
    ...         jax.scipy.stats.norm.logpdf(psi[1], 0, 2)     # log_sigma
    ...     ),
    ...     logprior_theta_given_psi_fn=lambda theta_j, psi: (
    ...         jax.scipy.stats.norm.logpdf(theta_j, psi[0], jnp.exp(psi[1])).sum()
    ...     ),
    ...     loglikelihood_per_group_fn=ll_fn,
    ...     data=data,
    ...     num_groups=J,
    ...     likelihood_depends_on_psi=False,  # Observation noise is fixed
    ... )
    >>>
    >>> # For models where likelihood depends on psi (e.g., inferred noise):
    >>> def ll_fn_with_psi(theta_j, psi, data_j):
    ...     sigma_obs = jnp.exp(psi[2])  # psi includes observation noise
    ...     return jax.scipy.stats.norm.logpdf(data_j, theta_j, sigma_obs).sum()
    >>>
    >>> algorithm_with_psi = swg.as_top_level_api(
    ...     ...,
    ...     loglikelihood_per_group_fn=ll_fn_with_psi,
    ...     likelihood_depends_on_psi=True,  # Must check constraint during psi updates
    ... )
    """

    # Stack data into array for efficient vectorized access
    # This enables vmap/scan instead of Python loops
    if isinstance(data, list):
        data_array = jnp.stack(data)
    else:
        data_array = jnp.asarray(data)

    def compute_total_logprior(position: Dict[str, ArrayTree]) -> Array:
        """Compute log p_0(psi) + Σ_j log p(theta_j | psi)."""
        psi = position["psi"]
        theta = position["theta"]
        logprior_psi = logprior_psi_fn(psi)
        logprior_theta = jax.vmap(lambda th: logprior_theta_given_psi_fn(th, psi))(theta).sum()
        return logprior_psi + logprior_theta

    # Create init_state_fn using SwG-specific initialization with caching
    def swg_init_state_fn(position, loglikelihood_birth=jnp.nan):
        return init_swg_state_strategy(
            position=position,
            logprior_fn=compute_total_logprior,
            loglikelihood_per_group_fn=loglikelihood_per_group_fn,
            data_array=data_array,
            loglikelihood_birth=loglikelihood_birth,
        )

    kernel = build_kernel(
        logprior_psi_fn,
        logprior_theta_given_psi_fn,
        loglikelihood_per_group_fn,
        data_array,
        num_groups,
        num_gibbs_sweeps,
        num_inner_steps_theta,
        num_inner_steps_psi=num_inner_steps_psi,
        num_delete=num_delete,
        likelihood_depends_on_psi=likelihood_depends_on_psi,
        stepper_fn=stepper_fn,
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
        delete_fn=delete_fn,
        max_steps=max_steps,
        max_shrinkage=max_shrinkage,
    )

    def init_fn(position, rng_key=None):
        return init(
            position,
            init_state_fn=jax.vmap(swg_init_state_fn),
            update_inner_kernel_params_fn=update_inner_kernel_params_fn,
        )

    def step_fn(rng_key, state):
        return kernel(rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)
