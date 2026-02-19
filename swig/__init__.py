"""SwiG: Nested Sampling with Slice-within-Gibbs for hierarchical Bayesian models.

This package implements NS-SwiG, an algorithm that reduces the per-replacement
computational cost of nested sampling for hierarchical models from O(J²) to O(J),
where J is the number of groups in the hierarchy.
"""

from swig.swg import (
    as_top_level_api,
    build_kernel,
    init_swg_state_strategy,
    update_inner_kernel_params,
    SwGKernelParams,
    SwGInfo,
    SwGStateWithLogLikelihood,
)

__all__ = [
    "as_top_level_api",
    "build_kernel",
    "init_swg_state_strategy",
    "update_inner_kernel_params",
    "SwGKernelParams",
    "SwGInfo",
    "SwGStateWithLogLikelihood",
]

__version__ = "0.1.0"
