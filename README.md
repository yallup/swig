# SwiG: Nested Sampling with Slice-within-Gibbs

SwiG implements NS-SwiG (Nested Sampling with Slice-within-Gibbs), an algorithm for efficient nested sampling of hierarchical Bayesian models.

SwiG reduces the per-replacement cost of nested sampling for hierarchical models from O(J²) to O(J), with O(1) likelihood evaluation per local block update via budget constraint decomposition. It is implemented in JAX with GPU acceleration, built on the BlackJAX nested sampling infrastructure.

## Installation

```bash
uv sync
```

To run the examples (adds `matplotlib` and `distrax`):
```bash
uv sync --extra examples
```

## Examples

| Script | Description |
|--------|-------------|
| `uv run python examples/funnel.py` | 10D funnel with analytic logZ validation |
| `uv run python examples/glm.py` | Hierarchical Gaussian with analytic evidence |
| `uv run python examples/radon.py` | Radon contextual effects (Minnesota, 85 counties) |
| `uv run python examples/sv.py` | Stochastic volatility (S&P 500, Markov variant) |

## Citation

If you use SwiG in your research, please cite:

```bibtex
@article{Yallup2026swig,
  author  = {Yallup, David},
  title   = {Nested Sampling with Slice-within-Gibbs: Efficient Evidence
             Calculation for Hierarchical Bayesian Models},
  journal = {arXiv preprint},
  year    = {2026},
}
```

SwiG builds on the following works:

```bibtex
@misc{yallup2026nss,
  title   = {Nested Slice Sampling: Vectorized Nested Sampling for
             GPU-Accelerated Inference},
  author  = {David Yallup and Namu Kroupa and Will Handley},
  year    = {2026},
  eprint  = {2601.23252},
  archivePrefix = {arXiv},
  primaryClass  = {stat.CO},
  url     = {https://arxiv.org/abs/2601.23252},
}

@misc{cabezas2024blackjax,
  title   = {BlackJAX: Composable {B}ayesian inference in {JAX}},
  author  = {Alberto Cabezas and Adrien Corenflos and Junpeng Lao
             and R\'{e}mi Louf},
  year    = {2024},
  eprint  = {2402.10797},
  archivePrefix = {arXiv},
  primaryClass  = {cs.MS},
}
```

## License

[Apache 2.0](LICENSE)
