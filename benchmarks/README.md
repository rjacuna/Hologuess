# Benchmark results

These reports record historical measurements and their exact verification protocols. The solver installation and tests are described in the [project README](../README.md).

## Jones and Hodge reconstruction

- [Original cold-process benchmark](results/2026-11-26/report.md)
- [Grouped validation rerun](results/2026-11-26-grouped/report.md)
- [Measured wrapper and parameter overhead](results/2026-11-26-overhead/report.md)
- [Larger Jones and Hodge cases](results/2026-11-26-scaling/report.md)

These runs measured the full wall time of a new Sage process and the reconstruction and exact validation phases separately. Each method ran three times with one solver worker, one native arithmetic thread and empty solver caches. Each comparison used the same fitting observations and exact, independently checked held-out windows.

## Upstream compatibility and corpus verification

The compatibility tests port mathematical examples from Ore algebra, SymPy, gfun, RISC Guess and FriCAS. The source inventories in the [worked examples](../examples/worked_examples.ipynb) identify the original files and supported mathematical cases. Run the compatibility tests in the Sage environment used for Hologuess.

```sh
sage -python -m unittest discover -s tests -p 'test_upstream*.py' -v
```

The differential corpus comes from the [D-algebraic-Guessing benchmark directory](https://github.com/T3gu1a/D-algebraic-Guessing/tree/19bc0b7e862d4af4b4a3ed713cf0511875fe62f0/benchmarks) at commit `19bc0b7e862d4af4b4a3ed713cf0511875fe62f0`. The [input manifest](results/2026-11-26-upstream/inputs-manifest.json) gives exact file hashes, effective order and degree bounds, and the number of terms in each row. The reports describe the reference kernels, fitting and held-out equations, and timing policies.

- [Compatibility coverage and corpus verification](results/2026-11-26-upstream/report.md)
- [Dense-case profile](results/2026-11-26-dense-profile/report.md)
- [Input-derived modular actions and reduced validation](results/2026-11-26-input-actions/report.md)

## Public API races

The paired measurements used fresh workers on both sides and a common start barrier. After the stated grace interval, the runner stopped an unfinished competitor once the other had a verified answer. The $\infty$ notation records such a stop. Each report states what is included in its timing interval.

- [Jones sequences against native Ore](results/2026-11-26-jones-ore/report.md)
- [Bulk FLINT kernel probe](results/2026-11-26-flint-probe/report.md)
- [FLINT races on Jones and differential cases](results/2026-11-26-flint-races/report.md)
- [Three additional FLINT differential cases](results/2026-11-26-flint-dense-next3/report.md)
- [Constant-coefficient races against Ore, SymPy and Wolfram](results/2026-11-26-constant-recurrence-races/report.md)
