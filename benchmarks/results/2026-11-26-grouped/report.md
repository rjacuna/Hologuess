# Hologuess 0.1.1 — grouped validation and cold rerun

All **45 cold runs** passed exact fitting and held-out checks and recovered identical normalized operators across methods. The **42 cases shared with the earlier run** also match the earlier full coefficient lists and input hashes exactly. Three runs add public-API coverage for5_2, which had no earlier public timing.

Three repetitions per method, sequential fresh Sage Python processes, empty solver stores, one worker and one native thread. These are process-cold measurements; OS filesystem caches were not flushed. Inputs, bounds and validators remain matched as described in [the protocol](../../README.md).

## Public q-sequence timings

Full cold-process medians, seconds, including startup/imports, stored-data loading, reconstruction and exact validation:

| Knot | Hologuess 0.1.0 public | Hologuess 0.1.1 public | qhsolve rerun | Hologuess 0.1.1 core |
|---|---:|---:|---:|---:|
| 3_1 | 3.841 | 3.594 | 2.661 | 3.560 |
| 4_1 | 13.071 | 6.065 | 4.721 | 5.978 |
| 5_2 | not timed | 2.858 | 2.668 | 2.737 |

For4_1 the public cold wall time falls from13.071s to6.065s, about2.15times faster (54% less time). It is now close to the Hologuess core time5.978s. Original qhsolve remains faster at4.721s; the public-wrapper validation regression is removed, while the separate package/core cold overhead remains. The source audit found no omitted scalar reconstruction arithmetic explaining that overhead. Worker/resource initialization remains an unisolated candidate, not a proven causal attribution.

## Algorithm plus validation

Excluding interpreter startup, imports and stored-data loading, median seconds:

| Knot | qhsolve | Hologuess core | Hologuess public |
|---|---:|---:|---:|
| 3_1 | 1.020 | 1.958 | 1.991 |
| 4_1 | 3.027 | 4.287 | 4.373 |
| 5_2 | 0.858 | 0.915 | 1.054 |

## Hodge rerun

The same hypersurface cases use independent Jacobian-ring data, fit dimensions0..46, check unseen dimensions47..92, and match the independently written recurrence formula. The prescribed action support, n-degree sweep and x-degree cap are unchanged. Generic support pruning is restored in0.1.1, so this rerun measures that change as well as the release packaging; grouped q-sequence validation is not used by the Hodge action oracle.

| Hypersurface degree | holo cold process | Hologuess cold process | holo algorithm + validation | Hologuess algorithm + validation |
|---|---:|---:|---:|---:|
| 3 | 1.632 | 1.840 | 0.067 | 0.227 |
| 4 | 1.632 | 1.854 | 0.073 | 0.233 |
| 5 | 1.649 | 1.852 | 0.077 | 0.237 |

Hologuess remains slower on these small order-two Hodge kernels: detection is faster, but parameter interpolation overhead dominates cheap exact rational extraction. These measurements establish neither a speedup nor a scaling conclusion for larger Hodge problems.

## Refinement audit and validation

- Shared grouped Laurent validation overZZ/QQ, extended to actual index origins; no per-monomial Fraction residual implementation in the wrapper.
- Rational inputs retain degree-envelope planning and vectorized modular evaluation after exact coefficient reduction. Integer inputs retain native integers; rational coefficient-image storage is bounded to the active prime.
- Generic parameter interpolation again prunes inactive support using rank-only probes and preserves original action coordinates and exact validators.
- Modular sequence oracles again accept compatible prefetched batches withK>=requested points.
- A zero input polynomial now causes safe degree-estimate fallback.
- All42 original core definitions remain.32 are AST-identical;10 reviewed extensions cover the grouped domain/index generalization, tensor/import/cache/resource adaptations and safe degree fallback. PM-Basis is byte-identical. No other omitted arithmetic refinement was identified.

See [the audit](../../../docs/refinements-audit-20261126.md). The test suite passes85 independent regressions. Exact operator and finite-data checks do not constitute all-index or global-minimality proofs.

[Timing summary](summary.json) · [Before and after](before-after.json) · [Original results](../2026-11-26/report.md)
