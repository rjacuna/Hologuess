# qhsolve refinement audit — 2026-11-26

This audit compares the current Hologuess implementation with the actual
JonesRec `aj/qhsolve.py`, the copied interpolation/PM-Basis modules, and the
solver-history requirements in JonesRec's NOTES, MISTAKES and changelog as
recorded in the full reading review. It is independent of the validator
implementation owner. Original repositories were read only. The only runtime
change made by this reviewer was the generic parameter support restriction
described below, with three small independent regression tests; no benchmark
or knot computation was launched here.

## Core comparison

All 42 original top-level functions/classes remain present. Before the current
validator correction, 35 were AST-identical to JonesRec. Seven differed only for
package-relative imports, generic action tensors, tensor point slicing, ordered
cache identities, and the native-thread worker initializer. All top-level
configuration assignments were identical. PM-Basis is byte-identical to the
original; fast interpolation changes only package-relative imports.

After the correction, the additional intended changes are exact ZZ/QQ domain
selection and starting-index support in the grouped Laurent validator, plus a
safe fallback from the heuristic envelope when sequence polynomials include
zero. These are explicitly scoped extensions of the original strategy.

The following arithmetic and search refinements remain in the original code
path, rather than being replaced by another algorithm:

| Refinement | Finding |
|---|---|
| Detection at two points; ascending order and binary degree search | Preserved in `corner_search`/`detect` |
| Dimension-one acceptance and separate gaps/multiple kernels | Preserved |
| Grow depth at two points before paying for the full point set | Preserved through the Sequence/grow contract |
| Sparse action support before extraction | Preserved for q-sequences; generic route fixed below |
| Degree-envelope planning, guard margins, reactive widening | Preserved; rational wrapper access fixed below |
| Blocked RREF and batched live pivot-profile handling | AST-identical |
| Exact limb-based modular GEMM | AST-identical |
| Measured reference selection requiring every witness | AST-identical |
| Common rational denominator reconstruction and held-out checks | Preserved |
| Keep a verified cleared reference while widening numerators | Preserved |
| Proportional oversampling with the minimum spare-point count | Constants and extraction logic preserved |
| CRT and Wang reconstruction with denominator coprimality | AST-identical |
| Stable successive-prime reconstruction; extra primes for height failures | Preserved |
| No LCM of partly failed rational reconstructions | Preserved |
| Raw per-point kernel persistence and incremental completed-task writes | Preserved; ordered cache keys strengthen input binding |
| Default measured RREF interpolation; optional tree interpolation/NTT | Preserved |
| Shared RAM fraction, worker baseline and task sizing | Preserved |

The old Jones application also supplies application-specific point selection,
stored sequences, deepest cached term depth, and theorem-backed certificates.
Those are not generic qhsolve primitives. Hologuess must not claim that a generic
finite prefix reproduces those application-specific all-index proofs.

## Refinements that the wrappers had bypassed

### Grouped exact validation

The public wrapper's Python Fraction loop bypassed `annihilates_exactly`, despite
that optimized checker remaining in the copied core. This was a real extraction
regression. The correction delegates to the grouped Sage Laurent checker: build
the coefficient polynomial for each shift first, then perform one polynomial
product per shift in each window. ZZ is used for integral inputs; QQ is used
when a coefficient has a nontrivial denominator. The wrapper's public starting
index is included in the q-power shift. There is no need to replace grouped
arithmetic when admitting rational Laurent coefficients.

### Rational sequence planning and evaluation

The initial rational wrapper returned `polys=None`, unnecessarily disabling the
degree-envelope planner. The planner reads exponent support, so rational scalar
coefficients do not invalidate it. The current wrapper exposes all exact Laurent
supports, retains integers as integers, and reduces rational coefficients modulo
the active prime before calling the existing vectorized `eval_laurent`. This
avoids per-monomial power loops and repeated coefficient inversion. Only the active
prime's reduced representation is retained, avoiding a large sequence copy for
every CRT prime. The integral-input normalization also keeps Python integer
coefficients directly instead of constructing a temporary Fraction for each term.

### Generic parameter support restriction

The initial `interpolate_relation` went directly from a line-rank probe to
`extract_one` on all actions. That skipped qhsolve's useful support-restriction
stage when the relation had identically zero coefficients. The correction unions
nonzero coefficient positions across the rank-one probes, ignores exceptional
higher-nullity probes, and extracts only those original action indices. It keeps
the full observation count and the original labels; it does not assume arbitrary
module observations have the q-sequence window-growth semantics. Each probe uses
one RREF; pivots provide rank/nullity, and only the unique free-column vector of a
line is inspected. Higher-dimensional probe bases are not materialized.

Metadata records original/retained action counts and original support indices.
The supplied exact validator still receives coefficients in the complete original
coordinate system. An unlucky modular support guess cannot be silently accepted:
reconstruction or exact validation must reject it. Three independent tests pass:
noncontiguous retained coordinates `(0,2)` with a known exact relation; an unlucky
four-point zero pattern that fails closed; and an exact-validator mismatch.

### Cached or prefetched spare points

The original Sequence contract allows `at(p, count)` to return `K >= count`
parameter points. The initial QSequence wrapper required equality, blocking
oracles that return a larger already available compatible batch. The current
wrapper accepts the original contract, checks the value matrix against the actual
returned K, and checks distinctness across every returned point. Exact-value
inputs still produce the requested count. This preserves reuse/prefetch options
without introducing a point-generation policy into the reconstruction core.

## Limits and other overhead

The generic finite-module route returns a full bounded kernel and is not a scalar
principal-ideal search. Its exact parameter-column expansion is an intentional
separate route, not evidence that the one-parameter interpolation strategy was
removed. Similarly, a caller-specified arbitrary observation tensor cannot use
the q-sequence tropical envelope without additional action-specific degree data.

The generic parameter route can support a line relation only; higher-dimensional
spaces require the finite-module route. Native q-sequence fixed-support APIs may
skip adaptive search when the caller explicitly requests incompatible bounds or
filters; those requests and bounds are documented.

One remaining convenience-wrapper sampling limitation is worth distinguishing
from a dropped core primitive: default QSequence points are `2,3,...`. Original
Jones applications choose admissible/large-order points. An exact sequence with a
common `(q-2)` factor has an exceptional first specialization and may produce a
point-disagreement gap rather than recover a corner. That is inconclusive, not a
false emptiness proof; callers can supply an admissible modular point oracle.

The worker native-thread limiter is an added resource safeguard, not a missing
refinement. Its repeated cold initialization can contribute overhead. The previous
measurements did not isolate that cause, so a causal speed claim needs separate
profiling. The present cold benchmark rerun, including exact validation, remains
the release owner's check of the actual end-to-end outcome.

No other omitted arithmetic refinement was identified in the copied qhsolve
pipeline. This statement distinguishes the corrected wrapper bypasses above from
intentional differences required by arbitrary module relations and fixed caller
supports.
