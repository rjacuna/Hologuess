# Changelog

Versions use Semantic Versioning. Tested capability milestones are tagged locally.

## Unreleased

- Use $S$ and $D$ for shifts and derivatives in Sage LaTeX display. Add
  `set_latex_symbols` for notebook preferences and `operator_latex` for
  copyable LaTeX. Keep the native Ore text and operator types.
- Add a short [SageMath quickstart](examples/quickstart.ipynb) and organize the
  [worked examples](examples/worked_examples.ipynb) by mathematical problem.
  Both notebooks run offline after installation and show native Sage results.
  Source references stay with each example, and the benchmark reports carry
  the timing and verification details.

## 0.3.0 — 2026-11-26

- Prefer FLINT for modular elimination when the bridge is installed or
  configured. Keep `kernel_backend='numpy'` available and fall back to it
  when no bridge is installed. The bridge builder now installs to an
  automatically discovered location when `--output` is omitted.
- Rerun the next three Dense losses with the optional FLINT bridge and a
  shared 0.5-second cutoff. FLINT wins two cases. All nine pairs agree
  exactly. [Results](benchmarks/results/2026-11-26-flint-dense-next3/report.md)
- Add a SageMath notebook with pinned example provenance and exact checks.
  Benchmark two constant-coefficient recurrence examples against native Ore,
  SymPy, and Wolfram at a shared 0.5-second cutoff. Show stopped races as
  $\infty$ in reports. [Notebook](examples/worked_examples.ipynb) ·
  [Constant-coefficient races](benchmarks/results/2026-11-26-constant-recurrence-races/report.md)

## 0.2.0 — 2026-11-26

- Add `kernel_backend='flint'` as an opt-in alternative to the default
  NumPy elimination in the q-sequence, modular module and parameter
  interpolation routes. Build the bridge with
  `python -m hologuess.build_flint` against an installed FLINT library.
- Preserve the prime pool, pivot normalization, live-point selection,
  cache coordinates and exact validation. Report the selected backend
  in the result metadata.
- Pass all 401 tests with a built FLINT bridge. The suite includes 11
  backend tests covering independent exact arithmetic, public solver
  parity, cache replay and forked point workers. The standalone wheel
  includes the bridge source and passed a build and reconstruction smoke check.
- Rerun the four Jones cases with one 60-second class grace and the first
  previous differential loss with a 0.5-second class grace. The selected
  differential case now takes 105.1 ms with FLINT versus 114.2 ms for native
  Ore. All 20 returned outputs match their exact references.
  [Cold race report](benchmarks/results/2026-11-26-flint-races/report.md)

### Earlier Jones and FLINT measurements

- Test a bulk FLINT bridge for the existing Jones kernel and interpolation
  matrices at the same primes. The benchmark prototype reduces solve and exact
  validation time by 8–29% on the four frozen inputs and recovers the same
  operators. [FLINT probe](benchmarks/results/2026-11-26-flint-probe/report.md)
- Race the two public guessers on four frozen Jones inputs with one 60-second
  cutoff for the class. Verify complete normalized coefficient arrays and exact
  fitting and suffix windows. The $3_1$ medians are essentially tied, Hologuess
  is 4.10 times as fast on $4_1$, and native Ore records $\infty$ on $5_2$ and $6_1$.
  [Jones race report](benchmarks/results/2026-11-26-jones-ore/report.md)

## 0.1.4 — 2026-11-26

- Check each stable nonempty fitting basis once with Sage's exact field matrix
  product. Reuse that certificate during result construction, preserving exact
  held-out checks, full held-out intersections and failed-candidate reconstruction.
- Construct standard parameter-free QQ/ZZ modular actions from original inputs
  reduced once per prime, reusing recursive shift, derivative, difference and
  algebraic series-power images. Keep exact action columns for the characteristic-
  zero certificate and cache identity; parameter and other exact-field routes
  retain their existing reconstruction behavior.
- Fall back to exact-matrix reduction for raw modular poles so cancellation and
  unused observations do not change accepted primes. Disable captured construction
  after a change of exact columns or the fitting-row count. Preserve source and
  complete fitting-matrix cache identity and rank, pivot-profile and residual replay.
- Bound cached modular action states relative to input and observed cells before
  allocation, with exact-matrix fallback for oversized high-order constructions.
- Leave the qhsolve arithmetic core, stable reconstruction, normalization and
  persistence strategy unchanged. Record fresh validation and per-problem pilot
  races with the same independent complete-space operator verification in
  benchmarks/results/2026-11-26-input-actions/report.md.

All 390 tests pass and all 419 corpus rows match their complete independent
reference spaces. The 114 reported pairs also match exactly, with no censored
runs and per-problem grace of 0.5 or 1 second. Hologuess wins 13 of 38 problem
medians, up from 7. Its order-15, degree-15 construction median falls from
2.146 s to 0.344 s, compared with 0.131 s for native Ore.

## Unreleased — dense-case diagnosis

- Profile the generic QQ differential route against native Ore without changing
  solver behavior or the recorded benchmark cohort. Two exact rational fitting
  checks account for 67.6% of the order-15/degree-15 profile; per-entry modular
  conversion accounts for most of the remaining gap. RREF is only about 3%.
- Retain eight exact-verified diagnostic calls, phase/cProfile evidence and
  prepared-input experiments comparing compiled exact products and input-first
  modular actions. These demonstrate stage costs, not an installed solver speedup.
  See benchmarks/results/2026-11-26-dense-profile/report.md.

## 0.1.3 — 2026-11-26

- Accept exact rational observations with polynomial coefficient parents by
  localizing input arithmetic while preserving the requested output algebra.
  Clear scalar basis denominators when needed by integral operator parents.
- Skip persistent-cache identity serialization when no store is supplied;
  preserve the full keys and rank/residual checks for selected persistent stores.
- Port public Ore/SymPy, gfun, RISC and FriCAS examples, retaining source-level
  coverage inventories and explicit native API/search-policy distinctions.
  Nonlinear equations are tested through declared exact module feature families.
- Independently verify all 419 DalgGuessing rows against complete bounded operator
  spaces, including exact fitting and held-out residuals. Use exact factorial
  quotients and per-basis coefficient reach so the final unseen term is checked.
- Add paired public-guesser races with a common setup barrier, separately
  calibrated per-problem grace intervals, exact full-space candidate admission,
  censoring records and fail-closed owned-process cleanup. Construction timings
  remain separate from startup-inclusive cold measurements and pilot results.

Validation: 333 integration tests pass with no skips; a separate final controller
run passes 28 race-protocol tests. These are separate recorded runs, not a claimed
single 361-test execution. All 419/419 corpus rows agree with the independently
certified full spaces. Native Maple, Mathematica/RISC and FriCAS runtimes were not
tested. All 114 reported paired races agree exactly without censoring, using
per-problem N from 0.5 to 5 seconds. Hologuess wins seven problem medians and
native Ore wins 31, including the larger dense differential cases. Construction
times and complete evidence are in
benchmarks/results/2026-11-26-upstream/report.md. A standalone 0.1.3 wheel passes
exact domain/parent and held-out checks from outside the checkout.

## Unreleased — benchmark evidence only

- Add a 36-run serialized cold scaling ladder, with three repetitions and exact
  full-operator/held-out agreement: Jones 6_1 (5_2 anchor), and complete
  intersections of two, three and four quadrics. No runtime change from 0.1.2.
- Prepare larger 6_1 input through a separate finite-validated Habiro C recurrence,
  exact Habiro summation and agreement with all 50 independent stored values;
  retain provenance and distinguish recurrence-derived data from skein evaluations.
- Record matrix/support sizes, final point horizons, input/operator coefficient
  heights and RSS. Independently check generating-series inputs, corrupted unseen
  residual rejection, stored Jones operators and full cold medians.

Validation: 110 tests pass. Public 6_1 cold median is 9.879s versus qhsolve 9.602s;
largest Hodge cold is 1.792s versus holo 2.637s (algorithm/checks 0.186s vs 1.035s).
All 36 runs pass; these finite workloads do not establish asymptotic or global
annihilation claims.

## 0.1.2 — 2026-11-26

- Profile remaining overhead before changing the execution path. The small Jones
  case created 62 worker pools and repeated native-library discovery about 62
  times. Single-worker batches now execute the identical kernel task in-process,
  preserving chunk sizes, live masks, counters and per-chunk cache persistence.
- Scope native limits to an operation, initialize them lazily at actual BLAS use
  or pool creation, and restore caller settings on success and failure. Independent
  Python-thread limit scopes serialize; multiworker children inherit the prepared
  controller. Small elementwise-only RREF batches avoid unused library scans.
- Batch parameter-polynomial evaluation, avoid inversions of unit denominators,
  share distinct denominators, and retain native polynomial input/validation.
  Heterogeneous rational rows keep a rational path without denominator-LCM growth;
  original poles and all exact training/held-out checks remain enforced.
- Reuse the exact original 60-prime pool without cold-import trial division or
  repeated verification of trusted constants. Independently test its full values,
  order and primality; unfamiliar caller primes are still checked.
- Add serial/pool parity, partial persistence, native-thread restoration and
  concurrency, lazy BLAS control, prime-pool and general rational-domain tests.

Validation: 107 tests pass; all 45 cold runs match the previous operators exactly
and retain all fitting/held-out checks. Public cold medians are 2.142s, 3.938s,
2.661s for 3_1, 4_1, 5_2, versus qhsolve 2.584s, 4.653s, 2.621s. Hodge full-cold
totals are near parity with holo; tiny-case algorithm times remain 6–13ms higher.
The original solvers and prior benchmark artifacts remain unchanged.

## 0.1.1 — 2026-11-26

- Reuse the copied qhsolve grouped exact validator from the public q-sequence
  wrapper. Extend the shared implementation to Laurent arithmetic over ZZ or QQ
  and nonzero starting indices. Keep one Sage polynomial product per shift and
  window; remove the per-monomial Fraction residual loop.
- Keep native integer coefficients as integers. Retain rational Laurent exponent
  data for degree-envelope planning and reuse vectorized modular evaluation after
  exact coefficient reduction, with a bounded active-prime coefficient cache.
- Restore zero-coefficient support pruning to generic line interpolation, retaining
  original action indices, labels and exact full-observation validation.
- Accept K >= requested cached/prefetched modular point batches, matching the
  original Sequence contract. A zero input polynomial safely disables the degree
  estimate rather than raising from an undefined degree.
- Audit the full qhsolve lineage and add independent scalar-reference, ring/domain,
  support and oracle-contract regressions. Cold benchmark rerun and full validation
  results accompany the release record; earlier measurements remain unchanged.

Validation: 85 tests pass. All 45 cold rerun records agree exactly, including the
42 shared with the original measurements. Public 4_1 falls from 13.071s to 6.065s
(2.15x); public 5_2 is now measured at 2.858s. Original qhsolve remains faster on
some small cold cases, and Hodge interpolation overhead still dominates the small
hypersurface examples. A standalone wheel verifies integer and rational grouped
validation without sibling imports. No original solver or data store changed.

## Unreleased — benchmark documentation

- Add a serialized cold-process benchmark harness for stored Jones values and
  hypersurface Hodge polynomials, with matched ansatzes and exact independent
  operator/held-out comparisons. Three repetitions per method: 42 runs total.
- Record results and raw operators under benchmarks/results/2026-11-26. All
  reconstructed operators agree. Current public q-sequence validation regresses
  from the original grouped Sage checker, and parameter interpolation overhead
  exceeds exact-extraction cost on the small Hodge examples. No solver change or
  performance improvement is claimed by this benchmark milestone.

## 0.1.0 — 2026-11-26

- Extract the current JonesRec qhsolve, fast interpolation and PM-Basis arithmetic
  with source hashes, relevant Git/version history and a review of all 248 Markdown
  files (179 distinct contents). Original projects remain unchanged.
- Preserve q-sequence corner detection, support restriction, degree planning,
  blocked/batched modular arithmetic, measured reference selection, rational
  denominator clearing, adaptive bands, multi-prime stable reconstruction and
  persistent per-point kernels. Include rational exact input coefficients and
  public nonzero-index conversion.
- Extend the reconstruction engine to arbitrary one-parameter action-observation
  tensors. Provide exact rational-function columns with pole skipping and full
  characteristic-zero residual validation.
- Add finite observable module kernels with full relation bases, exceptional-prime
  profile handling, exact residuals, held-out intersections and checked cache
  replay. Do not transfer principal-ideal assumptions to arbitrary modules.
- Add independently implemented standard Ore action adapters for shifts,
  derivatives, forward differences, q-shifts, algebraic series and mixed rectangular
  multivariate inputs. Preserve caller algebra/action names. The adapter never
  invokes ore_algebra.guess or the factor-search monkey patch.
- Add coordinate actions B*sigma(v)+delta(v), dynamic worker native-thread limits,
  explicit exactness/bounds checks, standalone packaging and examples.

Validation: 62 Sage tests pass; independent Jacobian-ring Hodge data recover the
cubic-hypersurface recurrence and agree on held-out dimensions; q-square and
48 stored trefoil terms reproduce the original qhsolve operators exactly with
fresh stores and matched exact finite-window validation. A wheel is checked from
outside the checkout without importing sibling repositories. Timings are recorded
as observations, not a universal speedup claim.

Scope: every search uses finite observations and explicit ansatz bounds. Results
do not prove all-index annihilation, global minimality or complete annihilator
ideals. Generic multivariate fitting uses expanded constant-field action columns;
the full parameter-interpolation strategy applies to a line relation over one
reconstruction parameter. It is not a replacement for every optional legacy
ore_algebra.guess keyword. User-defined modules require computable actions and
observations, with their algebraic compatibility supplied by the caller.
