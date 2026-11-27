# Upstream guesser integration and calibrated public-API races

2026-11-26; Hologuess 0.1.3; SageMath 10.10; installed ore_algebra 0.5.

**All 419 published DalgGuessing rows agree on their complete bounded exact
operator spaces. All 114 reported paired races completed, with no censoring or
incorrect answer.** A separate pilot for each of 38 comparable problems selected
its common grace interval N before three fresh reported pairs. N ranged from
0.5 to 5 seconds. Hologuess has the lower median construction time on 7 problems;
native Ore has it on 31.

The small `Dense_r5d6` case takes 63.455 ms versus 113.775 ms (Hologuess 1.79x
faster). The central `Dense_r15d15` case takes 2.146 s versus 0.128 s (native Ore
16.7x faster). These generic differential examples do not support a blanket
performance advantage for Hologuess. They use its explicit finite-module backend;
the earlier Jones/qhsolve and Hodge/holo comparisons are separate experiments.

## Test coverage and native execution

| Source suite | Source calls/assertions accounted for | Hologuess test methods |
|---|---:|---:|
| Ore algebra | 12 | shared Ore/SymPy total: 25 |
| SymPy | 19 | included above |
| gfun | 185 | 93 |
| RISC Guess / QREGuess | 66 | 67 |
| FriCAS | 73 | 31 |
| Total | 355 | 216 |

The full integration run passes **333 tests, zero failures/errors/skips**: these
216 upstream-specific methods plus 117 core/helper regressions. A separate final
run passes **28 race-protocol tests**, including actual owned process-tree cleanup
and mocked failed-inspection guards. These are separately recorded executions.
The inventories map every source call, including repeated presentations and
explicit unsupported API, default-search or internal-policy differences. They
are not 355 native successes or distinct datasets.

Mathematical ports exercise shifts, derivatives, differences, q-actions, mixed
multivariate actions, exact parameter/algebraic domains and declared nonlinear
module feature families. Explicit features do not claim automatic discovery of
every native nonlinear ansatz. Large RISC grids retain all candidate columns but
use separating observations and independent rank certificates; source-grid
subsampling is recorded. The literal Ore 1000-value/order-6/degree-90 case and
both 2000-coefficient GF(1091) examples retain their full published sizes.

Native SymPy's six test functions all pass. The native Ore guessing suite actually
runs 28 doctests, with four failures from the installed multivariate integer-ring
Singular constructor; its other 24 pass. Hologuess's corresponding mathematical
ports pass through its local coefficient-parent handling. Maple and FriCAS are
unavailable, and the restricted RISC implementation was not executed. No native
Maple/RISC/FriCAS pass or timing claim is made. The ordinary Sage doctest launcher was unavailable in this
installation, so its Python doctest controller executed the native Ore suite.

[Native test results](native-tests.json). Source URLs, commits and call-site mappings are in the four inventories under `tests/upstream_*_inventory.json`.

## Whole-corpus exact verification

The pinned DalgGuessing source is commit
`19bc0b7e862d4af4b4a3ed713cf0511875fe62f0` of
[T3gu1a/D-algebraic-Guessing](https://github.com/T3gu1a/D-algebraic-Guessing/tree/19bc0b7e862d4af4b4a3ed713cf0511875fe62f0).
All 40 input CSV files, 419 row positions and 409 distinct rows are included,
including the nine root Dense/Sparse files and the 29-row `Dense_r25d4` file.
The downloadable CSV bytes total 48,881,383; the input manifest records Git blob IDs and SHA-256 checksums.

`Dense_r19d3.csv` duplicates `Dense_r8d5.csv` byte for byte. Its original filename
is retained with explicit effective bounds (8,5); it is not an order-19 recovery.
The complete corpus has 413 kernel lines, five two-dimensional spaces and one
three-dimensional space: 426 basis vectors per engine, 852 serialized vectors.
There are no failed jobs or empty-ansatz substitutions.

The reference uses upstream finite-field `guess_raw` kernels followed by
independent Sage CRT/rational reconstruction, with a separately assembled Sage
finite-field kernel where the native data guard refuses the input. This is not
native Maple execution or native characteristic-zero Ore timing. Modular rank
lower bounds and independent exact lifted vectors certify completeness of each
rational fitting kernel; equality is checked on full canonical coefficient arrays.
A separate exact Taylor-action checker uses integer factorial quotients and
requires genuinely unseen-reaching suffix equations. Neither reconstruction
receives the held-out coefficients.

The final audit found seven sparse basis vectors whose positive coefficient
valuation required eight additional equations to reach every declared held-out
coefficient. All eight were independently verified exactly with Fraction/integer
arithmetic. Original reconstruction receipts are preserved together with a
[post-guard overlay](post-guard-check.json), rather than silently relabelled as a
full rerun. Blank-field and upstream unit-filter guards cannot affect this corpus,
as established by independent all-row checks, and now have focused regressions.

These are complete bounded finite-data spaces and exact suffix checks, not
all-index annihilation, global minimality or complete annihilator-ideal proofs.
Verification batches use a fresh process per CSV and process its rows serially;
their times are not cold per-row benchmark samples.

[Input manifest](inputs-manifest.json) · [Original data license](Dalg-LICENSE.txt)

## Hajij–Levitt protocol with per-problem N

The adaptation follows the JonesRec Hajij–Levitt comparison's common start
barrier and exact-reference admission rule. Each race has two fresh Sage processes,
one solver/native thread per method and empty solver stores. They import their
actual public guesser, load the same exact fitting prefix and construct the
requested algebra before announcing readiness. Only one paired race runs at a
time. Process setup and waiting, serialization and controller reference comparison
are outside the construction clock. Canonical full-space output conversion is
inside it. OS filesystem caches are not flushed; background applications are not
isolated. These are fresh-solver construction-race latencies, not isolated
startup-inclusive cold wall times.

Hologuess calls `guess(..., path=[(r,d)], backend='module')`. The competitor calls
actual public `ore_algebra.guess(..., path=[(r,d)], cut=None, ensure=0)`, not the
much slower direct-QQ diagnostic or the modular reference wrapper. Native Ore
returns a generator, so the timed adapter forms all its left monomial multiples
remaining inside the box. Equality with the independent complete reference
certifies completeness of this conversion on these inputs; no general theorem
about that simple closure construction is assumed.

For each problem a separate paired pilot starts with a 30-second grace; a
120-second pilot retry is allowed only if censored. Both pilot results must be
exactly verified. Choose

`N_ms = niceceil(max(250, 2 * 1000 * slower_verified_arrival_seconds))`,

rounding upward to a 1/2/5 decimal interval. The 250-ms floor therefore rounds to
500 ms. Each chosen N is saved before three new reported pairs and is identical
for both methods. All 38 pilots finished on their first attempt. The distribution
is 500 ms for 13 problems, 1000 ms for four, 2000 ms for six, and 5000 ms for 15.
No overrides were needed. Pilot timings are excluded from reported medians.
Earlier timing diagnostics existed, so this is not described as a selection
without prior timing knowledge.

At N, if neither answer is verified, keep waiting. If only one is verified, stop
the other; if both are verified, retain both finite times immediately. A valid
publication during termination still counts. The $\infty$ tag would mark a
competitor stopped without a verified answer. The first such stop would end the
remaining repetitions of that problem, while unrelated problems continue. The
setup/computation/termination resource caps are separate from N. Cleanup kills
only the newly owned groups/PIDs and requires a successful process-table check;
failed inspection cannot certify absence. All 152 pilot/reported pairs and their
304 result spaces pass; no owned workers remain.

Row zero of each distinct feasible file is raced, not all 419 rows. The duplicate
file is deduplicated. `OrderUpTo30/Dense_r30d0.csv` cannot enter an equal-data pair:
its native public minimum-data guard would consume the reserved held-out term.
That file's mathematical rows remain part of the complete corpus verification.
The earlier partial serial batch and aborted globally fixed-grace cohort are
superseded diagnostics and are excluded from every table below.

The [comparison data](comparison.json) give the reported timings.
All 32 source pins, including 15 installed Ore modules, and input hashes remain
unchanged across the cohort. Setup diagnostics include launcher overhead or
barrier waiting and must not be read as pure import speed or whole-process
elapsed time.

## Three-pair median construction times

Times are milliseconds; N is seconds. The final ratio is Ore/Hologuess: above 1
favours Hologuess, below 1 favours native Ore. Every cell uses all three finite
reported repetitions, with no pilot or censored-result selection. Full ranges,
public-API/output-adapter times and setup diagnostics are in the comparison JSON.

| Input file | r | d | N (s) | Hologuess (ms) | Native Ore (ms) | Ore/HG |
|---|---:|---:|---:|---:|---:|---:|
| `Dense_r5d6.csv` | 5 | 6 | 0.5 | 63.455 | 113.775 | 1.793 |
| `Dense_r19d3.csv` | 8 | 5 | 0.5 | 74.032 | 113.450 | 1.532 |
| `Dense_r4d10.csv` | 4 | 10 | 0.5 | 77.577 | 111.247 | 1.434 |
| `Sparse_r4d10.csv` | 4 | 10 | 0.5 | 71.459 | 110.963 | 1.553 |
| `OrderUpTo30/Dense_r29d1.csv` | 29 | 1 | 0.5 | 83.243 | 116.848 | 1.404 |
| `OrderUpTo30/Dense_r1d30.csv` | 1 | 30 | 0.5 | 87.167 | 111.803 | 1.283 |
| `Sparse_r20d2.csv` | 20 | 2 | 0.5 | 88.791 | 125.949 | 1.418 |
| `OrderUpTo30/Dense_r2d28.csv` | 2 | 28 | 0.5 | 165.402 | 111.502 | 0.674 |
| `OrderUpTo30/Dense_r28d2.csv` | 28 | 2 | 0.5 | 147.122 | 113.950 | 0.775 |
| `Sparse_r4d20.csv` | 4 | 20 | 0.5 | 160.935 | 116.263 | 0.722 |
| `Sparse_r20d4.csv` | 20 | 4 | 0.5 | 127.643 | 113.153 | 0.886 |
| `OrderUpTo30/Dense_r3d27.csv` | 3 | 27 | 0.5 | 247.327 | 117.470 | 0.475 |
| `OrderUpTo30/Dense_r27d3.csv` | 27 | 3 | 0.5 | 226.103 | 120.774 | 0.534 |
| `OrderUpTo30/Dense_r25d4.csv` | 25 | 4 | 1 | 284.014 | 115.527 | 0.407 |
| `OrderUpTo30/Dense_r4d26.csv` | 4 | 26 | 1 | 435.802 | 123.748 | 0.284 |
| `OrderUpTo30/Dense_r26d4.csv` | 26 | 4 | 1 | 337.682 | 117.903 | 0.349 |
| `OrderUpTo30/Dense_r5d25.csv` | 5 | 25 | 2 | 546.520 | 112.897 | 0.207 |
| `OrderUpTo30/Dense_r25d5.csv` | 25 | 5 | 1 | 480.567 | 118.393 | 0.246 |
| `OrderUpTo30/Dense_r6d24.csv` | 6 | 24 | 2 | 858.126 | 120.412 | 0.140 |
| `OrderUpTo30/Dense_r24d6.csv` | 24 | 6 | 2 | 676.869 | 119.544 | 0.177 |
| `OrderUpTo30/Dense_r7d23.csv` | 7 | 23 | 2 | 948.294 | 119.737 | 0.126 |
| `OrderUpTo30/Dense_r23d7.csv` | 23 | 7 | 2 | 771.302 | 113.831 | 0.148 |
| `OrderUpTo30/Dense_r22d8.csv` | 22 | 8 | 5 | 1057.420 | 115.559 | 0.109 |
| `OrderUpTo30/Dense_r8d22.csv` | 8 | 22 | 5 | 1209.434 | 121.932 | 0.101 |
| `OrderUpTo30/Dense_r21d9.csv` | 21 | 9 | 5 | 1011.784 | 112.084 | 0.111 |
| `OrderUpTo30/Dense_r9d21.csv` | 9 | 21 | 5 | 1295.499 | 122.329 | 0.094 |
| `OrderUpTo30/Dense_r10d20.csv` | 10 | 20 | 5 | 1820.835 | 125.454 | 0.069 |
| `OrderUpTo30/Dense_r20d10.csv` | 20 | 10 | 5 | 1603.344 | 116.983 | 0.073 |
| `OrderUpTo30/Dense_r11d19.csv` | 11 | 19 | 5 | 1691.134 | 115.123 | 0.068 |
| `OrderUpTo30/Dense_r19d11.csv` | 19 | 11 | 5 | 1657.053 | 115.135 | 0.069 |
| `OrderUpTo30/Dense_r12d18.csv` | 12 | 18 | 5 | 2165.975 | 125.927 | 0.058 |
| `OrderUpTo30/Dense_r18d12.csv` | 18 | 12 | 5 | 1276.630 | 126.454 | 0.099 |
| `Sparse_r12d18.csv` | 12 | 18 | 2 | 826.050 | 116.164 | 0.141 |
| `OrderUpTo30/Dense_r13d17.csv` | 13 | 17 | 5 | 1474.531 | 126.316 | 0.086 |
| `OrderUpTo30/Dense_r17d13.csv` | 17 | 13 | 5 | 2256.023 | 125.539 | 0.056 |
| `OrderUpTo30/Dense_r14d16.csv` | 14 | 16 | 5 | 2226.926 | 127.725 | 0.057 |
| `OrderUpTo30/Dense_r16d14.csv` | 16 | 14 | 5 | 2154.880 | 116.089 | 0.054 |
| `OrderUpTo30/Dense_r15d15.csv` | 15 | 15 | 5 | 2145.704 | 128.139 | 0.060 |

## Runtime changes

Two narrow fixes accompany this coverage: localize rational observation
arithmetic while preserving polynomial/integer output parents, and avoid cache-key
serialization when no persistent store was selected. Stored keys, cache validation
and solver arithmetic remain unchanged. The copied `_qhsolve.py` is unchanged in
this release. These tests retain its previous arithmetic and execution regressions.

Run the repository's compatibility tests with `sage -python -m unittest discover -s tests -p 'test_upstream*.py' -v`.
