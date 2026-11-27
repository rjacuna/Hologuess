# Why native Ore wins the larger dense differential cases

2026-11-26, Hologuess 0.1.3, SageMath 10.10. Solver and vendor sources are unchanged.

The gap is mainly **expensive duplicated rational validation and repeated
rational-to-modular conversion in Hologuess's generic module route**. It is not
an extra search, a larger answer space, or a slow CRT reconstruction. Native Ore
also uses linear algebra here; its default Hermite–Padé path applies to algebraic
series guessing, not these differential examples.

The original three-pair benchmark median for `Dense_r15d15` is 2.146 s Hologuess
versus 0.128 s native Ore. A new fresh diagnostic call with phase timers gives
2.085 s versus 0.126 s and returns the same independently certified full operator
space. This diagnosis supplements the [recorded benchmark](../2026-11-26-upstream/report.md);
it does not replace or modify that cohort.

## Where Hologuess spends its time

A cProfile run of the same order-15/degree-15 input takes 2.187 s. Its fitting
matrix is 264 by 256, rank 255. Both methods use two prime images, and the space
is one-dimensional. Hologuess has no exceptional-prime resets, skipped primes
or cache hits in this run.

| Hologuess work | Profile time | Share of total |
|---|---:|---:|
| First exact QQ fitting check, after reconstruction | 0.743 s | 34.0% |
| Same exact QQ fitting check again in result construction | 0.736 s | 33.7% |
| Rational-to-modular conversion and basis assembly, excluding RREF | 0.375 s | 17.1% |
| Exact action columns and rational matrix preparation | 0.253 s | 11.6% |
| Modular RREF, both primes combined | 0.065 s | 3.0% |
| Remaining public/output work | about 0.015 s | about 0.7% |

The first check is `solve_module`'s stable-candidate fitting check at
`hologuess/module.py:321`. It immediately calls `_exact_result`, whose loop at
line 147 repeats the same fitting dot products. In this call every module
observation is a fitting row; the benchmark's independently reserved suffix is
checked externally. Each pass multiplies and adds rational entries in Python
rather than using a compiled matrix product. The profile attributes time to each pass separately.
The input numerator/denominator heights reach 3,257/3,123 bits, so repeated
rational addition and reduction is expensive even with a small-coefficient
answer. The two passes account for about 68% of the measured public call.

The next cost is `_modular_image` at `hologuess/module.py:193–199`: for every
prime it walks all 67,584 expanded rational entries, converts each numerator
and denominator to Python integers, reduces them, and computes a modular inverse
for each denominator. There are 135,168 such cell reductions across two images.
The original input has only 279 fitting Taylor coefficients.

The public Ore adapter explicitly selects the generic module backend on these
QQ differential inputs. It does not use the grouped Laurent q-sequence validator
or the action-parameter interpolation route. The optimized qhsolve core remains
unchanged; reusing its RREF/CRT routines alone does not make these surrounding
rational loops efficient.

## What native Ore does differently

Native `guess` reduces the original input coefficients once per prime, then
builds derivative columns recursively in the finite field. It therefore avoids
constructing and reducing a large exact rational action matrix for the modular
solve. Its `guess_raw` still solves the full 264-by-256 finite-field system.

The native cProfile records two complete finite-field construction/solve calls
at 0.024 s combined, including 0.00535 s in Sage's kernel solver. Total native
construction is 0.153 s under cProfile; the unprofiled fresh call is 0.126 s.
Most of the remaining native first-call time is lazy library initialization
inside the public call, including NumPy loading during reconstruction. This is
included in the original construction timing, rather than moved out afterward.

Native CRT/rational reconstruction terminates when its reconstruction heuristic
succeeds; it does not then replay the original QQ fitting equations or require
Hologuess's repeated-candidate stability check. This describes the inspected
public path and options, not all optional native solvers. In the measured case
both engines use two primes anyway, so extra prime count does not explain the gap.

The benchmark still imposes the same external exact full-space and untouched
suffix obligation on both returned answers. The native algorithm's cheaper
internal policy should not be copied by dropping Hologuess's finite-kernel
certificate. The redundant check can be removed while retaining one exact check,
and exact arithmetic itself can be made much cheaper.

Source locations in the installed native package:

- `guessing.py:240–243,275–276`: QQ/ZZ to finite-field guessing.
- `guessing.py:678–690`: reduce the input list before forming action columns.
- `guessing.py:406–434`: construct and solve the finite-field action matrix.
- `guessing.py:866`: differential inputs select `guess_raw`; algebraic C selects HP.
- `nullspace.py:505–531`: Sage's native kernel.
- `guessing.py:1090–1112,633–777`: successful rational reconstruction ends the loop.

The [original comparison](../2026-11-26-upstream/report.md) identifies the benchmark inputs and software versions.

## Exact stage experiments

These tests use the already prepared rational matrix and the independently
certified reference basis. The compiled exact products are zero for that basis;
the same Sage product rejects a deliberately corrupted basis coefficient. Each
alternate construction method is compared entry by entry with the original
rational matrix reduced at the same prime. No fitting equation or coefficient
is omitted.

| Stage on the order-15/degree-15 input | Three-call median |
|---|---:|
| One existing Python QQ residual pass | 735.2 ms |
| Sage exact QQ matrix multiplication on the same equations | 22.6 ms |
| Existing per-entry Python modular conversion, prime 2,130,706,433 | 163.5 ms |
| Sage change_ring of the expanded QQ matrix, same prime | 68.1 ms |
| Reduce input first, recursively build derivative columns, same prime | 4.41 ms |

The exact matrix product is about 32.6 times faster than one scalar pass. Input-
first construction is about 37 times faster than expanded per-cell conversion
at the same prime. These are repeated stage experiments, not fresh end-to-end
solver speedups. They do not include the work needed to integrate these routes
with general parameter fields, held-out intersections, caches and exceptional
prime handling. No optimized runtime has been installed or benchmarked here.

The stage test also records the actual Sage matrix classes: the native-size prime
8,388,593 selects `Matrix_modn_dense_double`; Hologuess's 31-bit prime selects
`Matrix_modn_dense_flint`. The native logged fields and inspected default
constructor imply the former class; this is confirmed by the separate factory
experiment, rather than a class logged in the native profile. Hologuess converts
to NumPy and uses the qhsolve RREF. This backend difference is secondary in this
profile. The 67,584-cell matrix is below the inherited 100,000-
cell blocked-GEMM threshold; tuning that threshold addresses only about 3% of the
current total. CRT, reconstruction and final output conversion are also minor.

## Scaling cross-check and next changes

Single fresh calls with phase timers and no cProfile confirm the pattern:

| Input | Hologuess public plus output | Native public plus output | HG modular images, including RREF | HG repeated-result exact check alone |
|---|---:|---:|---:|---:|
| Dense_r5d6 | 68.5 ms | 108.3 ms | 4.3 ms | 2.3 ms |
| Dense_r4d26 | 407.0 ms | 119.5 ms | 79.2 ms | 117.2 ms |
| Dense_r15d15 | 2,084.5 ms | 126.5 ms | 397.2 ms | 733.0 ms |

Each answer matches the independent complete space and all reserved coefficients.
The small case is dominated by fixed preparation; the larger cases amplify
rational matrix work and the two exact residual passes. Output conversion in the
original large-case cohort is only about 3 ms Hologuess and 5 ms native, so the
full-space output contract does not explain this one-dimensional case's gap.

The priority is to reuse one exact fitting certificate and use compiled/grouped
exact arithmetic, while keeping held-out checks separate. Next, let standard
Ore actions construct modular images from modular inputs and share their input
reductions, preserving complete-kernel reconstruction and cache identity. Kernel
backend tuning comes after these costs. Any implementation requires independent
arithmetic regressions and a new equal-output benchmark; these diagnostics do
not establish the speed of that future implementation.

[Exact stage measurements](stage-experiments.json) · [Public corpus](../../README.md#upstream-compatibility-and-corpus-verification)
