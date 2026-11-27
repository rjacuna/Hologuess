# Larger cold reconstruction benchmarks — 2026-11-26

All **36 fresh-process runs** passed exact fitting and genuinely held-out
residual checks. Full primitive operators agree coefficient-for-coefficient
across methods and repetitions. Both Jones operators also agree directly with
the stored reference operators. Hologuess runtime remains unchanged at 0.1.2.

The larger Jones case retains qhsolve's scaling within about 3% public API
overhead. On the largest Hodge case here, Hologuess is 1.47 times faster in full
cold-process time and 5.55 times faster in reconstruction plus validation.
These are measured finite workloads, not asymptotic complexity claims.

## Protocol

Each method runs three times in a new Sage Python process, one solver worker,
one native arithmetic thread and an empty in-memory solver store. Method order
rotates between repetitions. Jobs run serially; OS filesystem caches are not
flushed. The primary metric is parent-measured subprocess wall time, including
startup, imports, data loading/generation, reconstruction, exact validation,
diagnostics, serialization and process shutdown. Solver/check phases are also
reported separately. The public API's built-in fitting check stays enabled.

Within each family/case, fitting data, fitting/held-out windows, declared bounds,
action support, prime and point policy are matched. No supplied answer seeds
reconstruction. All sources were hashed before and after the cohort; no solver
or benchmark source changed during timing. No original project file or store
was modified. The per-run timeout was 180 seconds; every run finished normally.

## Jones: more cells, interpolation points and exact input

| Case | Fit / total terms | Order / Q degree | Full / retained cells | Operator monomials | Final points per prime | Fitting input monomials |
|---|---:|---:|---:|---:|---:|---:|
| 5_2 anchor | 88 / 94 | 4 / 13 | 70 / 52 | 262 | 83 | 559,265 |
| 6_1 | 120 / 140 | 5 / 16 | 102 / 82 | 734 | 211 | 1,710,049 |

The reported point count is the final extraction horizon, not the cumulative
number of nullspace probes. Both cases used two reconstruction primes, a
512-point cap and measured reference selection. The 6_1 reconstruction exercised
denominator clearing and adaptive widening from the data-predicted band. Its
fitting coefficients reach 85 bits, versus 55 bits for 5_2; the longest held-out
polynomial has 58,026 monomials and coefficients reaching 100 bits. This is an
increase in algebraic and input size, not just an unchanged operator on a longer
prefix.

Median seconds:

| Case | qhsolve cold | HG core cold | HG public cold | qhsolve algorithm + checks | HG core algorithm + checks | HG public algorithm + checks |
|---|---:|---:|---:|---:|---:|---:|
| 5_2 | 2.698 | 2.695 | 2.734 | 0.836 | 0.841 | 0.898 |
| 6_1 | 9.602 | 9.564 | 9.879 | 6.687 | 6.735 | 7.023 |

The extracted core remains essentially tied with native qhsolve. Public 6_1
adds 0.276 seconds, about 2.9%, to median cold time. Its three cold times span
9.805–10.060 seconds, versus 9.529–9.661 seconds for qhsolve. It is slightly
slower on this case; the measurements do not show a Jones speed advantage.
From 5_2 to 6_1, full cold time grows 3.56 times for qhsolve and 3.61 times for
the public Hologuess API. Reconstruction plus exact checks grows about eightfold
for both engines as the retained matrix width, point horizon and polynomial
input size all increase.

The benchmark process's peak RSS reaches 3.51 GiB on 6_1, versus 0.91 GiB on
5_2. These are maximum individual parent-process RSS readings, not aggregate
parent-plus-worker peaks.

### Exact input provenance

The larger knot in this comparison is 6_1. JonesRec has 50 independently
stored exact Jones values for it, fewer than the 117 terms needed to detect
this full rectangle with the existing margin. The larger input was prepared
outside timing using the **separate finite-validated Habiro C recurrence**, row
337, and the two stored seeds J(1), J(2). Its leading coefficient is a Laurent
monomial unit, so extending C uses integer Laurent shifts and additions. Horner
evaluation of the exact Habiro sum then produces J(1) through J(140). All 50
independently stored Jones values agree exactly. The target A recurrence was
never used to generate the input.

Preparation took about 13.2 seconds, excluded from solver timing. The full 140-term input had SHA-256
`d8249ee587b7cdd180aedba76e2c0465816aa1f2c816cd7c223b49a4b5fe1350`.

Coordinates are F(k)=J(k+1), E F(k)=F(k+1), Q F(k)=q^k F(k), EQ=qQE. Full
operators are compared after removing only integer content, sign and a common
Laurent q unit. There are 6 and 20 unseen-reaching windows for 5_2 and 6_1.
Independent auditing also matched all nine runs per knot to stored valid A
rows 342 and 340, respectively, without any index or mirror transformation.

The extra Jones inputs are recurrence-derived, not new independent skein
evaluations. The C recurrence's finite validation is not promoted here to an
all-index theorem. These checks establish exact reconstruction agreement on
the declared input and do not prove global annihilation or minimality.

## Hodge: increasing complete-intersection complexity

The inputs are primitive Hodge polynomials of intersections of two, three and
four quadrics, generated by the exact Hirzebruch series. They use the same four
Liu–Wang actions P_n, P_(n+1), derivative of P_(n+1), P_(n+2), with polynomial
coefficients in n and parameter x. Both methods sweep n-degree with bound 6
and share the coefficient x-degree cap 20. Hologuess retains its modular
interpolation/reference/CRT route; holo retains its original modular detection
and ascending-degree exact extraction. Neither receives the actual answer's
degree as an interpolation hint.

| Quadrics | Fit / total dimension | Detected n-degree | Coefficient x-degree | Action columns | Operator terms | Exact held-out windows |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 64 / 96 | 1 | 2 | 8 | 9 | 32 |
| 3 | 80 / 120 | 2 | 4 | 12 | 38 | 40 |
| 4 | 96 / 144 | 3 | 9 | 16 | 127 | 48 |

Fit/total dimensions are inclusive endpoints starting at dimension zero. All
observed relation spaces are lines over QQ(x): a modular rank lower bound and
the recovered exact nonzero relation establish this for the finite observation
matrix. Full primitive polynomial coefficient vectors are compared up to a
common nonzero QQ(x) unit. No arbitrary vector is chosen from a higher-dimensional
kernel.

Median seconds:

| Quadrics | holo cold | HG cold | holo algorithm + checks | HG algorithm + checks | Cold speedup |
|---|---:|---:|---:|---:|---:|
| 2 | 1.658 | 1.678 | 0.0682 | 0.0785 | 0.99x |
| 3 | 1.747 | 1.745 | 0.1702 | 0.1392 | 1.00x |
| 4 | 2.637 | 1.792 | 1.0355 | 0.1864 | 1.47x |

The larger coefficient degree makes holo's exact extraction phase grow from
0.0238 to 0.9787 seconds; Hologuess's extraction phase grows from 0.0764 to
0.1780 seconds. At four quadrics, algorithm plus validation is 5.55 times
faster and full cold time is about 32% lower. The three cold runs range from
1.772–1.814 seconds for Hologuess and 2.630–2.731 seconds for holo. Small cases
remain dominated by common startup costs; Hologuess is slightly slower on the
smallest algorithm phase. The data support a scaling advantage on this ladder,
not a universal performance claim for arbitrary modules.

The input generator was independently checked against an all-quadrics closed
generating series and the hypersurface Jacobian-ring formula. A corruption in
the final unseen window is rejected by the exact residual checker. The full
suite passes **110 tests**. No sign condition or all-index Liu–Wang proof is
claimed merely from these finite reconstruction benchmarks.

[Timing summary](summary.json) · [Earlier small-case report](../2026-11-26-overhead/report.md)
