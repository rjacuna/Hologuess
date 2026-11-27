# Hologuess 0.1.2 — measured overhead fixes and cold rerun

All **45 new cold runs** passed exact fitting and genuinely held-out checks. Every full normalized operator and input hash agrees with its corresponding 0.1.1 run and with the native baselines. The six cases, bounds, prime values/order, point policy, worker count, native thread count, search support and independent verification are unchanged.

## What was slow, and what changed

1. **Repeated worker startup and library discovery.** The small Jones case performed62 point-batch solves in separate pool lifetimes. Diagnostic profiling measured62 native-library scans totaling1.297s; disabling only that initializer diagnostically reduced reconstruction2.504→1.126s with identical results. The shipped fix does not disable protection: workers=1 executes the identical kernel task in-process, preserving chunks, live masks, counters and per-chunk cache writes. Actual BLAS/pool use initializes a restoring controller once per operation. Elementwise/unblocked work does not scan unused BLAS libraries. Parallel children inherit the prepared controller.
2. **Per-entry rational work on polynomial Hodge data.** The profile counted23,594 Fraction constructions,4,344 entry evaluations and1,090 modular-inverse calls. Polynomial observations now stay native and are evaluated in batches. Unit denominators require no inversion; distinct nonunit denominators share batched inverses. Compatible shared row denominators can be cleared as exact unit row operations; heterogeneous rational rows remain rational, avoiding denominator-LCM growth. All original poles and original full training/held-out identities remain checked.
3. **Repeated proof of known prime constants.** Import-time pool generation and repeated trial division of the same trusted60-prime list cost about40–50ms each in the profile. The pool is now stored exactly, with values/order/construction/primality independently tested. Unfamiliar caller primes are still validated.

Profiles are diagnostic, not cold speedup measurements. Final unprofiled results below use three fresh processes per method, one worker, one native thread, empty stores and rotating method order. OS filesystem caches were not flushed.

## Jones public API — full cold-process time

Median seconds, including process startup/imports, input loading, reconstruction and exact validation:

| Knot | Hologuess 0.1.1 | Hologuess 0.1.2 | Original qhsolve | Hologuess core |
|---|---:|---:|---:|---:|
| 3_1 | 3.594 | 2.142 | 2.584 | 2.160 |
| 4_1 | 6.065 | 3.938 | 4.653 | 3.957 |
| 5_2 | 2.858 | 2.661 | 2.621 | 2.541 |

Hologuess public cold time is about17% lower than qhsolve on3_1 and15% lower on4_1. The5_2 difference is about1.5% in qhsolve’s favor and within the observed run-to-run spread; treat that case as near parity, not a demonstrated win.

## Hodge — same prescribed Liu–Wang search

| Hypersurface degree | HG 0.1.1 cold | HG 0.1.2 cold | holo cold | HG algorithm + validation | holo algorithm + validation |
|---|---:|---:|---:|---:|---:|
| 3 | 1.840 | 1.633 | 1.610 | 0.0766 | 0.0653 |
| 4 | 1.854 | 1.635 | 1.647 | 0.0834 | 0.0709 |
| 5 | 1.852 | 1.624 | 1.616 | 0.0805 | 0.0741 |

The prior0.21–0.23s full-cold Hodge gap is removed: current cold totals are essentially tied within these three-run measurements. Small-case Hologuess algorithm time remains about6–13ms higher than holo’s direct exact extraction, despite being substantially lower than the previous0.23s. No claim is made that every Hodge algorithm phase is faster. The general parameter interpolation/reference/CRT strategy remains in use; it was not replaced with the old solver or a supplied answer.

## Verification and safety

- Stored Jones prefixes remain48/80,64/80,88/94; reconstructed orders/Q-degrees remain(2,6),(3,8),(4,13), with14,64,262 monomials. Coordinates remain F(k)=J(k+1), E forward shift and Q multiplication by q**k.
- Hodge dimensions0..46 fit the same action support and declared bounds; unseen dimensions47..92 pass the same exact residuals and independent hypersurface recurrence comparison.
- Direct full coefficient-list comparison was independently repeated for all45 runs, alongside input/source hashes and all reported medians.
- Serial/pool kernel equality, live masks, per-chunk persistence, warm-cache reuse, exception propagation, native-thread restoration, lazy BLAS initialization and concurrent context restoration are tested.
- The original 60-prime pool is unchanged. No guess bound, stability criterion, interpolation guard, reference/denominator rule or exact verification was weakened.

[Timing summary](summary.json) · [Before and after](before-after.json) · [Previous cohort](../2026-11-26-grouped/report.md)
