# Cold Jones and Hodge reconstruction benchmarks — 2026-11-26

All **42 fresh-process runs** recovered exactly the same normalized operators within each case, and passed exact fitting and genuinely held-out residual checks. Raw coefficient lists were compared directly, in addition to SHA256 comparison. No solver implementation was changed during benchmarking.

Each entry below is the median of three runs, in seconds. Jobs ran sequentially, with one solver worker, one native arithmetic thread, empty in-memory solver stores, and rotating method order. “Cold” means a new Sage Python process and empty solver cache, not flushed OS filesystem caches.

## Jones data

**Full cold-process wall time**, including interpreter startup/imports, loading the stored exact data, reconstruction and exact validation:

| Knot | Original qhsolve | Hologuess core | Hologuess public API |
|---|---:|---:|---:|
| 3_1 | 2.648 | 3.558 | 3.841 |
| 4_1 | 4.856 | 6.134 | 13.071 |
| 5_2 | 2.782 | 2.710 | not timed |

**Reconstruction plus exact validation only**, excluding startup, imports and data loading:

| Knot | Original qhsolve | Hologuess core | Hologuess public API |
|---|---:|---:|---:|
| 3_1 | 1.002 | 1.941 | 2.196 |
| 4_1 | 3.053 | 4.382 | 11.311 |
| 5_2 | 0.870 | 0.888 | not timed |

The native core comparison uses identical ArraySequence adapters, parameter points, primes, bounds, empty stores and the grouped Sage exact validator on both sides. The public comparison calls the actual `guess_qsequence`, including its built-in Fraction-based fitting check, and then the same grouped validation of the held-out suffix. Core timing must not be presented as public API timing.

The stored prefixes are: 3_1 fits J(1)..J(48), validates unseen J(49)..J(80); 4_1 fits through64, validates65..80; 5_2 fits through88, validates89..94. Both solvers reconstruct in F(k)=J(k+1) coordinates, with E shifting k and Q multiplying by q**k. Search boxes are respectively (order,Q-degree)=(2,6),(3,8),(4,13); outputs have14,64,262 monomials. Normalization removes integer content, sign and a common Laurent q unit.

The public wrapper is substantially slower on4_1. Its exact check iterates over individual monomials and Fraction coefficients; original qhsolve groups coefficients by shift and uses Sage Laurent products. This is a regression from the original refined validator, not evidence that the modular reconstruction arithmetic became slower. The native core has additional cold overhead on3_1 and4_1; worker-initialization/resource-control overhead is a plausible cause, but these timings alone do not isolate it causally. On5_2 the core results are approximately tied.

After the smaller public cases exposed the validation regression, the larger5_2 case was restricted to core comparison; its public entry point was not timed. No claim about its public wall time is made.

## Hodge data

Smooth hypersurfaces of degrees3,4,5. Identical prescribed Liu–Wang support (P_n,P_(n+1),P′_(n+1),P_(n+2)), modular detection through n-degree4, coefficient-degree cap8, independently generated exact Jacobian-ring data. Fit dimensions0..46; validate unseen47..92. Each result also agrees with the independently written hypersurface recurrence formula.

| Hypersurface degree | holo cold process | Hologuess cold process | holo algorithm + validation | Hologuess algorithm + validation |
|---|---:|---:|---:|---:|
| 3 | 1.621 | 1.855 | 0.066 | 0.226 |
| 4 | 1.644 | 1.856 | 0.072 | 0.235 |
| 5 | 1.655 | 1.900 | 0.076 | 0.241 |

Hologuess uses its actual `interpolate_problem` route with measured reference selection, denominator handling, adaptive interpolation and multi-prime reconstruction. Its initial band comes only from the common declared degree cap plus guard, not the answer’s degree. holo uses its modular detector and ascending coefficient-degree exact rational extraction. Both search for the relation; neither receives its coefficients.

Hologuess detection is about26–27times faster (roughly1.5–1.6ms against41–44ms), but extraction/interpolation overhead dominates these tiny order-two, coefficient-degree-two examples. Total algorithm time is about3.2–3.4times slower, and full cold-process time about13–15% slower. This does not establish how larger Hodge ansatzes compare.

## Evidence and limits

[Timing summary](summary.json) · [Benchmark methods](../../README.md)

The data, operators and residual checks are exact; timings are ordinary wall-clock measurements on this machine with other background software uncontrolled. Three repetitions are sufficient to reveal these large differences, not to make a universal scaling claim. These finite-data checks are not all-index annihilation or minimality proofs. Original repositories, data stores and solvers remain unchanged.
