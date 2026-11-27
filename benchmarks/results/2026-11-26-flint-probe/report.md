# FLINT elimination on the Jones reconstruction

A bulk array call to FLINT reduced the median time through exact validation
by 8–29% on the four frozen Jones fixtures. Each returned operator agrees
coefficient for coefficient with the stored reference and passes the fitting
and reserved-suffix checks. [Jones timings and phase medians](summary.json)

| Knot | Current solve + checks | Bulk FLINT solve + checks | Reduction | Current / bulk process wall |
|---|---:|---:|---:|---:|
| $3_1$ | 0.613 s | 0.435 s | 29.0% | 2.269 / 2.090 s |
| $4_1$ | 2.341 s | 1.747 s | 25.4% | 4.114 / 3.451 s |
| $5_2$ | 0.910 s | 0.801 s | 12.0% | 2.786 / 2.642 s |
| $6_1$ | 6.892 s | 6.326 s | 8.2% | 9.510 / 8.962 s |

These are medians of three serial repetitions per route in fresh Sage processes.
The solver clock starts after input loading and parity checks and ends after
exact fitting and suffix validation. Process wall time includes startup,
input loading, parity checks and output. The measurements use one worker and
one native thread per run.

The first experiment called FLINT through a Sage matrix object for each
evaluation point. That route was slower on all four cases. For example, it
took 3.088 s through checks on $4_1$, versus 2.260 s for its control. The bulk
bridge passes contiguous NumPy arrays to the same FLINT library and copies
field elements in C. It handles the matrices in a batch in one call. The bridge
uses the current 31-bit primes and preserves the pivot profile, live-point
vote and free-coordinate normalization. Each paired repetition records the
same stage call counts and matrix shapes. The diagnostic checks exact modular
RREFs and surviving kernels on rectangular and rank-degenerate test matrices.
All 48 Jones outputs across the two cohorts passed exact verification.

On $6_1$, scalar and batched elimination together fell from about 0.68 s to
0.08 s. Laurent-polynomial evaluation still took about 4.0 s. The case uses
the earlier Habiro extension and independently stored first 50 Jones values.
[Input provenance](../2026-11-26-scaling/report.md)

## The dense rational differential case

Replacing only Hologuess's modular RREF on the order-15, degree-15 Taylor
case moves it from **0.326 s to 0.268 s**. The kernel portion drops from
67.2 ms to 5.5 ms, close to native Ore's 4.7 ms. Native Ore's full
construction took 0.131 s in the same diagnostic. The kernel substitution
closes about 30% of the measured whole-call gap. Hologuess also constructs
the rational action matrix and checks the complete fitting basis exactly
before returning it. All nine differential calls returned the same complete
space and passed the independent Taylor fitting and suffix checks.

The bridge was built against the FLINT 3 headers supplied by Sage on macOS arm64. The [installation instructions](../../../README.md#flint-elimination) describe the current bridge.
