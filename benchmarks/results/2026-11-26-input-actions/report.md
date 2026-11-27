# Hologuess 0.1.4 cold races against native Ore

Hologuess improved from **2.146 s to 0.344 s** on the order-15,
degree-15 dense case, a 6.24× speedup over 0.1.3. Native Ore completed
the same case in **0.131 s** and remains about 2.62× faster there.
Across all 38 problems, Hologuess had the lower median on 13, up from
seven in the previous cohort. Native Ore had the lower median on 25.
Hologuess improved on 37 case medians, with a median speedup of 3.66×
across the 38 ratios. The smallest case moved from 63.5 to 64.7 ms.

## Solver and arithmetic checks

The generic module solver checks a candidate fitting basis with a single
Sage exact matrix product before returning it. Successful validation covers
the complete expanded fitting matrix. Held-out equations still have their
own checks and can restrict the fitting space through its full intersection.

For standard parameter-free actions over QQ or ZZ, the adapter reduces the
input once per prime and forms the modular action columns from it. It reuses
intermediate derivatives, shifts, differences and truncated series powers. The
exact matrix remains available for the characteristic-zero check and cache
identity. Raw input poles and excessive intermediate storage trigger exact
matrix reduction, which preserves the old rules for accepting primes. All
419 corpus rows and every Hologuess race or pilot used two input-built images
and one compiled fitting check. None used the fallback.

The full 390-test suite passed. A separate reconstruction of all 419 corpus
rows recovered the same 426 canonical basis vectors as the original
independent reference and passed exact fitting and reserved-coefficient
checks. The independent checker uses each basis vector's actual support to
reach the end of the reserved input. A standalone wheel smoke check
also passed through the public Ore adapter.

## Race protocol and scope

The rerun used row zero of the same 38 distinct eligible CSV files, the same
fitting prefixes and order/degree boxes, and the same independent reference
spaces as 0.1.3. One duplicate file was deduplicated. The order-30,
degree-zero file remained in the full corpus check and was excluded from the
paired race because native Ore's minimum input length would consume the
reserved coefficient. [Previous cohort](../2026-11-26-upstream/report.md)

A separate fresh paired pilot for each problem set (N) to at least twice
the slower verified arrival, rounded up to a 1/2/5 interval in milliseconds,
with the same 250 ms floor applied before rounding. The selected (N)
was 500 ms for 25 problems and 1 s for 13, fixed before all three reported
repetitions on each problem. Each pair used fresh Sage processes with one
worker and one native thread per method and a shared start barrier. The
published construction clock starts after startup and input preparation
and ends after bounded operator-space conversion. The independent exact
comparison happens after that clock stops.

All **114 reported pairs** and 38 pilot pairs finished with zero censored
runs. Every output matched the complete canonical reference space exactly.
The reference spaces have exact fitting and untouched suffix certificates.
The controller compared each output with the reference, and both workers
received the same fitting data. Source and input hashes matched and
process cleanup passed for every run.

## Construction medians

Each time is the median of three reported races, in milliseconds.
Speedup is the previous Hologuess median divided by the current one.
Both cohorts timed construction after the common barrier.

| Input | (N) (s) | Hologuess 0.1.3 (ms) | Hologuess 0.1.4 (ms) | Native Ore (ms) | HG speedup |
|---|---:|---:|---:|---:|---:|
| `Dense_r5d6.csv` | 0.5 | 63.455 | 64.743 | 115.759 | 0.98× |
| `Dense_r19d3.csv` | 0.5 | 74.032 | 67.806 | 118.399 | 1.09× |
| `Dense_r4d10.csv` | 0.5 | 77.577 | 68.642 | 113.618 | 1.13× |
| `Sparse_r4d10.csv` | 0.5 | 71.459 | 65.679 | 112.313 | 1.09× |
| `OrderUpTo30/Dense_r29d1.csv` | 0.5 | 83.243 | 72.152 | 113.447 | 1.15× |
| `OrderUpTo30/Dense_r1d30.csv` | 0.5 | 87.167 | 69.906 | 115.588 | 1.25× |
| `Sparse_r20d2.csv` | 0.5 | 88.791 | 69.970 | 112.361 | 1.27× |
| `OrderUpTo30/Dense_r2d28.csv` | 0.5 | 165.402 | 83.265 | 109.411 | 1.99× |
| `OrderUpTo30/Dense_r28d2.csv` | 0.5 | 147.122 | 85.859 | 113.734 | 1.71× |
| `Sparse_r4d20.csv` | 0.5 | 160.935 | 91.882 | 117.037 | 1.75× |
| `Sparse_r20d4.csv` | 0.5 | 127.643 | 97.791 | 121.572 | 1.31× |
| `OrderUpTo30/Dense_r3d27.csv` | 0.5 | 247.327 | 117.156 | 137.665 | 2.11× |
| `OrderUpTo30/Dense_r27d3.csv` | 0.5 | 226.103 | 104.108 | 112.814 | 2.17× |
| `OrderUpTo30/Dense_r25d4.csv` | 0.5 | 284.014 | 124.997 | 123.527 | 2.27× |
| `OrderUpTo30/Dense_r4d26.csv` | 0.5 | 435.802 | 129.452 | 124.620 | 3.37× |
| `OrderUpTo30/Dense_r26d4.csv` | 0.5 | 337.682 | 123.758 | 115.870 | 2.73× |
| `OrderUpTo30/Dense_r5d25.csv` | 0.5 | 546.520 | 154.494 | 122.692 | 3.54× |
| `OrderUpTo30/Dense_r25d5.csv` | 0.5 | 480.567 | 150.812 | 121.555 | 3.19× |
| `OrderUpTo30/Dense_r6d24.csv` | 0.5 | 858.126 | 173.955 | 118.301 | 4.93× |
| `OrderUpTo30/Dense_r24d6.csv` | 0.5 | 676.869 | 170.676 | 115.053 | 3.97× |
| `OrderUpTo30/Dense_r7d23.csv` | 0.5 | 948.294 | 197.361 | 117.044 | 4.80× |
| `OrderUpTo30/Dense_r23d7.csv` | 0.5 | 771.302 | 203.512 | 120.123 | 3.79× |
| `OrderUpTo30/Dense_r22d8.csv` | 0.5 | 1057.420 | 229.612 | 115.152 | 4.61× |
| `OrderUpTo30/Dense_r8d22.csv` | 0.5 | 1209.434 | 228.533 | 118.662 | 5.29× |
| `OrderUpTo30/Dense_r21d9.csv` | 0.5 | 1011.784 | 244.654 | 113.200 | 4.14× |
| `OrderUpTo30/Dense_r9d21.csv` | 1 | 1295.499 | 251.093 | 120.562 | 5.16× |
| `OrderUpTo30/Dense_r10d20.csv` | 1 | 1820.835 | 281.253 | 123.219 | 6.47× |
| `OrderUpTo30/Dense_r20d10.csv` | 1 | 1603.344 | 278.502 | 113.670 | 5.76× |
| `OrderUpTo30/Dense_r11d19.csv` | 1 | 1691.134 | 295.618 | 112.120 | 5.72× |
| `OrderUpTo30/Dense_r19d11.csv` | 1 | 1657.053 | 301.871 | 114.997 | 5.49× |
| `OrderUpTo30/Dense_r12d18.csv` | 1 | 2165.975 | 323.164 | 125.614 | 6.70× |
| `OrderUpTo30/Dense_r18d12.csv` | 1 | 1276.630 | 303.068 | 124.925 | 4.21× |
| `Sparse_r12d18.csv` | 1 | 826.050 | 309.725 | 112.581 | 2.67× |
| `OrderUpTo30/Dense_r13d17.csv` | 1 | 1474.531 | 321.701 | 132.913 | 4.58× |
| `OrderUpTo30/Dense_r17d13.csv` | 1 | 2256.023 | 347.799 | 133.813 | 6.49× |
| `OrderUpTo30/Dense_r14d16.csv` | 1 | 2226.926 | 336.785 | 125.412 | 6.61× |
| `OrderUpTo30/Dense_r16d14.csv` | 1 | 2154.880 | 351.856 | 124.674 | 6.12× |
| `OrderUpTo30/Dense_r15d15.csv` | 1 | 2145.704 | 343.919 | 131.388 | 6.24× |

The 0.1.4 code is in commit `fd1ce0d`. The three-run ranges and the public
API, output conversion and setup medians are available in the
[comparison JSON](comparison.json).

[Public corpus](../../README.md#upstream-compatibility-and-corpus-verification)
