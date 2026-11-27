# Cold races with the optional bulk FLINT backend

Both classes used fixed grace values declared before their first pair.
The Jones class used $N=60$ s and the rational differential class used
$N=0.5$ s. Each pair started fresh Sage workers, prepared its inputs and
libraries before a common start barrier, and used one worker and one native
thread per method. If the runner stopped a competitor at the cutoff, it ended
repetitions for that case. FLINT was explicitly selected for these Hologuess workers.

## Jones

| Knot | Hologuess FLINT | Native Ore | Exact agreement during race |
|---|---:|---:|---|
| $3_1$ | 0.464 s | 0.642 s | Both methods, 3 pairs |
| $4_1$ | 1.767 s | 9.940 s | Both methods, 3 pairs |
| $5_2$ | 0.821 s | $\infty$ | Hologuess matched the stored reference |
| $6_1$ | 7.002 s | $\infty$ | Hologuess matched the stored reference |

Times include the public call, normalization and exact fitting and suffix
checks. The complete cases use medians of three pairs.
The $\infty$ tag means native Ore was stopped without a verified answer at
the 60-second cutoff. These cases show the one completed Hologuess sample.
The controller compared each returned normalized integer Laurent operator
coefficient by coefficient with the frozen Jones reference. This uses
$F(k)=J(k+1)$, with $E$ the forward shift, $QF(k)=q^k F(k)$, and
$EQ=qQE$. The $6_1$ fixture uses the prior Habiro extension verified
against the first 50 stored Jones values.
[Jones summary](jones-summary.json)

## First differential loss from the prior cohort

On row zero of `Dense_r25d4.csv`, Hologuess with FLINT
took **105.078 ms** and native Ore took
**114.182 ms** in the same cold cohort.
Every returned complete bounded operator space matched the independent
reference, including the same fitting and reserved observations.

This is the order-25, degree-4 box with 156 fitting and four held-out terms.
The prior cohort medians were 124.997 ms
for NumPy Hologuess and 123.527 ms for native Ore.
The differential clock includes the public call and conversion to the
canonical bounded space. The controller then compares that space with the
independent exact certificate for the fitting and reserved data.
[Differential summary](differential-summary.json)

## Validation

Audited 8 Jones pairs and 3 differential pairs.
The audit checked input and source hashes, the bridge digest in each
Hologuess candidate, fixed class grace, exact coefficients, and worker cleanup.
[Audit summary](audit.json)

The full 401-test suite passed with the bridge installed. A wheel extracted
outside the checkout successfully built the bridge and reconstructed the same
exact test recurrence with both backends.
