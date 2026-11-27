# Jones q-holonomic cold races against native Ore

Hologuess and native `ore_algebra.guess` were essentially tied on $3_1$.
Hologuess was 4.10 times as fast on $4_1$. On $5_2$ and $6_1$, Hologuess
returned exact answers and native Ore reached the shared 60-second verified
arrival cutoff. [Timing summary](summary.json)

| Knot | Fitting / total terms | Order / degree | Hologuess | Native Ore | Agreement during race |
|---|---:|---:|---:|---:|---|
| $3_1$ | 48 / 80 | 2 / 6 | 0.648 s | 0.650 s | Exact, all 3 pairs |
| $4_1$ | 64 / 80 | 3 / 8 | 2.432 s | 9.983 s | Exact, all 3 pairs |
| $5_2$ | 88 / 94 | 4 / 13 | 0.969 s | $\infty$ | Hologuess matched the stored operator |
| $6_1$ | 120 / 140 | 5 / 16 | 7.531 s | $\infty$ | Hologuess matched the stored operator |

Times include guessing, normalization and exact validation. The two complete
cases show medians of three fresh pairs. The $\infty$ tag means native Ore was
stopped without a verified answer at the 60-second cutoff, measured from the
common start. Those cases show Hologuess's single completed sample. A stop ends
repetitions for that knot. Both receipts record the stop request and verified
process cleanup.

One $N=60$ s was fixed for the entire Jones class before the measured cohort.
That gives about six times the largest earlier Hologuess cold median on these
fixtures. Each pair used fresh Sage processes and one worker and one native
thread per method. The clock started after library loading and exact input
preparation. Hologuess used the public `guess_qsequence` API and the prior
512-point cap. Native Ore used its public `guess` on symbolic
$\mathbb Q(q)[Q][E]$, with `path=[(order, degree)]`, `cut=None`, `ensure=0`
and `ncpus=1`. The search boxes and input prefixes are common to both.

We compared all 14 returned coefficient arrays to the stored operators in
the convention $F(k)=J(k+1)$, $EF(k)=F(k+1)$, $QF(k)=q^kF(k)$, and
$EQ=qQE$. Normalization removes only common rational scalar content and a
common Laurent $q$ monomial. Each returned operator passed exact fitting
and reserved-suffix checks with the same grouped Laurent arithmetic. The
controller then compared the full normalized operator, leaving that comparison
outside the elapsed times in the table. The operators have 14, 64, 262 and 734
terms respectively.

The first three inputs are independently stored Jones values. The $6_1$
input uses the same previously generated Habiro extension as the earlier
scaling benchmark. Its first 50 values agree with stored Jones data, and its
additional 90 values use a separate Habiro $C$ recurrence and two seeds.
Both methods receive the same 120 fitting and 20 reserved values. [Prior scaling report](../2026-11-26-scaling/report.md)

Fresh native-only checks gave both stopped cases up to 180 seconds each to
return an exact operator. These checks ran concurrently after the timed
cohort. For $5_2$, native Ore raised a `TypeError` during its CRT lift.
It attempted to cast $(q^4+q^2+1)/q$ into $\mathbb Z[q]$. For $6_1$, the
native check reached the 180-second limit without a candidate. Cross-method
operator agreement is therefore established for $3_1$ and $4_1$. On $5_2$
and $6_1$, Hologuess independently matches the stored operators.


## Verification

The controller audit checked fixture and source hashes, common barriers,
nonoverlap, exact coefficient arrays and process cleanup. All 28 protocol
tests passed. The two initial launches were discarded after command-line and
Sage rational conversion errors in the benchmark wrapper. The table comes
from a fresh cohort after both fixes.
