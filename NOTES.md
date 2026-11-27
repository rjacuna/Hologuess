# Implementation notes

The source is the current JonesRec aj/qhsolve.py, not the older holo.py in
Tomorrow's Talk. docs/source-provenance.json pins the original bytes and commit.
The source tree had unrelated user changes when this project was created; none
were promoted or modified here.

The q-sequence route retains detect/restrict/plan/extract, blocked/batched modular
elimination, reference measurement, rational denominator clearing, adaptive bands,
cross-prime reconstruction stability and per-point kernel persistence. The generic
action interpolation route uses the same extraction implementation. The generic
finite matrix route returns a relation space and does not assume a principal ideal.

Representing function data by Taylor coefficients is a module observation, not
sampling function values at floating-point points. Custom module actions are
supplied as exact feature columns or batched modular action observations.

Field support follows the explicit adapters and tests. Arbitrary modules need
user-supplied computable actions and enough separating observations; no algorithm
can infer an arbitrary function's global annihilator from a finite prefix alone.

## Cold benchmarks — 2026-11-26

The first repeated fresh-process comparisons are in
benchmarks/results/2026-11-26/report.md. These supersede any performance impression
from examples/compare_qhsolve.py, which runs baseline then package in one process.
The refined reconstruction strategy is retained, but preserving core arithmetic
does not automatically preserve application-level validation or cold startup cost.
Public guess_qsequence on 4_1 took a median 13.071s versus 4.856s for original
qhsolve with its native exact check. The extracted core alone took 6.134s; its
extra cold overhead requires separate profiling before assigning a cause.
On the tiny Hodge LW examples, detection improves while interpolation overhead
dominates the whole operation. Do not cite detector speed as end-to-end speed.

## Restored refinements — 0.1.1

The wrapper delegates exact validation to the original grouped arithmetic,
extended to ZZ/QQ and actual index origins. Rational exponent data remain
available to the degree planner; coefficients are reduced exactly modulo each
prime before the original vectorized evaluator. Native integers stay integers.
Generic line interpolation now prunes inactive action columns with one RREF
per probe, retaining original action coordinates. Modular oracles again permit
K >= requested compatible cached points, matching the original Sequence API.
The scalar search/reconstruction pipeline, constants and alternative interpolation
backend were audited against the pinned original. The audit distinguishes these
restored wrappers from intentional general-module/fixed-support differences.

## Measured overhead — 0.1.2

Profiling separated execution overhead from reconstruction arithmetic. The small
Jones case still has the same 62 point-batch calls, but no longer creates 62
process pools or repeatedly scans native libraries. BLAS/pool controls are lazy,
restoring and operation scoped; concurrent Python-thread limit scopes serialize.
Pure polynomial parameter observations avoid rational-field roundtrips and
redundant inversions. Heterogeneous rational denominators retain an exact batched
rational route, avoiding denominator-LCM expansion. The verified native prime
pool keeps its values/order while avoiding repeated generation/validation.

The 45-run rerun is benchmarks/results/2026-11-26-overhead/report.md. Full cold
Hodge totals are near parity, not proof that every Hodge phase is faster: those
small interpolation algorithms still cost roughly 6–13ms more than direct exact
extraction. Preserve that distinction when reporting performance.

## Larger exact workloads — 2026-11-26

The 36-run scaling cohort extends Jones to permitted local knot 6_1 and Hodge
to complete intersections of two, three and four quadrics. Runtime code remains
0.1.2. Full operator arrays, input/source hashes and all exact fitted/held-out
checks agree. Independent auditing also matches the stored Jones A operators
and regenerates the Hodge input from a different generating-series expansion.

6_1 has 734 operator monomials, 82 retained cells and a 211-point final horizon,
versus 262, 52 and 83 for 5_2. Public cold time is 9.879s versus qhsolve 9.602s;
the extracted core is tied. The largest Hodge operator has 127 terms and
coefficient degree 9: cold medians are 1.792s HG versus 2.637s holo, with
algorithm/check times 0.186s versus 1.035s. Keep cold and algorithm speedups
distinct; the small Hodge cases remain near parity in cold time.

Only 50 independent exact Jones values were stored for 6_1, fewer than required
for this ansatz. Input extension uses a separate finite-validated Habiro C
recurrence and exact Habiro summation, never the target A. All stored 50 match.
The extra heldouts are independent of reconstructed A but recurrence-derived,
not new skein evaluations or an all-index proof. The full report and provenance
are benchmarks/results/2026-11-26-scaling/report.md.

## Upstream mathematical compatibility — 0.1.3

The recorded integration run passes 333 tests; the final race controller has a
separate 28-test passing run. Source inventories distinguish mathematical ports,
explicit finite nonlinear feature families, native search-policy differences,
and private evaluator/formatting APIs. Native Maple, Mathematica/RISC and FriCAS
were not executed. Every one of the 419 published DalgGuessing rows matches the
independently certified complete bounded operator space, rather than just one
operator or its order. A multidimensional kernel remains a full relation space.

The corpus exposed a coefficient-parent boundary: exact rational observations
can lie in the fraction field of the caller's polynomial coefficient ring.
Localize the observation domain for arithmetic while retaining the requested
polynomial operator parent; clear scalar denominators in returned basis vectors
when required by an integral parent. Tests include QQ[t][n] with rational t-data
and ZZ[n] with exact rational list/file input.

Large exact matrices also paid for persistent-cache serialization when no store
was supplied. The no-store path now omits that identity work; actual persistent
stores retain their full input/source-bound keys, rank checks and residual checks.
No interpolation, CRT or reconstruction stability condition was relaxed.

Independent Taylor residuals use exact factorial quotients. Their held-out range
is determined separately for each basis vector from its largest nonzero b-j:
positive coefficient valuation can otherwise hide the last unseen coefficients
behind the common order cutoff. The checker explicitly reaches every withheld
source coefficient and rejects corruption in the final one.

Paired races are separate from compatibility verification and from full-cold
timings. Both public APIs construct the same certified bounded space after a
common setup barrier. A separate pilot supplies a per-problem grace interval,
with symmetric headroom over the slower verified arrival; that interval is fixed
before fresh reported repetitions. The $\infty$ tag records a stop at or after
the cutoff without a verified answer. Process cleanup fails closed: only owned workers/groups
are signaled, and failed inspection or surviving workers cannot be reported as
successful termination. Full evidence belongs in
benchmarks/results/2026-11-26-upstream/report.md.

## Compiled fitting checks and modular input actions — 0.1.4

The dense differential profile identified repeated exact rational fitting checks
and reduction of every expanded rational action entry. The generic solver now
checks each stable candidate basis once with Sage's exact matrix product and
passes the successful fitting certificate to result construction. Independent
held-out checks are still performed on the original observations. If heldouts
select a subspace, the result continues to record that selection rather than
calling it an untouched validation of the selected combinations.

Parameter-free QQ/ZZ adapters capture construction of modular actions from the
reduced original data. Recursive derivative, difference, shift and algebraic-power
images are reused within each prime. The exact action columns remain the source
for the characteristic-zero fitting check and persistent-cache identity; parameter
identities and finite/algebraic constant fields retain their existing route.
Metadata records which primes use modular inputs, which request exact-matrix
fallback and how many candidate fitting checks were performed.

A raw input can have a pole at a prime while the exact action matrix is regular,
through cancellation or because that input is not used by the fitting rows. Such
primes use exact-matrix reduction, preserving acceptance and finite prime budgets.
An observation-relative state-cell bound selects the same fallback before allocating
oversized recursive action caches, including intermediate powers absent from the
selected support.
The capture is also disabled when columns or the fitting cut change after prepare.
Cache hits retain their complete input/source identity and rank/pivot/residual
replay. The qhsolve arithmetic core and previous ZZ/QQ Laurent validation remain
unchanged. Fresh race evidence belongs in
benchmarks/results/2026-11-26-input-actions/report.md.
