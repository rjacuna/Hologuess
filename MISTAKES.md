# Lessons inherited from JonesRec

- A successful Wang reconstruction at one modulus is not a coefficient height
  certificate. Require stable independent prime extensions and exact residuals.
- Reject denominators noninvertible modulo the CRT product.
- Do not take an arbitrary vector out of a higher-dimensional scalar q-kernel.
  For general Ore modules return the relation space with its actual scope instead
  of misapplying the univariate principal-ideal argument.
- A shared normalization must remain fixed across primes. Reference changes,
  denominator changes and exponent-band changes invalidate incompatible residues.
- Measure every witness before declaring a reference Laurent. Clear rational
  denominators instead of endlessly widening a Laurent band that cannot fit them.
- Coefficient-height failure needs more primes; exponent-support failure needs
  a wider band. Never form enormous common denominators from failed reconstructions.
- Record skipped or data-limited boxes as gaps, never certified empty.
- An exact check on fitted windows is not a held-out check or all-index proof.
- Cache actual observations and ordered actions, not just user-friendly names.
- An optimization needs a cold end-to-end equal-output benchmark. Keep the measured
  default interpolation method even when another has better asymptotics.
- Sage polynomial backends may have smaller characteristic limits than NumPy's
  modular arithmetic. Test the actual domain and avoid unsafe backend assumptions.

## 2026-11-26: dropping a refined validator during extraction

The new qsequence._residual loop uses Fraction arithmetic per operator monomial
and sequence term. Original qhsolve.annihilates_exactly already groups by shift
and uses fast Sage Laurent products. The earlier same-process comparison used
the new slow checker on both sides and did not expose this native validation
regression. Fresh-process end-to-end measurements with each actual implementation
make it visible: public 4_1 is about 2.7 times slower in cold wall time. Preserve
all refined stages, including validation, and distinguish core from public API
measurements. The benchmark commit records the regression; it does not fix it.

Resolved in 0.1.1 by delegating to the shared grouped validator over ZZ/QQ.
Independent scalar-reference tests cover rational coefficients, negative Laurent
powers, index origins, corrupted data and nonvacuous checks. The same audit also
found and restored rational degree planning/vectorized evaluation, generic support
pruning and prefetched point batches; the original benchmark records are retained.

## 2026-11-26: resource controls and generic wrappers also have costs

The first copied-core wrapper created 62 one-worker pools for a small Jones
reconstruction, repeating native-library discovery at roughly 20–22ms per pool.
That cost approximately 1.3s despite unchanged modular arithmetic. Small Hodge
polynomial inputs also paid for per-entry Fraction conversion, evaluation and
inversion of unit denominators, and repeated trial division of trusted primes.

Resolved in 0.1.2 using identical in-process tasks for workers=1, restoring limits
activated only at real BLAS/pool use, batched native polynomial evaluation and
validation, and the verified original prime constants. Preserve all cache writes,
resource safety and exact checks; a no-op limiter is a diagnostic, not a fix.
Use end-to-end cold results, not just the faster detector, to assess the outcome.

## 2026-11-26: confusing observations with the operator coefficient parent

The upstream Ore example uses an operator parent over QQ[t][n], but its exact
observations include 1/(n+t). Casting those observations into QQ[t] rejects a
valid input even though polynomial operator coefficients suffice. Existing tests
had avoided the issue by constructing the operator parent over QQ(t) instead.
Version 0.1.3 localizes the observation arithmetic while preserving the caller's
polynomial parent, clearing scalar denominators when an integral output requires
it. Keep literal upstream parents in compatibility regressions.

## 2026-11-26: doing persistent-cache work without a cache

The generic modular image path serialized every exact matrix entry to form a
persistent-cache identity even with store=None, repeating it at every prime.
The no-store path now skips that work. A chosen persistent store still needs its
full mathematical identity and replayed rank/residual checks. Removing unnecessary
serialization must not become weakening cache validity.

## 2026-11-26: an independent checker can lose exactness or miss the suffix

Taylor derivative coefficients contain factorial ratios. Ordinary Python division
can convert those exact integers into floating point at high order; use an exact
integer quotient. Separately, a common len(values)-order cutoff does not ensure
that every basis vector reaches the last unseen input: x^3(D-1), for example,
has positive coefficient valuation. Extend each vector's checked indices using
its actual nonzero b-j reach. Regressions now include high-order exact exponentials
and corruption of the final unseen coefficient. Full-space equality cannot rescue
a shared, incorrect residual checker.

## 2026-11-26: a requested stop is not verified cleanup

A launcher can exit while its solver child continues. Process inspection can
fail, and a signaling permission error can race with process exit. A race must
track its newly owned groups and worker PIDs, record the audit, and verify that
no live owned process remains. Inspection failure or surviving workers fails
closed; an observed signal error may be recorded as a harmless exit race only
after independent inspection confirms termination. Never turn a censored result
into a completion time or a claim that a solver cannot terminate.

## 2026-11-26: rational work around a fast modular kernel

The generic module path checked the same QQ fitting equations after stable
reconstruction and again while building its result. It also expanded rational
action columns before reducing every entry separately at each prime. Reusing
qhsolve's RREF and CRT did not remove those surrounding costs. The 0.1.3 profile
isolates them; its phase experiments are not measurements of an installed speedup.

Version 0.1.4 uses one exact compiled fitting certificate per stable candidate and
constructs standard rational modular actions from reduced original inputs. Keep
the exact fitting and held-out obligations: successful CRT reconstruction alone
does not replace either. Check complete kernels and genuine untouched suffixes
when validating this change, and use fresh public races for performance claims.

Reducing a raw value too eagerly can also reject a prime that the exact action
matrix accepts. An input denominator may cancel under differentiation, or the
value may occur only in unused observations. Such a construction must fall back
to reduction of the actual exact matrix. Finally, a captured constructor can
become stale when a mutable problem replaces columns or changes its fitting cut.
Bind it to the immutable captured columns and count, and retain cache rank and
residual replay rather than treating capture or cache presence as correctness.

A sparse or heavily filtered support can request a high derivative order with
very few observed rows. Caching every intermediate input array would then exceed
the work represented by the requested observation matrix. Bound cached state
cells against input and observed cells before allocation, counting intermediate
powers even when the term filter excludes them, and use the exact-matrix fallback
when that conservative bound is exceeded.
