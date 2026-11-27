# Hologuess design

Appendix B of AJ formulates reconstruction in an Ore algebra acting on a module:
choose finitely many action monomials, apply them to the input, and use linear
observations to build the relation matrix. Reconstruct its coefficient functions
from finite-field images. An operator is useful only if the resulting exact
residual vanishes in the intended scope.

Three cooperating routes implement that formulation:

1. QSequence: the current JonesRec scalar q-sequence corner search and full
   reconstruction. This route preserves the original performance machinery.
2. ActionOracle: a caller-defined batch of module-action observations over one
   reconstruction parameter. It reuses the same reference/denominator/interpolation/
   CRT engine for any action support whose observed relation space is a line.
3. ModuleProblem: exact observed action columns with rational or finite-field
   coefficients, including polynomial/rational parameter identities. Modular rank,
   canonical kernel reconstruction and exact residual checks recover the full
   bounded finite relation space. Mixed and multivariate supports use this route.

The Ore adapter constructs these columns for the standard actions documented by
ore_algebra.guess. It uses ore_algebra for algebra construction and recognition,
not for guessing or kernel solving. User-defined actions bypass the adapter.

These routes must not be confused with an all-index proof. Finite matrix kernel
validity, held-out observation validity, a supplied exact validator and a global
annihilation theorem are distinct claims. Search bounds are explicit and finite.
An absent result says nothing about holonomicity outside those bounds.

Copied arithmetic modules have package-relative imports. The action extension to
_qhsolve accepts a (observation, action, parameter-point) tensor alongside its
original sequence matrix; q-sequence arithmetic is unchanged. Tensor cache keys
hash the actual modular action matrix and ordered support. Fast interpolation's
optional NTT/PM-Basis backend is preserved; measured RREF interpolation stays default.

Execution policy avoids introducing work unrelated to the algebra. One-worker
tasks run directly with identical kernel chunks and per-chunk persistence. Native
library discovery is lazy and scoped across a reconstruction, with limits restored
after BLAS work or failures. Polynomial parameter rows use batched evaluation and
native ZZ/QQ residual products; shared denominators may be cleared by exact unit
row scaling, while heterogeneous rational rows retain rational evaluation. Original
poles and all original training/held-out identities remain enforced.

The generic finite-kernel route checks a stable reconstructed basis by multiplying
the exact expanded fitting matrix by its basis matrix in Sage. Successful result
construction consumes that certificate instead of recomputing the original
fitting dot products. Expansion clears nonzero row denominators and separates
parameter monomials, preserving the fitting kernel. An unsuccessful exact check
continues reconstruction; no stability or prime-profile condition is relaxed.
Non-QQ exact-field kernels receive the same exact product check. Full-rank modular
images retain their finite rank certificate when there are no basis vectors to
check. Held-out rows remain outside fitting and retain their original exact
checks and full-space intersection behavior.

For standard Ore actions on parameter-free rational input, the adapter also
captures a modular matrix constructor. It reduces the original observations once
per prime, recursively forms action images and applies the prescribed coefficient
monomials. Its rows, filters, starts and columns match the exact training matrix.
Parameter identities and other exact domains keep their existing matrix route.
Raw input or q poles request reduction of the exact expanded matrix for that
prime: an unused input or a factor cancelled by the action must not create an
extra rejected prime.
Before allocating cached action arrays, a conservative state-cell bound relative
to the input and fitting observations selects exact-matrix fallback when needed.

The captured constructor is valid only while the immutable exact columns are the
same tuple and the fitting-row count is unchanged. A replacement of either
invalidates the capture and selects the exact-matrix route. Cache identity still
uses the exact fitting observations in their ordered column coordinates, source
and Sage identities and the prime. An input-derived image does not replace that
identity or bypass rank, pivot-profile and residual replay of cached kernels.
The qhsolve arithmetic core, CRT normalization and reconstruction strategy are
unchanged by these adapter and validation changes.

The optional FLINT bridge replaces modular RREF and the elimination within
each batch. It accepts contiguous integer arrays and uses exact word-size
field operations at the existing primes. Each matrix's canonical RREF fixes
its pivot profile. A sequential majority vote over surviving profiles
recovers the original batch mask, including ties. Surviving basis vectors
use the same identity on their free columns, so they can share the same
cache coordinates. The default selects an installed or configured FLINT
bridge, with NumPy as the fallback when no bridge is available. Kernel
selection is scoped to one solver call and is inherited by its forked
workers. Laurent evaluation, the choice of reference cell, interpolation
support, CRT and exact residual checks use the same reconstruction code.
