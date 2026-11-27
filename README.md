# Hologuess

Hologuess finds candidate recurrences and equations from exact data in SageMath.
Give it values of a sequence or coefficients of a Taylor series, choose the
allowed orders and coefficient degrees, and it returns operators in the chosen
range. For example, from initial Fibonacci values it recovers $S^2-S-1$,
which gives $F_{n+2}=F_{n+1}+F_n$, where $S$ shifts the index forward.

| If you have | Hologuess can find |
|---|---|
| Exact sequence values | A recurrence with polynomial coefficients in the index |
| Ordinary Taylor coefficients | A differential operator or an algebraic equation for the series |
| A sequence of Laurent polynomials in $q$ | A $q$-recurrence with coefficients depending on $q$ and $q^n$ |
| Values of several specified actions | A basis of linear relations among those actions on the observed data |

Hologuess reconstructs candidates using modular arithmetic and checks the
result on the exact inputs. You can reserve some observations to check the
relation after fitting. The returned evidence describes the finite data checked.
Establishing a recurrence for every index requires a separate proof.

The solver builds on JonesRec's `qhsolve`, including its modular linear algebra,
support pruning, adaptive interpolation, reference selection, denominator
clearing, CRT reconstruction and persistent per-point kernels.

The mathematical interface follows [Appendix B of the AJ paper](https://github.com/rjacuna/Preprints/blob/main/aj.pdf).
Specify how an Ore algebra acts on a module, evaluate those actions on your
data, and reconstruct linear relations among the resulting columns. You can
supply custom actions as well as standard shifts and derivatives.

Start with the [SageMath quickstart](examples/quickstart.ipynb) to guess a
recurrence, a differential equation, or a q-recurrence from exact data. The
[worked examples](examples/worked_examples.ipynb) cover additional coefficient
domains, generating functions, and custom actions.

## Install and test

Install [SageMath](https://doc.sagemath.org/html/en/installation/index.html)
with Jupyter support. The notebooks use `ore_algebra` in that same Sage
environment. The commands below use the tested Ore algebra revision with
SageMath 10.10. Platform-specific instructions are in the
[Ore algebra installation guide](https://github.com/mkauers/ore_algebra#installation).

```sh
git clone https://github.com/rjacuna/Hologuess.git
cd Hologuess
sage -pip install --no-build-isolation 'git+https://github.com/mkauers/ore_algebra.git@2904d75321ef50f5b02b5e8bc355c7d9d74a484f'
sage -pip install .
sage -n jupyter examples/quickstart.ipynb
```

Select the SageMath kernel in Jupyter. Run `make test` from the checkout to
check the installation. For development, use `sage -pip install -e .`.

The installation pulls in NumPy and threadpoolctl. The Ore adapter uses
`ore_algebra` for its operator types, and Hologuess implements the reconstruction.
The core $q$-sequence and module solvers can run with SageMath alone and the
installed Python dependencies. [FLINT](#flint-elimination) is an optional
acceleration for the modular solver.

## FLINT elimination

The default `kernel_backend='auto'` prefers an installed FLINT bridge and uses
the original NumPy elimination when no bridge is installed. Build the bridge
against the headers and shared library of one installed FLINT 3 distribution.
Use that distribution's prefix, which can be the local directory of a Sage
build. The measured build uses FLINT 3.4.0.

```sh
sage -python -m hologuess.build_flint --prefix /path/to/flint
```

By default, the builder installs into `$XDG_CACHE_HOME/hologuess/flint-abi1`
or `~/.cache/hologuess/flint-abi1` if `XDG_CACHE_HOME` is unset. It uses
`libhologuess_flint.dylib` on macOS and `libhologuess_flint.so` on Linux. This
location is discovered automatically by `guess_qsequence`, `guess`,
`solve_module`, `interpolate_relation` and `interpolate_problem`.

For a custom `--output`, set `HOLOGUESS_FLINT_LIBRARY` to the result or pass it
as `flint_library` to the solver. The argument takes precedence over the
environment variable, and both take precedence over automatic discovery. An
unusable configured bridge raises an error. Use `kernel_backend='flint'` to
require FLINT, or `kernel_backend='numpy'` to explicitly use NumPy. Result
metadata records the selected `kernel_backend`. In `guess`, `backend` selects
the reconstruction route and `kernel_backend` selects its modular elimination.

Both kernel options use the same reconstruction primes, pivot conventions
and exact checks. The existing cache keys bind the exact input and ordered
support. The FLINT arithmetic tests run when a bridge is installed or
configured via `HOLOGUESS_FLINT_LIBRARY`.

See the [initial FLINT cold races](benchmarks/results/2026-11-26-flint-races/report.md)
for four Jones cases and the first dense differential case. See the
[follow-up races](benchmarks/results/2026-11-26-flint-dense-next3/report.md)
for three more dense differential cases.

## Standard Ore inputs

```python
from sage.all import QQ, PolynomialRing, factorial
from ore_algebra import OreAlgebra
from hologuess import guess

R = PolynomialRing(QQ, 'n')
A = OreAlgebra(R, 'Sn')
result = guess([factorial(n) for n in range(20)], A, order=2, degree=2)
print(result.operator)  # equivalent to Sn - n - 1

R = PolynomialRing(QQ, 'x')
A = OreAlgebra(R, 'Dx')
result = guess([1/factorial(n) for n in range(20)], A, order=2, degree=2)
print(result.operator)  # equivalent to Dx - 1, for the Taylor series exp(x)
```

The adapter covers standard shifts, derivatives on ordinary Taylor coefficients,
forward differences, q-shifts, algebraic power-series equations through truncated
convolution, and rectangular multivariate data with mixed actions. Polynomial and
rational parameter coefficients are supported by explicit bounded parameter
monomials. Prime/extension finite fields and exact algebraic constants use their
exact domain arithmetic. Caller-supplied Ore generator names are retained.

In the standard univariate search, `order` and `degree` give the largest bounds
to try. Hologuess tries smaller orders first and stops when it finds an operator.
Use `path=[(3, 0)]` to try just one pair of bounds, allowing shifts through
$S^3$ with constant coefficients. Each entry is `(max_order, max_coefficient_degree)`.
With several entries, Hologuess tries them in the order listed and stops at the
first one that gives an operator.

This is an Ore-input adapter, not a promise to accept every historical
`ore_algebra.guess` keyword or an unbounded terminating decision procedure.
For parameter fields, `parameter_degree` bounds the polynomial coefficients after
clearing a common denominator. A missing relation is bounded by the actual search;
it never means the input is non-holonomic. A multivariate result exposes
`.operators`, the recovered bounded relation space, without claiming a complete
annihilator ideal or a unique minimal generator. A multivariate output algebra
may be coefficient-localized to avoid constructor defects in the installed Ore
package; its action and generators are retained.

For native univariate QQ(q) q-sequences, `guess` dispatches to the mature qhsolve
route when the requested filters/bounds permit it. `backend='module'` explicitly
selects the generic bounded finite-matrix route; `backend='qhsolve'` requires the
native route. Generic multivariate fitting currently expands coefficient monomials
into scalar observations; it does not yet have the native q-route's interpolation
cost profile. No claim of a speedup for every supported action is made.

For parameter-free QQ observations, including rational observations with a ZZ
operator parent, the standard adapter constructs modular actions from the inputs
reduced once per prime. It reuses derivative, difference, shift and algebraic
series-power images in the requested column order. Exact columns remain available
for validation and cache identity. Parameter rings and other exact constant
fields retain their existing reconstruction routes. If an input has a modular
pole that may cancel under an action or occur only in unused data, the solver
reduces the exact action matrix instead, preserving the original prime policy.
An observation-relative bound also sends oversized cached-action constructions
to that fallback before their arrays are allocated.

Each stable nonempty reconstructed fitting basis receives one exact Sage matrix
product check. Result construction reuses that fitting certificate; held-out
observations and their full relation-space intersection are checked separately.
Replacing a prepared problem's columns or changing its training cut disables its
captured modular construction. Persistent cache hits still replay rank, pivot
profile and residual checks against the actual modular matrix.

## Notebook display and LaTeX export

In a SageMath notebook, `%display latex` uses $S$ for a forward shift and
$D$ for a derivative. When an operator has several shifts or several
derivatives, Hologuess adds their variable subscripts. The $q$-shift uses $E$,
with $Q$ still denoting multiplication by $q^n$. `print(P)` uses the native
Ore names, such as `Sn` and `Dx`.

```python
from hologuess import operator_latex, set_latex_symbols

set_latex_symbols(S=r"\mathcal{S}", D=r"\partial_x")
P = result.operator
print(latex(P))                      # copyable TeX using the notebook's settings
print(operator_latex(P, D="D"))      # different derivative symbol for this export
set_latex_symbols()                  # restore the defaults
```

Use the exact Ore generator name for a specific axis, such as
`set_latex_symbols(Dx=r"\partial_x", Dy=r"\partial_y")`. Exact names take
precedence over action defaults, and each call to `set_latex_symbols` replaces
the prior settings. `operator_latex` also formats operators produced by direct
Ore arithmetic and returns the expression without math delimiters.

## Direct q-sequence reconstruction

```python
from hologuess import guess_qsequence, SQLiteKernelStore

values = [{n*n: 1} for n in range(32)]  # Laurent coefficient dictionaries
result = guess_qsequence(values, max_order=2, max_degree=3,
                        store=SQLiteKernelStore('/tmp/my-q-kernels.sqlite'))
print(result.to_ore())  # equivalent to E - q*Q**2
```

Here E shifts forward, Q multiplies by q**n, and E*Q=q*Q*E. Rational Laurent
coefficients, nonzero starting indices and modular sequence oracles are supported.
The q-route's `max_degree=0` retains qhsolve's adaptive-degree convention; the
standard adapter's `degree=0` is an explicit degree-zero bound. `QSequence` can
supply modular values directly and grow without constructing exact polynomials.
The default memory share and RREF interpolation method are inherited unchanged.
Exact rational input coefficients exclude primes dividing their denominators.
Exact validation also reuses qhsolve's grouped Sage Laurent products, selecting
ZZ or QQ from the actual coefficients. Rational inputs retain degree-envelope
planning and vectorized modular evaluation. Modular oracles may return compatible
prefetched batches with more points than requested.
Native BLAS threads are bounded inside each modular worker, even when NumPy was
already loaded by a notebook. Aggregate pool/task memory follows qhsolve's budget.
Single-worker batches execute in-process with the same chunks and cache writes.
Native limits initialize lazily at actual BLAS use or worker-pool creation and
restore caller settings afterward; unblocked elementwise kernels avoid unused
library discovery. The original prime pool is precomputed and independently
verified, while unfamiliar caller primes still receive primality checks.

## Arbitrary module actions

`OreAction(B, sigma, delta)` applies `B*sigma(v)+delta(v)` entrywise, including
the derivation at every iterate. `module_problem` builds exact columns from these
actions. `ModuleProblem(columns, labels, training_rows=...)` accepts exact observed
action columns directly; `solve_module` returns their full finite kernel basis.
Unknown relation coefficients in this interface are scalars in the constant field;
include parameter/coefficient monomials in the columns to search those coefficients.
`ActionOracle` and `interpolate_relation` reuse qhsolve's parameter interpolation
engine for arbitrary action-observation matrices with a one-dimensional relation
space. Its optional exact validator checks the reconstructed relation.
`interpolate_problem(problem, parameter)` builds this efficient route directly
from exact univariate rational action columns, skips parameter poles, and checks
the reconstruction against all supplied training and held-out observations.

```python
from hologuess import ModuleProblem, interpolate_problem
R = PolynomialRing(QQ, 't'); t = R.gen(); K = R.fraction_field()
f = [K((t+1)**n/(t-2)) for n in range(16)]
problem = ModuleProblem([f[:-1], f[1:]], labels=('f(n)', 'f(n+1)'),
                        training_rows=10)
relation = interpolate_problem(problem, t)  # (t+1)*f(n) - f(n+1)
```

## Evidence and development

Results distinguish finite training-kernel checks, held-out checks, and supplied
exact validators. None is labelled an all-index proof or a global minimality proof.
When held-out equations select linear combinations of training relations, the
metadata records that selection; those equations then participated in selection
and are no longer an untouched independent test of the selected combination.
Metadata records bounds/backend and the scope checked. Exact data are required;
floating-point values are rejected.
Generic finite-matrix cache hits replay modular rank and residual checks before
using stored rank information. This costs rank computation; it is an integrity
check, not a claimed generic-kernel speedup. The q-route retains per-point reuse.

See [design](docs/DESIGN.md), [provenance](docs/source-provenance.json),
[notes](NOTES.md), [mistakes](MISTAKES.md), and [changelog](CHANGELOG.md).
The compatibility baseline is local `ore_algebra` commit
`2904d75321ef50f5b02b5e8bc355c7d9d74a484f`, checked against KJJ,
[Ore Polynomials in Sage](https://arxiv.org/abs/1306.4263), and the
[official guessing documentation](https://www3.risc.jku.at/research/combinat/software/ore_algebra/doc/guessing.html).

Repeated cold-process measurements against original qhsolve and holo are recorded
in the [2026-11-26 benchmark report](benchmarks/results/2026-11-26/report.md).
Those historical 0.1.0 measurements exposed the validation regression corrected
in 0.1.1. The [refinement audit](docs/refinements-audit-20261126.md) records the
source comparison and other restored wrapper refinements.
The [0.1.1 cold rerun](benchmarks/results/2026-11-26-grouped/report.md) includes
45 runs with exact operator agreement. Public 4_1 improves from 13.071s to 6.065s;
the report also records remaining cold overhead and the unchanged Hodge outcome.
The [0.1.2 overhead fixes and rerun](benchmarks/results/2026-11-26-overhead/report.md)
remove the measured startup/evaluation costs. Jones public timings are now faster
than qhsolve on 3_1 and 4_1; 5_2 and full-cold Hodge totals are near parity.
The [larger-case scaling report](benchmarks/results/2026-11-26-scaling/report.md)
adds 36 cold runs: the 734-term 6_1 Jones operator and complete intersections
of two, three and four quadrics. All full operators and held-out checks agree.
Public 6_1 is about 3% slower than qhsolve; the largest Hodge case is 1.47 times
faster in full cold time and 5.55 times faster in reconstruction plus validation.
See [the benchmark reports and protocols](benchmarks/README.md) for input
provenance and timing scope.

## Worked examples and benchmark results

The [worked SageMath notebook](examples/worked_examples.ipynb) gives the
inputs and public calls for each example, with links to the original sources.
The benchmark reports below provide the native timings and exact verification.

Version 0.1.3 records **333 passing integration tests** and a separate **28-test
race-protocol run**. The mathematical ports cover public examples from
ore_algebra, SymPy, Maple gfun, the RISC Guess/qGeneratingFunctions notebooks,
and FriCAS. Their source expressions, hashes, translated finite feature families
and native API differences are retained in the
[Ore/SymPy](tests/upstream_ore_inventory.json),
[gfun](tests/upstream_gfun_inventory.json),
[RISC](tests/upstream_risc_inventory.json) and
[FriCAS](tests/upstream_fricas_inventory.json) inventories. Nonlinear differential,
recurrence and functional examples use exact observable feature relations where
appropriate. These are Hologuess compatibility tests, not claims that native
Maple, Mathematica/RISC or FriCAS test suites were executed.

All **419 DalgGuessing corpus rows** also match independent complete bounded
operator spaces, with exact residual and held-out coefficient checks. The
reference uses upstream Ore finite-field kernels and independent Sage CRT;
the corpus does not supply expected operators. Input aliases and any source
filename-bound corrections remain explicit in its manifest.

The paired Hajij–Levitt-style races use fresh solver processes with libraries
and exact inputs loaded before a common start. The original 0.1.3 and 0.1.4
races used problem-specific grace intervals, fixed from two-sided pilots
before reported repetitions. The Jones and FLINT races use one shared
interval per problem class.
A result counts only after its full bounded space matches the independent
reference. The $\infty$ tag means the competitor was stopped at the cutoff
with no verified answer. Its eventual completion time is unknown. This
construction clock excludes startup and controller-side verification, so it
differs from the earlier full-cold-process measurements. See the
[upstream report](benchmarks/results/2026-11-26-upstream/report.md) and
[input and verification notes](benchmarks/README.md#upstream-compatibility-and-corpus-verification).
The [constant-coefficient recurrence races](benchmarks/results/2026-11-26-constant-recurrence-races/report.md)
compare the FLINT default with native Ore, SymPy, and Wolfram on two SymPy test cases.

In the 0.1.3 cohort, all 114 reported pairs across 38 problems finish with exact
agreement and no censoring; calibrated N ranges from 0.5 to 5 seconds. Hologuess wins seven problem
medians, while native Ore wins 31 and is substantially faster on larger dense
differential cases. These generic finite-module results do not imply a universal
speed advantage from the q-sequence refinements.

Version 0.1.4 passes 390 tests and recovers all 419 corpus spaces exactly.
All 114 race pairs match without censoring. Hologuess wins 13 of 38 problem
medians and its order-15, degree-15 median drops from 2.146 s to 0.344 s.
Native Ore takes 0.131 s on that case. [Full 0.1.4 report](benchmarks/results/2026-11-26-input-actions/report.md).

Run the ports with `make test-upstream` and the controller checks with
`make test-race-protocol`. Corpus preparation, verification and races have
separate Make targets and require fresh output paths.
