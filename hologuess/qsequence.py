"""The JonesRec reconstruction strategy, independent of JonesRec or knot data."""
from dataclasses import dataclass
from fractions import Fraction
import numpy as np

from . import _qhsolve as core
from ._primes import reconstruction_primes


def _modular_array(value, p):
    values = np.asarray(value)
    if not np.issubdtype(values.dtype, np.integer):
        if values.dtype != object or any(isinstance(x, (float, complex)) or
                (hasattr(x, 'parent') and not x.parent().is_exact()) or
                int(x) != x for x in values.flat):
            raise TypeError('Modular points and values must be exact integers.')
    return np.asarray(values % p, dtype=np.int64)


def _polynomial(value):
    if isinstance(value, dict):
        items = value.items()
    elif hasattr(value, 'dict'):
        items = value.dict().items()
    else:
        items = [(0, value)]
    result = {}
    for e, c in items:
        inexact = isinstance(c, (float, complex))
        if hasattr(c, 'parent') and not c.parent().is_exact():
            try: inexact = not c.is_exact()
            except (AttributeError, TypeError, ValueError): inexact = True
        if inexact:
            raise TypeError('Use exact Laurent coefficients, not approximate numbers.')
        if int(e) != e: raise ValueError('Laurent exponents must be integers.')
        if c:
            if isinstance(c, (int, np.integer)):
                result[int(e)] = int(c)
                continue
            numerator = getattr(c, 'numerator', None)
            denominator = getattr(c, 'denominator', None)
            if callable(numerator) and callable(denominator):
                c = Fraction(int(numerator()), int(denominator()))
            else:
                c = Fraction(c)
            result[int(e)] = int(c) if c.denominator == 1 else c
    return result


class QSequence(core.Sequence):
    """Finite exact Laurent polynomials, or a growable modular value oracle.

    Values are indexed from ``start``. ``at(p, count)`` returns K >= count
    parameter points and a (terms, K) modular array. A modular oracle must use
    distinct nonzero points, and stable points as count grows. Supplying exact
    values enables the independent characteristic-zero residual check.
    """
    def __init__(self, values=None, *, at=None, terms=None, grow=None, start=0):
        self.values = None if values is None else [_polynomial(v) for v in values]
        self._polys = self.values
        self._integral = self.values is not None and all(isinstance(c, int)
                         for v in self.values for c in v.values())
        self._modular_polys = {}
        if self.values is None and (at is None or terms is None):
            raise ValueError('Supply exact values or at and terms.')
        self._at, self._grow = at, grow
        if terms is not None and (isinstance(terms, bool) or int(terms) != terms):
            raise ValueError('Term counts must be integers.')
        if isinstance(start, bool) or int(start) != start:
            raise ValueError('Starting indices must be integers.')
        self._terms = len(self.values) if terms is None else int(terms)
        self._cap = self._terms
        self.start = int(start)
        if self._terms < 2 or (self.values is not None and self._terms > len(self.values)):
            raise ValueError('Need at least two available terms.')

    @property
    def terms(self): return self._terms

    @property
    def polys(self):
        return self._polys

    def cap(self, n): self._cap = int(n)

    def grow(self, n):
        if self._grow is None: return None
        count = self._grow(int(n))
        if count is not None: self._terms = self._cap = int(count)
        return count

    def at(self, p, count):
        if self._at is not None:
            points, values = self._at(int(p), int(count))
            points = _modular_array(points, p)
            values = _modular_array(values, p)[:self._cap]
        else:
            if count + 2 >= p: raise ValueError('Not enough distinct parameter points.')
            points = np.arange(2, count + 2, dtype=np.int64)
            if self._integral:
                values = core.eval_laurent(self.polys[:self._cap], points, p)
            else:
                if p not in self._modular_polys:
                    # Coefficient images are cheap to recreate; keep only the
                    # active prime, not a copy of a large sequence per CRT prime.
                    self._modular_polys.clear()
                    reduced = []
                    inverses = {}
                    for poly in self.values:
                        out = {}
                        for e, c in poly.items():
                            denominator = c.denominator
                            if denominator % p == 0: raise ValueError('Coefficient denominator vanishes modulo prime.')
                            if denominator not in inverses:
                                inverses[denominator] = pow(denominator % p, -1, p)
                            out[e] = c.numerator % p * inverses[denominator] % p
                        reduced.append(out)
                    self._modular_polys[p] = reduced
                values = core.eval_laurent(self._modular_polys[p][:self._cap], points, p)
        if points.ndim != 1 or len(points) < count or values.ndim != 2 or values.shape[1] != len(points):
            raise ValueError('Oracle returned inconsistent point/value dimensions.')
        if len(set(map(int, points % p))) != len(points) or np.any(points % p == 0):
            raise ValueError('Oracle points must be distinct and nonzero modulo p.')
        return points % p, values % p


def _residual(coefficients, values, start):
    return core.annihilates_exactly(coefficients, values, start=start)


@dataclass
class QGuess:
    coefficients: dict
    metadata: dict

    @property
    def order(self): return max(b for b, j, e in self.coefficients)

    def to_ore(self, algebra=None):
        """Use E for forward shift and Q for multiplication by q**n."""
        from sage.all import PolynomialRing, QQ
        from ore_algebra import OreAlgebra
        from ._display import _with_latex
        if algebra is None:
            ring = PolynomialRing(QQ, ('q', 'Q'))
            q, Q = ring.gens()
            algebra = OreAlgebra(ring.fraction_field(), ('E', {Q: q * Q}, {}))
        else:
            if algebra.ngens() != 1 or algebra.is_Q() is False:
                raise ValueError('Output needs one standard q-shift generator.')
            Q, q = algebra.is_Q()
        E = algebra.gen()
        operator = sum((algebra(c * q**e * Q**j) * E**b
                        for (b, j, e), c in self.coefficients.items()), algebra.zero())
        return _with_latex(operator)


def guess_qsequence(values, *, start=0, max_order=12, max_degree=0,
                    primes=None, workers=1, band=None, max_points=2400,
                    grow_to=0, store=None, verbose=False,
                    kernel_backend='auto', flint_library=None):
    """Reconstruct a q-recurrence with the current qhsolve pipeline.

    This reports finite-window validity, never an all-index annihilation or
    global minimality theorem. A result of None means no line corner was
    recovered within the requested bounds, not non-holonomicity.

    Modular elimination uses an installed FLINT bridge by default and NumPy
    when no bridge is installed. flint_library can name a specific bridge.
    """
    from ._kernels import kernel_scope
    seq = values if isinstance(values, QSequence) else QSequence(values, start=start)
    seq.cap(seq.terms)
    if any(isinstance(v, bool) or int(v) != v for v in (workers,max_order,max_degree,max_points,grow_to)) or workers < 1 or max_order < 1 or max_degree < 0 or max_points < 10 or grow_to < 0:
        raise ValueError('Invalid search bounds or worker count.')
    primes = reconstruction_primes(primes)
    skipped = []
    if seq.values is not None and not seq._integral:
        denominators = {c.denominator for value in seq.values for c in value.values()}
        skipped = [p for p in primes if any(d % p == 0 for d in denominators)]
        primes = [p for p in primes if p not in skipped]
        if len(primes) < 2: raise ValueError('Need two primes not dividing input coefficient denominators.')
    with kernel_scope(kernel_backend, flint_library) as kernel:
        candidate, info = core.solve(seq, primes, procs=workers, max_order=max_order,
            max_degree=max_degree, band=band, max_points=max_points, grow_to=grow_to,
            store=store, verbose=verbose)
    if not candidate: return None
    # Core windows start at zero; public coefficients act at the actual index.
    candidate = {(b, j, e - j * seq.start): c for (b, j, e), c in candidate.items()}
    info.update(status='guessed', sequence_start=seq.start, all_indices_proved=False,
                kernel_backend='numpy' if kernel is None else 'flint',
                finite_window_valid=None, skipped_input_primes=skipped)
    if seq.values is not None:
        ok, windows = _residual(candidate, seq.values, seq.start)
        if not ok: raise ArithmeticError('Reconstructed operator failed the exact sequence residual.')
        info.update(finite_window_valid=True, exact_windows=windows)
    return QGuess(candidate, info)
