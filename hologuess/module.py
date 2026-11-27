"""Finite observable-module kernels, with exact checks and modular reconstruction.

The rows are observations, not a presumed faithful representation of a whole
function module. A complete kernel of these rows is a finite-data statement.
There is no principal-ideal or one-dimensional-kernel assumption here.
"""
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import numpy as np

from . import _qhsolve as core


@dataclass
class ModuleProblem:
    columns: object
    labels: object = None
    training_rows: object = None
    name: str = 'module observations'

    def __post_init__(self):
        self.columns = tuple(tuple(col) for col in self.columns)
        if not self.columns:
            raise ValueError('At least one action column is required.')
        n = len(self.columns[0])
        if not n or any(len(c) != n for c in self.columns):
            raise ValueError('Columns must have the same positive observation count.')
        self.labels = tuple(range(len(self.columns))) if self.labels is None else tuple(self.labels)
        if len(self.labels) != len(self.columns):
            raise ValueError('There must be one label per action column.')
        if self.training_rows is None:
            self.training_rows = n
        if (isinstance(self.training_rows, bool) or
                int(self.training_rows) != self.training_rows or
                not 1 <= self.training_rows <= n):
            raise ValueError('training_rows must be between one and the observation count.')
        self.training_rows = int(self.training_rows)


@dataclass
class ModuleResult:
    basis: tuple
    labels: tuple
    status: str
    metadata: dict = field(default_factory=dict)
    holdout_failures: tuple = ()
    validated_basis: tuple = ()

    @property
    def coefficients(self):
        return self.basis

    @property
    def valid_basis(self):
        """Complete subspace satisfying training and heldout observations.

        When training basis vectors fail individually, this may contain new
        linear combinations selected by the heldout equations. Such selection
        is recorded explicitly; it is not a further independent holdout test.
        """
        return self.validated_basis

    @property
    def relations(self):
        return self.basis


def _parameter_structure(parent):
    """Distinguish parameter rings from exact constant fields, including extensions."""
    from sage.all import QQ, ZZ
    from sage.rings.fraction_field import FractionField_generic
    if parent in (ZZ, QQ):
        return QQ, ()
    if isinstance(parent, FractionField_generic):
        return _parameter_structure(parent.ring())
    if parent.is_field():
        if not parent.is_exact():
            raise TypeError('Observable module reconstruction needs exact coefficients.')
        return parent, ()
    base = parent.base_ring()
    if base is parent or not hasattr(parent, 'variable_names'):
        raise TypeError('Use exact numbers, polynomials or rational functions as observations.')
    constants, names = _parameter_structure(base)
    own = tuple(parent.variable_names())
    if set(own) & set(names):
        raise ValueError('Parameter names must be distinct across coefficient towers.')
    return constants, names + own


def _flatten(value, parent, target):
    """Map polynomial/fraction towers into one parameter fraction field exactly."""
    from sage.rings.fraction_field import FractionField_generic
    if isinstance(parent, FractionField_generic):
        ring = parent.ring()
        return _flatten(value.numerator(), ring, target) / _flatten(value.denominator(), ring, target)
    if parent.is_field() or parent.base_ring() is parent:
        return target(value)
    names = tuple(parent.variable_names())
    gens = {str(g): g for g in target.gens()}
    out = target.zero()
    for powers, coefficient in parent(value).dict().items():
        try:
            exponents = tuple(powers)
        except TypeError:
            exponents = (int(powers),)
        term = _flatten(coefficient, parent.base_ring(), target)
        for name, e in zip(names, exponents):
            term *= gens[name] ** e
        out += term
    return out


def _expand(problem):
    """Clear denominators rowwise, then expand parameter monomials into scalar rows."""
    from sage.all import QQ, Sequence, PolynomialRing, matrix, lcm
    values = [v for col in problem.columns for v in col]
    parent = Sequence(values, use_sage_types=True).universe()
    constants, names = _parameter_structure(parent)
    ncols, count = len(problem.columns), len(problem.columns[0])
    if not names:
        rows = [[constants(problem.columns[j][i]) for j in range(ncols)] for i in range(count)]
        return constants, matrix(constants, problem.training_rows, ncols,
                                 sum(rows[:problem.training_rows], [])), rows, ()
    ring = PolynomialRing(constants, names=names)
    target = ring.fraction_field()
    rows, expanded = [], []
    for i in range(count):
        row = [_flatten(parent(problem.columns[j][i]), parent, target) for j in range(ncols)]
        rows.append(row)
        if i >= problem.training_rows:
            continue
        denominator = ring.one()
        for value in row:
            denominator = lcm(denominator, value.denominator())
        polynomials = [ring(value * denominator) for value in row]
        keys = sorted(set().union(*(set(p.dict()) for p in polynomials)))
        expanded.extend([[constants(p.dict().get(key, 0)) for p in polynomials] for key in keys])
    return constants, matrix(constants, len(expanded), ncols, sum(expanded, [])), rows, names


def _annihilates_matrix(observations, basis):
    """Check every fitting equation with the exact compiled field backend."""
    if not basis:
        return True
    from sage.all import matrix
    vectors = matrix(observations.base_ring(), len(basis), observations.ncols(),
                     [c for vector in basis for c in vector])
    return (observations * vectors.transpose()).is_zero()


def _exact_result(problem, basis, rank, rows, metadata, *, training_verified):
    # The complete expanded fitting matrix was checked before this call.
    # Clearing nonzero row denominators and expanding parameter monomials
    # preserves its kernel, so replaying the original fitting rows is redundant.
    if not training_verified:
        raise ArithmeticError('A module result requires an exact fitting certificate.')
    training = problem.training_rows
    failures = []
    for k, coefficients in enumerate(basis):
        if any(sum((a * c for a, c in zip(row, coefficients)), 0) != 0 for row in rows[training:]):
            failures.append(k)
    basis = tuple(tuple(c for c in row) for row in basis)
    validated_basis = basis
    if failures:
        # Holdout constraints are linear on the WHOLE training kernel. Testing
        # its basis members separately loses valid cancelling combinations.
        residual_columns = [
            [sum((a*c for a,c in zip(row, v)), 0) for row in rows[training:]]
            for v in basis]
        constants, constraints, _, _ = _expand(ModuleProblem(residual_columns))
        combinations = constraints.right_kernel().basis()
        validated_basis = tuple(
            tuple(sum((weight * basis[k][j] for k, weight in enumerate(v)), constants.zero())
                  for j in range(len(problem.columns)))
            for v in combinations)
    metadata.update(rank=rank, nullity=len(basis), exact_training=True,
                    exact_training_backend='compiled_field_matrix_product' if basis else 'full_rank_certificate',
                    exact_holdout=None if training == len(rows) else not failures,
                    training_observations=training, holdout_observations=len(rows)-training,
                    all_indices_proved=False, full_training_kernel=True,
                    valid_nullity=len(validated_basis), full_holdout_intersection=True,
                    holdout_used_to_select=bool(failures))
    status = ('holdout_failed' if failures else
              'no_relations' if not basis else
              'validated' if training < len(rows) else 'finite_kernel')
    return ModuleResult(basis, problem.labels, status, metadata,
                        tuple(failures), validated_basis)


def _cache_key(matrix, p):
    # Matrix entries are exact rationals, so repr is canonical within this Sage runtime.
    import sage.version
    identity = ('module-rref-v1', sha256(Path(core.__file__).read_bytes()).hexdigest(),
                sha256(Path(__file__).read_bytes()).hexdigest(), sage.version.version,
                matrix.nrows(), matrix.ncols(), tuple(map(str, matrix.list())), int(p))
    return sha256(repr(identity).encode()).digest()


def _modular_image(matrix, p, store, modular_matrix=None):
    # Serializing every exact entry is unnecessary when the caller has chosen
    # no persistent store, and dominates large rational observation matrices.
    key = _cache_key(matrix, p) if store is not None else None
    cached = store.get_many([key]) if store is not None else {}
    n = matrix.ncols()
    data = None
    if modular_matrix is not None:
        try:
            data = np.asarray(modular_matrix(p))
        except NotImplementedError:
            # A raw input may have a pole cancelled by the action. Preserve
            # prime acceptance by reducing the actual exact matrix in that case.
            pass
        if data is not None:
            if (data.shape != (matrix.nrows(), n) or
                    not np.issubdtype(data.dtype, np.integer) or
                    np.any(data < 0) or np.any(data >= p)):
                raise ValueError('Invalid input-derived modular observation matrix.')
            data = data.astype(np.int64, copy=False)
    if data is None:
        entries = []
        for coefficient in matrix.list():
            denominator = int(coefficient.denominator()) % p
            if not denominator:
                raise ZeroDivisionError('A coefficient denominator vanishes at this prime.')
            entries.append(int(coefficient.numerator()) % p * pow(denominator, -1, p) % p)
        data = np.asarray(entries, dtype=np.int64).reshape((matrix.nrows(), n))
    if key in cached:
        alive, payload = cached[key]
        payload = np.asarray(payload)
        if (not alive or payload.ndim != 1 or
                not np.issubdtype(payload.dtype, np.integer) or
                len(payload) < n or (len(payload)-n) % n or
                any(int(x) not in (0, 1) for x in payload[:n])):
            raise ValueError('Invalid cached module kernel payload.')
        pivots = tuple(i for i in range(n) if payload[i])
        free = [i for i in range(n) if i not in pivots]
        basis = np.asarray(payload[n:], dtype=np.int64).reshape((-1, n))
        if (basis.shape[0] != len(free) or
                np.any(basis < 0) or np.any(basis >= p) or
                not np.array_equal(basis[:, free], np.eye(len(free), dtype=np.int64))):
            raise ValueError('Cached module kernel is not a normalized complete basis.')
        # A valid nullspace vector is only an upper-rank certificate. In
        # particular an empty cached basis proves nothing about full rank.
        # Recompute the actual input rank before trusting completeness; source
        # hashes and immutable storage do not make corrupt rank claims true.
        actual_pivots = tuple(core.rref_mod(data, p)[1])
        if len(actual_pivots) != len(pivots):
            raise ValueError('Cached module kernel rank differs from the actual input.')
        if actual_pivots != pivots:
            raise ValueError('Cached module kernel pivot profile differs from the actual input.')
        for row in data:
            for vector in basis:
                if sum(int(a)*int(c) for a, c in zip(row, vector)) % p:
                    raise ValueError('Cached module kernel fails an actual modular residual.')
        return pivots, basis, True
    reduced, pivots = core.rref_mod(data, p)
    pivots = tuple(pivots)
    free = [i for i in range(n) if i not in pivots]
    basis = np.zeros((len(free), n), dtype=np.int64)
    for k, f in enumerate(free):
        basis[k, f] = 1
        for i, pivot in enumerate(pivots):
            basis[k, pivot] = int(-reduced[i, f]) % p
    if store is not None:
        mask = np.asarray([int(i in pivots) for i in range(n)], dtype=np.int64)
        store.put_many([(key, p, 1, np.concatenate([mask, basis.reshape(-1)]))])
    return pivots, basis, False


def solve_module(problem, *, primes=None, min_primes=2, max_primes=60, store=None,
                 kernel_backend='auto', flint_library=None):
    """Return the complete constant-coefficient kernel of finite observations.

    Polynomial/rational-function parameters are identities: their monomials
    become scalar rows after clearing denominators. Include parameter powers
    among the action columns when polynomial parameter coefficients are wanted.
    QQ kernels use modular RREF, CRT and Wang reconstruction. An installed
    FLINT bridge is preferred for modular elimination. Finite and algebraic
    extension constant fields use Sage's exact field kernel.
    A separate heldout suffix is checked without contributing fitting rows.
    """
    from ._kernels import kernel_scope
    with kernel_scope(kernel_backend, flint_library) as kernel:
        result = _solve_module(problem, primes, min_primes, max_primes, store)
    result.metadata['kernel_backend'] = 'numpy' if kernel is None else 'flint'
    return result


def _solve_module(problem, primes, min_primes, max_primes, store):
    from sage.all import QQ, ZZ, is_prime
    if not isinstance(problem, ModuleProblem):
        problem = ModuleProblem(problem)
    if isinstance(min_primes, bool) or int(min_primes) != min_primes or min_primes < 2:
        raise ValueError('min_primes must be an integer at least two.')
    if isinstance(max_primes, bool) or int(max_primes) != max_primes or max_primes < min_primes:
        raise ValueError('max_primes must be at least min_primes.')
    constants, matrix, rows, parameters = _expand(problem)
    metadata = dict(name=problem.name, coefficient_field=str(constants),
                    expanded_rows=matrix.nrows(), columns=matrix.ncols(),
                    parameters=parameters, attempted_primes=[], skipped_primes=[],
                    profile_resets=0, cache_hits=0,
                    modular_input_primes=[], modular_input_fallbacks=[], exact_training_checks=0)
    if constants != QQ:
        basis = tuple(tuple(v) for v in matrix.right_kernel().basis())
        metadata['exact_training_checks'] += int(bool(basis))
        if not _annihilates_matrix(matrix, basis):
            raise ArithmeticError('An exact field kernel failed its fitting observations.')
        metadata['backend'] = 'exact_constant_field'
        return _exact_result(problem, basis, matrix.ncols()-len(basis), rows, metadata,
                             training_verified=True)
    builder = getattr(problem, '_modular_matrix', None)
    # Adapter columns are immutable tuples of exact QQ entries. Replacing them
    # or changing the fitting cut invalidates their captured input construction.
    if (not callable(builder) or
            getattr(problem, '_modular_columns', None) is not problem.columns or
            getattr(problem, '_modular_training_rows', None) != problem.training_rows):
        builder = None

    def modular_matrix(p):
        try:
            data = builder(p)
        except NotImplementedError:
            metadata['modular_input_fallbacks'].append(p)
            raise
        metadata['modular_input_primes'].append(p)
        return data
    selected = list(core.PRIMES if primes is None else primes)
    if len(set(map(int, selected))) != len(selected):
        raise ValueError('Reconstruction primes must be distinct.')
    if any(int(p) != p or p < 3 or p >= 2**31 or not is_prime(p) for p in selected):
        raise ValueError('Use distinct odd primes below 2^31 for exact modular arithmetic.')
    selected = list(map(int, selected[:int(max_primes)]))
    if len(selected) < min_primes:
        raise ValueError('Not enough reconstruction primes for min_primes.')
    metadata['backend'] = 'modular_qhsolve'
    profile, images, used, previous = None, [], [], None
    for p in selected:
        metadata['attempted_primes'].append(p)
        try:
            pivots, image, cached = _modular_image(matrix, p, store,
                                                  modular_matrix if builder is not None else None)
        except ZeroDivisionError:
            metadata['skipped_primes'].append((p, 'denominator'))
            continue
        metadata['cache_hits'] += int(cached)
        # Exact RREF has maximum rank and lexicographically earliest pivot columns.
        if profile is None or len(pivots) > len(profile) or (len(pivots) == len(profile) and pivots < profile):
            if profile is not None:
                metadata['profile_resets'] += 1
            profile, images, used, previous = pivots, [], [], None
        elif pivots != profile:
            metadata['skipped_primes'].append((p, 'exceptional_pivot_profile'))
            continue
        images.append(image)
        used.append(p)
        metadata.update(primes=tuple(used), pivot_columns=profile)
        if len(profile) == matrix.ncols():
            return _exact_result(problem, (), len(profile), rows, metadata,
                                 training_verified=True)
        lifts, modulus = core.crt_pair(images, used)
        candidate = []
        failed = False
        for row in lifts:
            vector = []
            for coefficient in row:
                fraction = core.ratrec(int(coefficient), modulus)
                if fraction is None:
                    failed = True
                    break
                a, b = fraction
                vector.append(QQ(a) / QQ(b))
            if failed:
                break
            candidate.append(tuple(vector))
        candidate = tuple(candidate) if not failed else None
        stable = candidate is not None and candidate == previous and len(used) >= min_primes
        previous = candidate
        if stable:
            # Modular rank is a lower bound; n-r independent exact kernel vectors
            # give the matching upper bound and certify the full finite kernel.
            metadata['exact_training_checks'] += int(bool(candidate))
            if _annihilates_matrix(matrix, candidate):
                return _exact_result(problem, candidate, len(profile), rows, metadata,
                                     training_verified=True)
    metadata.update(exact_training=False, exact_holdout=None, full_training_kernel=False,
                    all_indices_proved=False, rank_lower_bound=0 if profile is None else len(profile))
    return ModuleResult((), problem.labels, 'reconstruction_exhausted', metadata)


guess_module = solve_module
