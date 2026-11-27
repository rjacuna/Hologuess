"""Ore-compatible finite-data adapters, implemented from coefficient actions.

No ore_algebra guessing routine is called. Sequence inputs are zero indexed;
function inputs are ordinary Taylor coefficients. E is the forward shift and
Q denotes multiplication by q to the index in Hologuess's own notation. Names
of generators supplied by an existing OreAlgebra are retained in its output.
"""
from dataclasses import dataclass
from itertools import product
from math import comb, prod

from ._display import _with_latex


def _rectangular(data):
    """Return a rectangular tensor's shape and its leaves indexed by tuples."""
    if not isinstance(data, (list, tuple)) or not data:
        raise ValueError('data must be a nonempty list or rectangular nested list')
    def walk(value):
        if not isinstance(value, (list, tuple)):
            return (), {(): value}
        if not value:
            raise ValueError('data axes must be nonempty')
        parts = [walk(item) for item in value]
        shape = parts[0][0]
        if any(s != shape for s, _ in parts):
            raise ValueError('data must be rectangular, with no missing entries')
        return (len(value),) + shape, {(j,) + k: v for j, (_, d) in enumerate(parts) for k, v in d.items()}
    return walk(data)


def _require_exact(value):
    """Reject approximation before a coefficient-ring coercion can hide it."""
    from decimal import Decimal
    if isinstance(value, (float, complex, Decimal)) or getattr(getattr(value, 'dtype', None), 'kind', '') in ('f', 'c'):
        raise TypeError('guessing data must be exact; approximate inputs are not rationalized')
    parent = getattr(value, 'parent', None)
    if callable(parent):
        exact = getattr(parent(), 'is_exact', None)
        if callable(exact) and not exact():
            # The Symbolic Ring can contain both exact rationals and floats.
            # KJJ's public examples explicitly use exact symbolic rationals.
            try:
                element_exact = value.is_exact() is True
            except (AttributeError, TypeError, ValueError):
                element_exact = False
            if not element_exact:
                raise TypeError('guessing data must be exact')
    if getattr(value, 'is_Float', False):
        raise TypeError('guessing data must be exact')
    return value


def _natural(value, name):
    if isinstance(value, bool) or int(value) != value or value < 0:
        raise ValueError(name + ' must be a nonnegative integer')
    return int(value)


def _orders(order, count):
    if isinstance(order, (list, tuple)):
        if len(order) != count:
            raise ValueError('one order bound is required per generator')
        return tuple(_natural(r, 'order') for r in order)
    return (_natural(order, 'order'),) * count


def _kind(algebra, i, variables):
    for letter in ('D', 'S', 'F', 'Q'):
        found = getattr(algebra, 'is_' + letter)(i)
        if found is not False:
            variable, q = found if letter == 'Q' else (found, None)
            return letter, variables.index(variable), q
    if algebra.is_C(i):
        if len(variables) != 1 or algebra.ngens() != 1:
            raise NotImplementedError('algebraic power-series guessing uses one commutative generator')
        return 'C', 0, None
    raise NotImplementedError('custom actions require ModuleProblem or a module-action oracle')


def _parameters(K):
    """Independent polynomial parameters in a polynomial/fraction-field tower."""
    found = []
    current = K
    while True:
        try:
            if current.is_prime_field() or current.is_finite():
                break
        except (AttributeError, NotImplementedError):
            pass
        try:
            # Algebraic generators are not independent polynomial parameters.
            if current.is_number_field():
                break
        except (AttributeError, NotImplementedError):
            pass
        try:
            names, generators = current.variable_names(), current.gens()
        except AttributeError:
            break
        for name, generator in zip(names, generators):
            value = K(generator)
            if str(value) not in {str(v) for v in found}:
                found.append(value)
        try:
            base = current.base_ring()
        except AttributeError:
            break
        if base is current:
            break
        current = base
    return tuple(found)


def _filter_term(predicate, exponents, operator):
    if predicate is None:
        return True
    try:
        answer = predicate(exponents)
    except (TypeError, AttributeError):
        return bool(predicate(operator))
    return bool(answer)


def _modular_input_builder(data, shape, kinds, starts, operator_powers,
                           coefficient_powers, points):
    """Build QQ action images from reduced inputs, preserving column order.

    The callback is private: its caller binds it to the unchanged exact columns
    and training-row count. A pole in a raw input can cancel under an action or
    belong only to an unused observation, so it requests the exact-matrix
    fallback instead of declaring the prime unusable.
    """
    def build(p):
        input_cells = prod(shape)
        state_count = prod(max(v[i] for v in operator_powers)+1 for i in range(len(kinds)))
        if kinds[0][0] == 'C':
            state_count = max(2, state_count)  # Both the unit and input are seeded.
        observation_cells = len(operator_powers)*len(coefficient_powers)*len(points)
        if state_count*input_cells > 4*(input_cells+observation_cells):
            raise NotImplementedError('Modular action state bound exceeds the observation-relative budget.')
        import numpy as np
        from sage.all import GF
        F = GF(p)
        try:
            reduced = np.asarray([int(F(c)) for c in data], dtype=np.int64).reshape(shape)
            q_values = {
                i: int(F(q)) for i, (kind, axis, q) in enumerate(kinds)
                if kind == 'Q' and any(u[axis] for u in coefficient_powers)
            }
        except (ZeroDivisionError, ValueError) as exc:
            raise NotImplementedError('A raw modular input pole needs exact-matrix reduction.') from exc
        zero_order = (0,) * len(kinds)
        images = {zero_order: reduced}
        if kinds[0][0] == 'C':
            unit = np.zeros(shape, dtype=np.int64)
            unit[0] = 1
            images = {zero_order: unit, (1,): reduced}

        def action(v):
            if v in images:
                return images[v]
            i = next(k for k, exponent in enumerate(v) if exponent)
            kind, axis, _ = kinds[i]
            previous_order = list(v)
            previous_order[i] -= 1
            previous = action(tuple(previous_order))
            if kind == 'C':
                # Python integer accumulation avoids int64 convolution overflow.
                image = np.asarray([
                    sum(int(previous[j]) * int(reduced[n-j]) for j in range(n+1)) % p
                    for n in range(shape[0])], dtype=np.int64)
            else:
                tail = [slice(None)] * len(shape)
                tail[axis] = slice(1, None)
                image = previous[tuple(tail)]
                if kind == 'D':
                    weights = np.arange(1, previous.shape[axis], dtype=np.int64) % p
                    axes = [1] * len(shape)
                    axes[axis] = len(weights)
                    image = image * weights.reshape(axes) % p
                elif kind == 'F':
                    head = [slice(None)] * len(shape)
                    head[axis] = slice(None, -1)
                    image = (image - previous[tuple(head)]) % p
            images[v] = image
            return image

        indices = tuple(np.fromiter((n[axis] for n in points), dtype=np.int64)
                        for axis in range(len(shape)))
        factors = {}
        for i, (kind, axis, _) in enumerate(kinds):
            if kind in ('D', 'C'):
                continue
            for exponent in {u[axis] for u in coefficient_powers}:
                if kind == 'Q':
                    values = [pow(q_values[i], (n+starts[axis])*exponent, p)
                              for n in range(shape[axis])] if exponent else [1]*shape[axis]
                else:
                    values = [pow((n+starts[axis]) % p, exponent, p)
                              for n in range(shape[axis])]
                factors[axis, exponent] = np.asarray(values, dtype=np.int64)
        columns = []
        for v in operator_powers:
            transformed = action(v)
            for u in coefficient_powers:
                at = []
                valid = np.ones(len(points), dtype=bool)
                for i, (kind, axis, _) in enumerate(kinds):
                    index = indices[axis] - u[axis] if kind in ('D', 'C') else indices[axis]
                    valid &= index >= 0
                    at.append(np.maximum(index, 0))
                # Kinds and axes can be permuted relative to algebra generators.
                by_axis = [None] * len(shape)
                for index, (_, axis, _) in zip(at, kinds):
                    by_axis[axis] = index
                column = transformed[tuple(by_axis)]
                column = np.where(valid, column, 0)
                for kind, axis, _ in kinds:
                    if kind not in ('D', 'C'):
                        column = column * factors[axis, u[axis]][indices[axis]] % p
                columns.append(column)
        return np.column_stack(columns)
    return build


def prepare(data, algebra, *, order=4, degree=4, training_rows=None,
            holdout=0, point_filter=None, term_filter=None, start=0, name='Ore data',
            terms=None, parameter_degree=None):
    """Build the exact bounded action matrix for a supported Ore algebra.

    Supports ordinary and q shifts, forward differences, Taylor-series
    derivatives, and univariate algebraic equations. Mixed multivariate
    differential/shift/q-shift inputs are rectangular nested coefficient lists.
    `degree` bounds total coefficient degree; `order` can be per generator.
    `parameter_degree` bounds total polynomial degree in constant-field parameters
    (default: degree); rational operator coefficients are represented after
    clearing a common denominator. No source values beyond the supplied tensor are assumed to be zero.
    A callable univariate sequence may be sampled with `terms=N`.
    """
    from .module import ModuleProblem
    R = algebra.base_ring()
    variables = list(R.gens())
    K = R.base_ring()
    # Observations may be rational functions even when the requested operator
    # coefficients are polynomial. Localize only the observation domain; all
    # parameter monomials and output labels still belong to the caller's parent.
    observation_field = K if K.is_field() else K.fraction_field()
    count = algebra.ngens()
    # The installed upstream multivariate constructor requires a field of
    # coefficients for its Singular representation. Localize an equivalent
    # parent per request; never patch the caller's algebra or package classes.
    output_algebra = algebra.change_ring(R.fraction_field()) if count > 1 else algebra
    if count > 1:
        from sage.all import PolynomialRing
        # Upstream 0.5 hardcodes one variable when forcing Singular's
        # representation. Seed the standard cache on this localized parent
        # with the correctly sized polynomial ring; no class is monkey-patched.
        output_algebra._commutative_ring = PolynomialRing(output_algebra.base_ring(), names=output_algebra.variable_names())
    bounds = _orders(order, count)
    degree = _natural(degree, 'degree')
    holdout = _natural(holdout, 'holdout')
    parameter_degree = degree if parameter_degree is None else _natural(parameter_degree, 'parameter_degree')
    parameters = _parameters(K)
    parameter_monomials = [prod((x**e for x, e in zip(parameters, exponents)), start=K.one())
                          for exponents in product(range(parameter_degree + 1), repeat=len(parameters))
                          if sum(exponents) <= parameter_degree]
    if callable(data):
        if terms is None or count != 1:
            raise ValueError('a callable sequence requires terms=N and one generator')
        data = [data(i + int(start)) for i in range(_natural(terms, 'terms'))]
    elif isinstance(data, str):
        with open(data, encoding='utf-8') as source:
            data = [observation_field(line.strip()) for line in source if line.strip()]
    shape, leaves = _rectangular(data)
    if len(shape) != len(variables) or count != len(variables):
        raise ValueError('data axes, coefficient variables, and generators must correspond one to one')
    kinds = [_kind(algebra, i, variables) for i in range(count)]
    if len({axis for _, axis, _ in kinds}) != count:
        raise ValueError('each standard generator must act on a distinct data axis')
    if isinstance(start, (tuple, list)):
        if len(start) != count:
            raise ValueError('one starting index is required per data axis')
        starts = tuple(_natural(s, 'start') for s in start)
    else:
        starts = (_natural(start, 'start'),) * count
    if any(k in ('D', 'C') and starts[axis] for k, axis, _ in kinds):
        raise ValueError('Taylor coefficient arrays start at degree zero')
    data_map = {key: observation_field(_require_exact(value)) for key, value in leaves.items()}
    coeff_powers = [u for u in product(range(degree + 1), repeat=count) if sum(u) <= degree]
    operator_powers = []
    for v in product(*(range(r + 1) for r in bounds)):
        op = prod((output_algebra.gen(i) ** e for i, e in enumerate(v)), start=output_algebra.one())
        if _filter_term(term_filter, v, op):
            operator_powers.append((v, op))
    max_by_axis = [0] * count
    for i, (kind, axis, _) in enumerate(kinds):
        if kind != 'C':
            max_by_axis[axis] = bounds[i]
    ranges = [range(n - r) for n, r in zip(shape, max_by_axis)]
    if any(n <= r for n, r in zip(shape, max_by_axis)):
        raise ValueError('not enough data for the requested order')
    points = [n for n in product(*ranges) if point_filter is None or point_filter(*(n[j] + starts[j] for j in range(count)))]
    if not points:
        raise ValueError('point filter leaves no observations')
    series_powers = None
    if kinds[0][0] == 'C':
        length = shape[0]
        series = [data_map[(i,)] for i in range(length)]
        series_powers = [[K.one()] + [K.zero()] * (length - 1)]
        for _ in range(bounds[0]):
            previous = series_powers[-1]
            series_powers.append([sum((previous[j] * series[n-j] for j in range(n+1)), K.zero()) for n in range(length)])

    def observation(n, u, v):
        if series_powers is not None:
            k = n[0] - u[0]
            return K.zero() if k < 0 else series_powers[v[0]][k]
        target = list(n)
        factor = K.one()
        differences = []
        for i, (kind, axis, q) in enumerate(kinds):
            a, b = u[axis], v[i]
            nv = n[axis] + starts[axis]
            if kind == 'D':
                base = n[axis] - a
                if base < 0:
                    return K.zero()
                target[axis] = base + b
                factor *= prod(range(base + 1, base + b + 1))
            else:
                factor *= K(q) ** (nv * a) if kind == 'Q' else K(nv) ** a
                if kind == 'F':
                    differences.append((axis, b))
                else:
                    target[axis] += b
        if not differences:
            return factor * data_map[tuple(target)]
        total = K.zero()
        for shifts in product(*(range(b + 1) for _, b in differences)):
            at = target.copy(); weight = 1
            for (axis, b), k in zip(differences, shifts):
                at[axis] += k
                weight *= (-1) ** (b-k) * comb(b, k)
            total += weight * data_map[tuple(at)]
        return factor * total

    labels, columns = [], []
    for v, op in operator_powers:
        for u in coeff_powers:
            label = prod((x ** e for x, e in zip(variables, u)), start=R.one()) * op
            values = [observation(n, u, v) for n in points]
            for monomial in parameter_monomials:
                labels.append(monomial * label)
                columns.append([monomial * value for value in values])
    if not columns:
        raise ValueError('term filter leaves no candidate terms')
    if training_rows is not None and holdout:
        raise ValueError('choose training_rows or holdout, not both')
    if training_rows is None and holdout:
        if holdout >= len(points):
            raise ValueError('holdout consumes every observation')
        training_rows = len(points) - holdout
    problem = ModuleProblem(columns, labels=labels, training_rows=training_rows, name=name)
    from sage.all import QQ
    if observation_field == QQ and not parameters:
        problem._modular_matrix = _modular_input_builder(
            tuple(data_map[key] for key in product(*(range(n) for n in shape))),
            shape, tuple(kinds), starts, tuple(v for v, _ in operator_powers),
            tuple(coeff_powers), tuple(points[:problem.training_rows]))
        problem._modular_columns = problem.columns
        problem._modular_training_rows = problem.training_rows
    problem.adapter_metadata = dict(adapter='Ore coefficient actions',
        coefficient_degree_bound=degree, operator_order_bound=bounds,
        parameter_degree_bound=parameter_degree, parameter_names=tuple(map(str, parameters)),
        observation_coefficient_domain=str(observation_field),
        sequence_start=starts, source_shape=shape, observation_points=tuple(points),
        action_terms=tuple(map(str, labels)), point_filter_applied=point_filter is not None,
        term_filter_applied=term_filter is not None, output_coefficients_localized=count > 1,
        all_indices_proved=False)
    return problem


@dataclass
class OreGuessResult:
    """Bounded relation basis with explicit finite-data evidence."""
    problem: object
    module_result: object
    operators: tuple

    @property
    def metadata(self):
        return self.module_result.metadata

    @property
    def status(self):
        return self.module_result.status

    @property
    def operator(self):
        if not self.operators:
            raise ValueError('no relation was found in the requested bounds')
        first = self.operators[0]
        if any(op.leading_coefficient() * first != first.leading_coefficient() * op for op in self.operators[1:]):
            raise ValueError('this bounded space has several independent operators; use .operators')
        return first


def _try_qfast(data, algebra, order, degree, options, workers, verbose, store, primes, max_points,
               kernel_backend, flint_library):
    """Keep the mature qhsolve reconstruction for its native QQ(q) case."""
    if algebra.ngens() != 1 or algebra.is_Q() is False or degree == 0 or options.get('parameter_degree') is not None:
        return None
    if options.get('holdout', 0) or options.get('training_rows') is not None or options.get('point_filter') or options.get('term_filter'):
        return None
    R = algebra.base_ring(); K = R.base_ring()
    parameters = _parameters(K)
    q = algebra.is_Q()[1]
    if len(parameters) != 1 or q != parameters[0] or len(R.gens()) != 1:
        return None
    from sage.all import QQ, LaurentPolynomialRing
    from .qsequence import guess_qsequence
    L = LaurentPolynomialRing(QQ, str(q))
    if callable(data):
        if options.get('terms') is None:
            return None
        start = int(options.get('start', 0))
        data = [data(i + start) for i in range(int(options['terms']))]
    if isinstance(data, str):
        with open(data, encoding='utf-8') as source:
            data = [K(line.strip()) for line in source if line.strip()]
    try:
        values = [L(_require_exact(value)).dict() for value in data]
    except (ValueError, TypeError):
        return None  # rational-function data use the bounded module adapter
    result = guess_qsequence(values, start=int(options.get('start', 0)),
        max_order=int(order), max_degree=int(degree), workers=workers,
        verbose=verbose, store=store, primes=primes, max_points=max_points,
        kernel_backend=kernel_backend, flint_library=flint_library)
    if result is None:
        return None
    Q, E = R.gen(), algebra.gen()
    op = sum((algebra(c * q**e * Q**j) * E**b for (b, j, e), c in result.coefficients.items()), algebra.zero())
    result.metadata['backend'] = 'qhsolve'
    # QGuess exposes metadata; use the same evidence protocol as module results.
    from types import SimpleNamespace
    evidence = SimpleNamespace(metadata=result.metadata, status='guessed')
    return OreGuessResult(None, evidence, (_with_latex(op),))


def guess(data, algebra, *, order=4, degree=4, path=None, min_order=1,
          min_degree=0, backend='auto', workers=1, verbose=False, store=None,
          primes=None, max_points=2400, kernel_backend='auto', flint_library=None,
          **kwargs):
    """Guess using Hologuess modular reconstruction, never ore_algebra.guess.

    Bounds are finite and are not overruled. A multivariate request solves
    its full specified box; a univariate request sweeps ascending order then
    degree. The returned operators only claim finite-data validity.
    kernel_backend selects modular elimination, preferring an installed FLINT bridge.
    """
    from .module import solve_module
    if backend not in ('auto', 'module', 'qhsolve'):
        raise ValueError('backend must be auto, module, or qhsolve')
    if backend != 'module' and path is None and not isinstance(order, (tuple, list)) and min_order == 1 and min_degree == 0:
        result = _try_qfast(data, algebra, order, degree, kwargs, workers, verbose,
                            store, primes, max_points, kernel_backend, flint_library)
        if result is not None:
            return result
        if backend == 'qhsolve':
            raise ValueError('qfast requires Laurent-polynomial QQ(q) data and a recoverable line corner')
    elif backend == 'qhsolve':
        raise ValueError('Explicit qhsolve backend does not support a path, tuple order or nondefault lower bounds.')
    if path is None:
        if algebra.ngens() > 1:
            path = [(order, degree)]
        else:
            maximum = _orders(order, 1)[0]
            path = [(r, d) for r in range(_natural(min_order, 'min_order'), maximum + 1)
                    for d in range(_natural(min_degree, 'min_degree'), _natural(degree, 'degree') + 1)]
    last = None
    attempted = []
    unresolved = []
    for r, d in path:
        problem = prepare(data, algebra, order=r, degree=d, **kwargs)
        solved = solve_module(problem, primes=primes, store=store,
                              kernel_backend=kernel_backend, flint_library=flint_library)
        attempted.append({'order': r, 'degree': d, 'status': solved.status})
        if solved.status == 'reconstruction_exhausted':
            unresolved.append({'order': r, 'degree': d})
        solved.metadata.update(problem.adapter_metadata)
        solved.metadata.update(adapter='Ore coefficient actions', search_path=attempted.copy(),
            unresolved_boxes=unresolved.copy(),
            all_indices_proved=False)
        basis = solved.valid_basis
        output_parent = problem.labels[0].parent()
        operators = []
        for row in basis:
            # A rational normalized kernel vector can still represent an
            # integral operator. Clear its scalar denominator before coercion
            # instead of letting arithmetic silently enlarge the output parent.
            from sage.all import lcm
            denominator = lcm([c.denominator() if hasattr(c, 'denominator') else 1
                               for c in row]) if not algebra.base_ring().base_ring().is_field() else 1
            operators.append(sum((output_parent(denominator*coefficient)*term
                                  for coefficient, term in zip(row, problem.labels)), output_parent.zero()))
        operators = tuple(_with_latex(operator) for operator in operators)
        last = OreGuessResult(problem, solved, operators)
        if operators:
            return last
    if last is None:
        raise ValueError('search path is empty')
    return last
