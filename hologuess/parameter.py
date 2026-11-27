"""qhsolve's adaptive interpolation applied to exact module-observation columns."""
import numpy as np

from . import _qhsolve as core
from .interpolation import ActionOracle, interpolate_relation


def _modular_polynomials(polynomials, p):
    """Reduce rational scalar coefficients once per prime, without rounding."""
    inverses = {1: 1}
    result = []
    for polynomial in polynomials:
        converted = {}
        for exponent, numerator, denominator in polynomial:
            if denominator % p == 0:
                raise ZeroDivisionError('A rational coefficient denominator vanishes modulo this prime.')
            if denominator not in inverses:
                inverses[denominator] = pow(denominator % p, -1, p)
            converted[exponent] = numerator % p * inverses[denominator] % p
        result.append(converted)
    return result


class _RationalOracle(ActionOracle):
    def at(self, p, count):
        if self._prime != p:
            modular = _modular_polynomials(self.polynomials, p)
            poles = _modular_polynomials(self.pole_polynomials, p)
            denominators = _modular_polynomials(self.denominator_polynomials, p)
            self._modular, self._poles, self._denominators = modular, poles, denominators
            self._prime = p
        # Row denominators are shared by all action columns. Only distinct,
        # nonconstant denominators have parameter poles to exclude.
        selected, candidate = [], 2
        while len(selected) < count:
            if candidate >= p:
                raise ValueError('Parameter poles exhaust the available nonzero field points.')
            width = min(max(32, count - len(selected)), p - candidate)
            points = np.arange(candidate, candidate + width, dtype=np.int64)
            good = np.ones(width, dtype=bool)
            if self._poles:
                good &= np.all(core.eval_laurent(self._poles, points, p) != 0, axis=0)
            selected.extend(map(int, points[good]))
            candidate += width
        points = np.asarray(selected[:count], dtype=np.int64)
        evaluated = core.eval_laurent(self._modular, points, p)
        if self._denominators:
            denominator_values = core.eval_laurent(self._denominators, points, p)
            inverses = core.inv_mod(denominator_values, p)
            active = self.denominator_indices >= 0
            evaluated[active] = evaluated[active] * inverses[self.denominator_indices[active]] % p
        values = evaluated.reshape(len(self.labels), self.observations, count).transpose(1, 0, 2)
        return points, values


def interpolate_problem(problem, parameter, **options):
    """Reconstruct a line relation over QQ(parameter) from exact observations.

    This uses the preserved qhsolve reference selection, denominator clearing,
    adaptive Laurent bands, multi-prime CRT, Wang reconstruction and kernel
    caching. Parameter points at poles of any input observation are skipped.
    A line kernel is required; general higher-dimensional spaces use
    solve_module with explicitly bounded parameter monomials.

    The result coefficients[(column, exponent)] describe integer Laurent
    coefficients in `parameter`. They are checked against EVERY original
    observation, including the held-out suffix; no all-index claim is made.
    """
    from sage.all import QQ, PolynomialRing
    from .module import ModuleProblem
    if not isinstance(problem, ModuleProblem):
        problem = ModuleProblem(problem)
    name = str(parameter)
    if not name.isidentifier():
        raise ValueError('parameter must be a single indeterminate')
    ring = PolynomialRing(QQ, name); field = ring.fraction_field()
    exact, numerators, denominators = [], [], []
    one = ring.one()
    for column in problem.columns:
        converted, nums, dens = [], [], []
        for value in column:
            if isinstance(value, (float, complex)) or (hasattr(value, 'parent') and not value.parent().is_exact()):
                raise TypeError('Parameter observations must be exact rational functions.')
            if hasattr(value, 'parent') and value.parent() is ring:
                # Preserve native polynomial inputs; a fraction-field roundtrip
                # would normalize a denominator known to be one for every cell.
                converted.append(value); nums.append(value); dens.append(one)
            else:
                rational = field(value)
                converted.append(rational)
                num, den = rational.numerator(), rational.denominator()
                if den.degree() <= 0 and den != one:
                    num, den = ring(num / den[0]), one
                nums.append(num); dens.append(den)
        exact.append(tuple(converted)); numerators.append(nums); denominators.append(dens)
    if not any(value for column in exact for value in column):
        raise ValueError('All-zero observations do not specify a unique line relation.')
    # A common row denominator is a unit row operation over QQ(parameter).
    # Clear it only when already shared: heterogeneous denominators stay as
    # rational entries rather than creating a high-degree product/LCM.
    row_count = len(exact[0])
    observations = [[] for _ in exact]
    polynomial_row = []
    poles = set()
    for i in range(row_count):
        row_denominators = {column[i] for column in denominators if column[i].degree() > 0}
        poles.update(row_denominators)
        clear_row = len(row_denominators) <= 1
        polynomial_row.append(clear_row)
        denominator = next(iter(row_denominators), one)
        for j, column in enumerate(exact):
            if clear_row:
                num, den = numerators[j][i], denominators[j][i]
                value = num if den == denominator else num * denominator
            else:
                value = column[i]
            observations[j].append(value)

    def encode(poly):
        return tuple((int(e), int(c.numerator()), int(c.denominator()))
                     for e, c in poly.dict().items())

    integral = all(column[i].denominator() == 1 for i, is_poly in enumerate(polynomial_row) if is_poly
                   for column in observations)
    from sage.all import ZZ
    validation_ring = PolynomialRing(ZZ if integral else QQ, name)
    for j, column in enumerate(observations):
        for i, is_poly in enumerate(polynomial_row):
            if is_poly: column[i] = validation_ring(column[i])
    v = validation_ring.gen()

    def validate(coefficients):
        if not coefficients:
            return False
        # Clear the common Laurent unit once, then use native polynomial
        # products on polynomial rows and rational products only where needed.
        offset = max(0, -min(e for j, e in coefficients))
        weights = [validation_ring.zero() for _ in exact]
        for (j, e), coefficient in coefficients.items():
            weights[j] += coefficient * v**(e + offset)
        live = [j for j, weight in enumerate(weights) if weight]
        if not live: return False
        field_weights = [field(weight) for weight in weights] if not all(polynomial_row) else None
        for i, is_poly in enumerate(polynomial_row):
            used = weights if is_poly else field_weights
            zero = validation_ring.zero() if is_poly else field.zero()
            if sum((used[j] * observations[j][i] for j in live), zero) != 0:
                return False
        return True

    # Fit on the requested training prefix. The exact validator above retains
    # every original observation, including the untouched held-out suffix.
    oracle = _RationalOracle(evaluate=None, labels=tuple(problem.labels),
        observations=problem.training_rows, exact_validate=validate)
    oracle.polynomials = []
    denominator_polynomials, denominator_lookup, denominator_indices = [], {}, []
    for j, column in enumerate(observations):
        for i, value in enumerate(column[:problem.training_rows]):
            if polynomial_row[i]:
                numerator, denominator = value, one
            else:
                numerator, denominator = numerators[j][i], denominators[j][i]
            if denominator.degree() <= 0:
                # Fraction fields normally make denominators monic, but fold
                # any remaining scalar denominator into exact coefficients.
                if denominator != 1:
                    numerator = ring(numerator / denominator[0])
                denominator_indices.append(-1)
            else:
                if denominator not in denominator_lookup:
                    denominator_lookup[denominator] = len(denominator_polynomials)
                    denominator_polynomials.append(encode(denominator))
                denominator_indices.append(denominator_lookup[denominator])
            oracle.polynomials.append(encode(numerator))
    oracle.denominator_polynomials = denominator_polynomials
    oracle.denominator_indices = np.asarray(denominator_indices, dtype=np.int64)
    oracle.pole_polynomials = [encode(d) for d in sorted(poles, key=str)]
    oracle._prime = None
    result = interpolate_relation(oracle, **options)
    result.metadata.update(parameter=name, training_observations=problem.training_rows,
        holdout_observations=len(exact[0])-problem.training_rows,
        exact_all_observations=True, input_poles_skipped=True,
        backend='qhsolve_module_interpolation', row_denominators_cleared=all(polynomial_row),
        polynomial_observation_rows=sum(polynomial_row),
        validation_ring=str(validation_ring.base_ring()),
        validation_mode='native_polynomial' if all(polynomial_row) else 'mixed_polynomial_rational',
        rational_validation_field=None if all(polynomial_row) else str(field),
        all_indices_proved=False)
    return result
