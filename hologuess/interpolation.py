"""Univariate parameter reconstruction for arbitrary observable module actions."""
from dataclasses import dataclass
import numpy as np

from . import _qhsolve as core
from .qsequence import _modular_array
from ._primes import reconstruction_primes


@dataclass
class ActionOracle:
    """A batch evaluator for an explicitly chosen module-action support.

    evaluate(p, points) returns (observations, actions, points) modulo p.
    The caller chooses observations (Taylor coefficients, sequence indices,
    evaluation functionals, vector coordinates, etc.). No commutation or
    principal-ideal assumptions are imposed on these actions.
    """
    evaluate: object
    labels: tuple
    observations: int
    exact_validate: object = None

    def at(self, p, count):
        if count + 2 >= p: raise ValueError('Parameter point request exhausts the field.')
        points = np.arange(2, count + 2, dtype=np.int64)
        values = _modular_array(self.evaluate(p, points), p)
        if values.shape != (self.observations, len(self.labels), count):
            raise ValueError('Action oracle must return (observations, actions, points).')
        return points, values % p


@dataclass
class InterpolatedRelation:
    coefficients: dict
    labels: tuple
    metadata: dict


def interpolate_relation(oracle, *, band=None, primes=None, workers=1,
                         max_points=2400, store=None, verbose=False,
                         kernel_backend='auto', flint_library=None):
    """Recover a one-dimensional relation over QQ(parameter).

    For higher-dimensional relation spaces use solve_module, or specify a
    support/normalization making the desired relation unique. The full
    qhsolve reference, denominator, band, CRT, cache and pool machinery is used.
    kernel_backend selects modular elimination, preferring an installed FLINT bridge.
    """
    from ._kernels import kernel_scope
    with kernel_scope(kernel_backend, flint_library) as kernel:
        result = _interpolate(oracle, band, primes, workers, max_points, store, verbose)
    result.metadata['kernel_backend'] = 'numpy' if kernel is None else 'flint'
    return result


def _interpolate(oracle, band, primes, workers, max_points, store, verbose):
    if not oracle.labels or oracle.observations < 1 or workers < 1:
        raise ValueError('Need nonempty actions, observations and positive workers.')
    cells = [(i, 0) for i in range(len(oracle.labels))]
    p = reconstruction_primes(primes)
    points, values = oracle.at(p[0], 4)
    probes = core.point_matrices(values, cells, points, p[0], oracle.observations)
    reductions = [core.rref_mod(m, p[0]) for m in probes]
    dimension = min(len(cells) - len(pivots) for reduced, pivots in reductions)
    if dimension != 1: raise core.KernelNotALine(dimension)
    # Preserve qhsolve's support restriction before expensive reconstruction.
    # Exceptional higher-nullity probes cannot contribute an arbitrary vector.
    # Cell identities stay in the caller's original coordinate system, so the
    # tensor builder selects the retained columns and exact validation still
    # refers to every original action.
    live = np.zeros(len(cells), dtype=bool)
    for reduced, pivots in reductions:
        if len(cells) - len(pivots) == 1:
            pivot_set = set(pivots)
            free = next(i for i in range(len(cells)) if i not in pivot_set)
            live[free] = True
            for row, pivot in enumerate(pivots):
                live[pivot] |= reduced[row, free] % p[0] != 0
    support = tuple(int(i) for i in np.flatnonzero(live))
    cells = [(i, 0) for i in support]
    u, info = core.extract_one(oracle.at, cells, p, band, workers, verbose,
        nrows=oracle.observations, ref_mode='measured', max_points=max_points, store=store)
    coefficients = {(i, e): c for (i, _, e), c in u.items()}
    info.update(status='guessed', all_indices_proved=False, exact_validation=None,
                original_actions=len(oracle.labels), retained_actions=len(support),
                support_indices=support)
    if oracle.exact_validate is not None:
        if not oracle.exact_validate(coefficients):
            raise ArithmeticError('Interpolated relation failed the supplied exact validator.')
        info['exact_validation'] = True
    return InterpolatedRelation(coefficients, tuple(oracle.labels), info)
