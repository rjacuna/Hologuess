"""Finite free Ore-module actions from AJ Appendix B."""
from dataclasses import dataclass


@dataclass(frozen=True)
class OreAction:
    """D(v) = B sigma(v) + delta(v), with maps applied entrywise.

    The caller supplies valid coefficient maps and compatible commuting module
    actions. The matrix B records images of basis vectors, not the full action
    on arbitrary coefficient columns. In particular D**2 is not B**2.
    """
    matrix: object
    sigma: object = lambda a: a
    delta: object = lambda a: 0

    def __call__(self, value):
        from sage.all import vector
        if len(value) != self.matrix.ncols():
            raise ValueError('Module coordinate dimension does not match the action matrix.')
        ring = self.matrix.base_ring()
        return self.matrix * vector(ring, [self.sigma(a) for a in value]) + vector(ring, [self.delta(a) for a in value])


def act(actions, powers, value):
    """Apply D_1**a_1 ... D_r**a_r in literal composition order."""
    if len(actions) != len(powers) or any(int(p) != p or p < 0 for p in powers):
        raise ValueError('Need one nonnegative integer power per action.')
    out = value
    for action, power in reversed(tuple(zip(actions, powers))):
        for _ in range(int(power)): out = action(out)
    return out


def module_problem(actions, value, support, *, observe=None, name='Ore module'):
    """Construct exact observation columns from coordinate module actions.

    support is a list of (coefficient, powers) pairs. Coefficients act on the
    left after the action word. observe maps a vector to scalar observations;
    by default all coordinates are used. Bounds/coefficients are caller chosen.
    """
    from .module import ModuleProblem
    support = tuple(support)
    if not support: raise ValueError('Action support must be nonempty.')
    observe = tuple if observe is None else observe
    columns = [tuple(observe(coefficient * act(actions, powers, value)))
               for coefficient, powers in support]
    return ModuleProblem(columns, labels=support, name=name)
