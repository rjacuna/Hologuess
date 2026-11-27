"""Sage LaTeX for the exact normal form, as in JonesRec's Operator.pretty."""

from collections import Counter


_latex_symbols = {}


def set_latex_symbols(**symbols):
    r"""Set the notebook's LaTeX symbols. Call with no arguments to reset.

    Use ``S`` for shifts, ``D`` for derivatives, ``E`` for q-shifts, or an
    exact generator name such as ``Sn`` or ``Dy``. Values are LaTeX fragments.
    Each call replaces the previous settings. Plain Ore text is unaffected.
    """
    global _latex_symbols
    _check_symbols(symbols)
    _latex_symbols = symbols


def _check_symbols(symbols):
    if any(not isinstance(symbol, str) or not symbol.strip() for symbol in symbols.values()):
        raise ValueError('Each operator symbol must be a nonempty LaTeX string.')


def _action_names(algebra):
    actions = []
    for i in range(algebra.ngens()):
        for detector, symbol in (('is_S', 'S'), ('is_D', 'D'), ('is_Q', 'E')):
            action = getattr(algebra, detector)(i)
            if action is not False:
                axis = action[0] if detector == 'is_Q' else action
                actions.append((symbol, axis))
                break
        else:
            actions.append((None, None))
    return actions


def _group(symbol):
    # Bind powers and subscripts to the entire TeX fragment.
    return symbol if len(symbol) == 1 else '{' + symbol + '}'


def _operator_latex(polynomial, algebra, symbols):
    from sage.misc.latex import latex

    ring = polynomial.parent()
    native = tuple(ring.latex_variable_names())
    actions = _action_names(algebra)
    repeated = Counter(symbol for symbol, _ in actions)
    names = []
    for name, original, (symbol, axis) in zip(algebra.variable_names(), native, actions):
        if name in symbols:
            display = _group(symbols[name])
        elif symbol is None:
            display = original
        else:
            display = _group(symbols.get(symbol, symbol))
            if repeated[symbol] > 1:
                display += '_{' + latex(axis) + '}'
        names.append(display)

    if tuple(names) == native:
        return latex(polynomial)
    if algebra.ngens() == 1:
        return polynomial._latex_(name=names[0])

    # Feed display names to the same structural formatter Sage uses for
    # multivariate polynomials. Coefficients keep their own Sage renderer.
    from sage.rings.polynomial.polydict import PolyDict
    return PolyDict(polynomial.dict()).latex(
        names, atomic_coefficients=ring.base_ring()._repr_option('element_is_atomic'),
        sortkey=ring.term_order().sortkey,
    )


def operator_latex(operator, **symbols):
    r"""Return copyable LaTeX for an Ore operator, without math delimiters.

    Keyword symbols override the current settings for this call only.
    ``operator_latex(P, D=r'\partial_x')`` changes the derivative symbol.
    The operator's coefficients, parent and arithmetic are unaffected.
    """
    _check_symbols(symbols)
    return _operator_latex(operator.polynomial(), operator.parent(), _latex_symbols | symbols)


class _OperatorLatex:
    """Picklable per-result Sage hook. It holds no reference to the operator."""

    __slots__ = ('polynomial', 'algebra')

    def __init__(self, polynomial, algebra):
        self.polynomial = polynomial
        self.algebra = algebra

    def __call__(self, name=None):
        if name is not None and self.algebra.ngens() == 1:
            return self.polynomial._latex_(name=name)
        return _operator_latex(self.polynomial, self.algebra, _latex_symbols)


def _with_latex(operator):
    """Return a native Ore operator with Hologuess's Sage LaTeX hook."""
    polynomial, algebra = operator.polynomial(), operator.parent()
    # A fresh element keeps even cached generators and units unmodified.
    displayed = type(operator)(algebra, polynomial)
    displayed._latex_ = _OperatorLatex(displayed.polynomial(), algebra)
    return displayed
