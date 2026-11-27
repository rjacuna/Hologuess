"""Validate caller primes without re-proving the built-in verified prime pool."""
from . import _qhsolve as core

_TRUSTED = frozenset(core.PRIMES)


def reconstruction_primes(primes=None):
    values = list(core.PRIMES if primes is None else primes)
    if len(values) < 2:
        raise ValueError('Reconstruction needs at least two primes.')
    result = []
    for p in values:
        if isinstance(p, (bool, float, complex)) or int(p) != p or not 2 < p < 2**31:
            raise ValueError('Use distinct odd integer primes smaller than 2**31.')
        p = int(p)
        if p not in _TRUSTED and not core._is_prime(p):
            raise ValueError('Use distinct odd primes smaller than 2**31.')
        result.append(p)
    if len(set(result)) != len(result):
        raise ValueError('Reconstruction primes must be distinct.')
    return result
