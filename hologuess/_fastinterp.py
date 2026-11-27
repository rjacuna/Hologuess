"""Fast interpolation over GF(p): subproduct trees, multipoint evaluation, and interpolation.

von zur Gathen & Gerhard, *Modern Computer Algebra*, Algorithms 10.3 (subproduct tree), 10.5 (multipoint
evaluation), 10.9 (linear combination) and 10.11 (interpolation).  Interpolating n points costs O(M(n) log n) where
M(n) is the cost of multiplying two degree-n polynomials, against O(n^2) for Newton and O(n^3) for solving the
Vandermonde system by elimination.

Whether that is actually faster here depends entirely on M(n).  With schoolbook multiplication the log n factor makes
this LOSE to a single Gaussian elimination shared across many right-hand sides, which is what the caller has.  It only
wins with an FFT, and it wins big: these primes all satisfy 2^24 | p - 1, so a number-theoretic transform is
available.  `benchmark()` below measures the crossover rather than assuming it; `aj/qhsolve.py` picks a method from
the result.
"""
import numpy as np

_NTT = {}


def _ntt_for(p):
    """The project's NTT for this prime, or None if it cannot be loaded (then everything falls back to schoolbook)."""
    if p in _NTT: return _NTT[p]
    obj = None
    try:
        from . import _pmbasis as pmbasis
        g = _primitive_root(p)
        obj = pmbasis.NTT(p, g)
        a = np.arange(1, 9, dtype=np.int64); b = np.arange(3, 11, dtype=np.int64)
        if not np.array_equal(_mul_ntt(obj, a, b, p), _mul_school(a, b, p)): obj = None   # convention check
    except Exception:
        obj = None
    _NTT[p] = obj
    return obj


def _primitive_root(p):
    fac, m = [], p - 1
    d = 2
    while d * d <= m:
        if m % d == 0:
            fac.append(d)
            while m % d == 0: m //= d
        d += 1
    if m > 1: fac.append(m)
    for g in range(2, 200):
        if all(pow(g, (p - 1) // f, p) != 1 for f in fac): return g
    raise RuntimeError("no primitive root found for %d" % p)


def _mul_school(a, b, p):
    """O(len(a) * len(b)) with a reduction per row: int64-safe, since each row adds one product below p^2."""
    if len(a) == 0 or len(b) == 0: return np.zeros(0, dtype=np.int64)
    if len(a) > len(b): a, b = b, a
    out = np.zeros(len(a) + len(b) - 1, dtype=np.int64)
    for i in range(len(a)):
        ai = int(a[i])
        if ai: out[i:i + len(b)] = (out[i:i + len(b)] + ai * b) % p
    return out


def _mul_ntt(ntt, a, b, p):
    n = len(a) + len(b) - 1
    L = 1 << max(1, (n - 1).bit_length())
    A = np.zeros(L, dtype=np.int64); A[:len(a)] = a
    B = np.zeros(L, dtype=np.int64); B[:len(b)] = b
    C = ntt.transform(ntt.transform(A) * ntt.transform(B) % p, inverse=True)
    return C[:n]                     # this NTT's inverse transform already divides by L


SCHOOL_CUTOFF = 48          # below this the transform's own overhead dominates


def polymul(a, b, p, ntt=None):
    a = np.trim_zeros(np.asarray(a, dtype=np.int64), 'b'); b = np.trim_zeros(np.asarray(b, dtype=np.int64), 'b')
    if len(a) == 0 or len(b) == 0: return np.zeros(1, dtype=np.int64)
    if ntt is None or min(len(a), len(b)) < SCHOOL_CUTOFF: return _mul_school(a, b, p)
    return _mul_ntt(ntt, a, b, p)


def polyrem(a, b, p, ntt=None):
    """a mod b by plain long division; b is monic in every use here."""
    a = np.asarray(a, dtype=np.int64).copy(); db = len(b) - 1
    if db < 0: raise ZeroDivisionError
    if len(a) - 1 < db: return a
    inv_lead = pow(int(b[db]), p - 2, p)
    for i in range(len(a) - 1, db - 1, -1):
        c = int(a[i]) * inv_lead % p
        if c: a[i - db:i + 1] = (a[i - db:i + 1] - c * b) % p
    return a[:db] if db else np.zeros(1, dtype=np.int64)


def subproduct_tree(u, p, ntt=None):
    """Algorithm 10.3: level 0 holds x - u_i, each level up is the pairwise product.  Returned bottom-up."""
    level = [np.array([(-int(x)) % p, 1], dtype=np.int64) for x in u]
    tree = [level]
    while len(level) > 1:
        nxt = [polymul(level[i], level[i + 1], p, ntt) if i + 1 < len(level) else level[i]
               for i in range(0, len(level), 2)]
        tree.append(nxt); level = nxt
    return tree


def multipoint_eval(f, u, p, tree=None, ntt=None):
    """Algorithm 10.5: evaluate f at every u_i by taking remainders down the subproduct tree."""
    n = len(u)
    if n == 0: return np.zeros(0, dtype=np.int64)
    if tree is None: tree = subproduct_tree(u, p, ntt)
    cur = [polyrem(np.asarray(f, dtype=np.int64), tree[-1][0], p, ntt)]
    for lvl in range(len(tree) - 2, -1, -1):
        nxt = []
        for i, r in enumerate(cur):
            for c in (2 * i, 2 * i + 1):
                if c < len(tree[lvl]): nxt.append(polyrem(r, tree[lvl][c], p, ntt))
        cur = nxt
    return np.array([int(r[0]) % p if len(r) else 0 for r in cur[:n]], dtype=np.int64)


def _pad(a, b, p):
    n = max(len(a), len(b)); out = np.zeros(n, dtype=np.int64)
    out[:len(a)] = a; out[:len(b)] = (out[:len(b)] + b) % p
    return out


def interpolate(u, v, p, ntt=None, tree=None, si=None):
    """Algorithm 10.11: the unique polynomial of degree < n through (u_i, v_i), as coefficients low to high.

    `tree` and `si` depend only on the points, so a caller interpolating many value vectors through the SAME points
    computes them once and passes them in -- which is the whole reason this can beat one shared elimination."""
    u = np.asarray(u, dtype=np.int64); v = np.asarray(v, dtype=np.int64); n = len(u)
    if n == 0: return np.zeros(0, dtype=np.int64)
    if n == 1: return np.array([int(v[0]) % p], dtype=np.int64)
    if tree is None: tree = subproduct_tree(u, p, ntt)
    if si is None: si = interp_weights(u, p, ntt, tree)
    c = v * si % p
    out = _lc_indexed(c, tree, len(tree) - 1, 0, p, ntt, 0)
    return out[:n]


def _lc_indexed(c, tree, lvl, i, p, ntt, base):
    """As _linear_combination, but tracking which leaves the subtree covers so c is indexed correctly."""
    if lvl == 0: return np.array([int(c[base]) % p], dtype=np.int64)
    l, r = 2 * i, 2 * i + 1
    if r >= len(tree[lvl - 1]): return _lc_indexed(c, tree, lvl - 1, l, p, ntt, base)
    nleft = len(tree[lvl - 1][l]) - 1
    cl = _lc_indexed(c, tree, lvl - 1, l, p, ntt, base)
    cr = _lc_indexed(c, tree, lvl - 1, r, p, ntt, base + nleft)
    return _pad(polymul(tree[lvl - 1][r], cl, p, ntt), polymul(tree[lvl - 1][l], cr, p, ntt), p)


def interp_weights(u, p, ntt=None, tree=None):
    """1 / m'(u_i) for m = prod (x - u_i): the part of interpolation that depends only on the points."""
    u = np.asarray(u, dtype=np.int64)
    if tree is None: tree = subproduct_tree(u, p, ntt)
    if len(set(int(x) % p for x in u)) != len(u):
        raise ValueError("fastinterp: the evaluation points must be distinct; m'(u_i) vanishes on a repeat and the "
                         "interpolation would return silent garbage")
    m = tree[-1][0]
    deriv = (np.arange(1, len(m), dtype=np.int64) * m[1:]) % p
    d = multipoint_eval(deriv, u, p, tree, ntt)
    from ._qhsolve import inv_mod
    return inv_mod(d, p)
