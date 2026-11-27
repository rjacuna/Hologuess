"""A q-holonomic solver: the minimal recurrence of a sequence, from its values at specialisations q -> g in GF(p).

Nothing here knows what the sequence means.  A sequence s(0), s(1), ... of Laurent polynomials in q is q-holonomic when
some operator  T = sum_{b <= r, j <= d} u_{bj}(q) x^j E^b  annihilates it, where E shifts the index and x acts as q^k at
index k:

    sum_{b, j} u_{bj}(q) q^{jk} s(k + b) = 0   for every window k >= 0.

At a point q = g each cell (b, j) collapses to one unknown and each window to one scalar equation, so the operator at g
is the nullspace of a (#windows x #cells) matrix over GF(p).  Normalised by a reference cell, the per-cell values across
many points are the values of Laurent polynomials in q, recovered by a Vandermonde solve on a band of exponents, CRT
across primes and rational reconstruction.

The caller supplies a `Sequence`: something that can evaluate the sequence at points, say how many terms it can reach,
and (when it can) hand over the exact Laurent polynomials, which are what the band predictor and the certificates use.
`solve` then runs the four stages -- detect the corner, restrict to the support, plan the band, extract -- and every
bound it uses is measured from the data rather than assumed, because the shape of one sequence's operator says nothing
about another's.

The governing rule, and the reason nothing here ever "handles" a larger kernel: the Ore algebra over the rational
function field is a left principal ideal domain, so two independent annihilators of the same order force the minimal one
to have LOWER order and HIGHER degree -- outside the box being looked at.  A kernel of dimension > 1 therefore means the
bounds are wrong, and the answer is to adapt them, never to pick an element out of the module.
"""
import os
import math, time
import numpy as np
from ._kernels import active_kernel
from ._resources import limit_extraction_threads, operation_native_limits

MARGIN = 10          # extra windows beyond the unknown count in every detection / extraction system
GUARD = 8            # the reconstructed exponent support must stay this far from the band edges
NUSE_EXTRA = 16      # windows beyond the cell count in an extraction system -- see below

# NUSE_EXTRA is the most expensive constant in the project.  `terms_needed = cells + NUSE_EXTRA + order + 2` is how
# many colours EVERY point must be carried to, and colours are the engine's entire cost, growing like N^3 each.
# It was 60 for no recorded reason.  Measured on four knots with certified operators (7_7, 7_3, 6_3, 8_3), the
# extraction returns the identical operator at 60, 16 and even 8, while the colour depth falls by 26 to 46 percent:
#
#     7_7  170 -> 126 -> 118 colours        6_3   96 -> 52 -> 44
#     7_3  150 -> 106 ->  98                8_3  126 -> 82 -> 74
#
# 16 rather than 8 because the margin is what absorbs a knot whose support is less regular than these, and the
# saving between the two is small.  Too few windows fails loudly, not silently: a spurious kernel raises
# KernelNotALine, and a spurious annihilator is caught by the exact-annihilation certificate.
MAX_POINTS = 2400    # widest interpolation band tried with one reference cell before switching to the next candidate
FALLBACK_BAND = 64   # starting half-band when the degrees cannot be continued and nothing can be predicted
OVERSAMPLE = 0.15    # points solved beyond the band's need, as a fraction of it (at least 32): see best_reference
                     # for why it must scale with the band -- a flat 32 starved the reference measurement
MIN_PRIMES = 2       # primes whose reconstructions must agree before the extraction stops adding primes


# ----------------------------------------------------------------------------------------------------------------------
# arithmetic mod p
# ----------------------------------------------------------------------------------------------------------------------
def _is_prime(n):
    if n < 2 or n % 2 == 0: return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0: return False
        d += 2
    return True

# Exact JonesRec pool, precomputed to avoid trial division at every cold import.
# tests/test_prime_pool.py verifies both original construction and primality.
PRIMES = [
    2130706433, 2113929217, 2013265921, 1811939329, 1711276033, 1224736769,
    1107296257, 754974721, 469762049, 167772161, 2147483647, 2147483629,
    2147483587, 2147483579, 2147483563, 2147483549, 2147483543, 2147483497,
    2147483489, 2147483477, 2147483423, 2147483399, 2147483353, 2147483323,
    2147483269, 2147483249, 2147483237, 2147483179, 2147483171, 2147483137,
    2147483123, 2147483077, 2147483069, 2147483059, 2147483053, 2147483033,
    2147483029, 2147482951, 2147482949, 2147482943, 2147482937, 2147482921,
    2147482877, 2147482873, 2147482867, 2147482859, 2147482819, 2147482817,
    2147482811, 2147482801, 2147482763, 2147482739, 2147482697, 2147482693,
    2147482681, 2147482663, 2147482661, 2147482621, 2147482591, 2147482583,
]
# Ten primes is 300 bits, and 8_6's coefficients need more than that.  Nothing in the solver needs 2^24 | p - 1 --
# that was for the subproduct-tree interpolation kept as an unused option -- so the list continues with the primes
# just below 2^31, which every route (limb matmul, Kauffman points, CRT) accepts.  The first ten stay first, so
# stored residues and point sets keep their primes.
def _more_primes(count, below=(1 << 31) - 1):
    out = []; n = below
    while len(out) < count:
        if _is_prime(n) and n not in PRIMES: out.append(n)
        n -= 2
    return out


def pow_mod(a, e, p):
    """a^e mod p for an int64 array a and integer e >= 0."""
    x = np.ones_like(a); b = a % p
    while e:
        if e & 1: x = x * b % p
        b = b * b % p; e >>= 1
    return x

def inv_mod(a, p): return pow_mod(a, p - 2, p)

def eval_laurent(polys, q, p):
    """polys: list of {e: c} integer Laurent polynomials -> (len(polys), K) values at the points q."""
    K = q.shape[0]; emax = max((max(abs(e) for e in d) for d in polys if d), default=0)
    pos = np.empty((emax + 1, K), dtype=np.int64); pos[0] = 1
    for e in range(1, emax + 1): pos[e] = pos[e - 1] * q % p
    qi = inv_mod(q, p); neg = np.empty((emax + 1, K), dtype=np.int64); neg[0] = 1
    for e in range(1, emax + 1): neg[e] = neg[e - 1] * qi % p
    out = np.zeros((len(polys), K), dtype=np.int64)
    for i, d in enumerate(polys):
        if not d: continue
        es = np.fromiter(d.keys(), dtype=np.int64, count=len(d)); cs = np.fromiter((c % p for c in d.values()), dtype=np.int64, count=len(d))
        pw = np.where(es[:, None] >= 0, pos[np.abs(es)], neg[np.abs(es)])          # (terms, K)
        acc = np.zeros(K, dtype=np.int64)
        for s in range(0, len(es), 4096): acc = (acc + (cs[s:s + 4096, None] * pw[s:s + 4096] % p).sum(axis=0)) % p
        out[i] = acc
    return out


# ----------------------------------------------------------------------------------------------------------------------
# linear algebra mod p
# ----------------------------------------------------------------------------------------------------------------------
def gemm_mod(p, X, Y):
    """X @ Y over GF(p) through BLAS.  float64 is exact on integers below 2^53 and an unreduced entry of the product
    is at most C (p-1)^2, so one matmul suffices when that fits and three on 16-bit limbs otherwise."""
    if X.shape[-1] == 0: return np.zeros((X.shape[0], Y.shape[1]), dtype=np.int64)
    with operation_native_limits():
        if int(X.shape[-1]) * (int(p) - 1) ** 2 < (1 << 53):
            return (X.astype(np.float64) @ Y.astype(np.float64)).astype(np.int64) % p
        Xh = (X >> 16).astype(np.float64); Xl = (X & 0xFFFF).astype(np.float64)
        Yh = (Y >> 16).astype(np.float64); Yl = (Y & 0xFFFF).astype(np.float64)
        Phh = Xh @ Yh; Pll = Xl @ Yl; Pm = (Xh + Xl) @ (Yh + Yl) - Phh - Pll
        Phh = Phh.astype(np.int64) % p; Pll = Pll.astype(np.int64) % p; Pm = Pm.astype(np.int64) % p
        return (Phh * ((1 << 32) % p) + Pm * 65536 + Pll) % p

PANEL = 48          # columns eliminated before the trailing submatrix is brought up to date in one product
BLOCK_MIN = 100000  # elements below which the simple loop is faster than maintaining the panel's transformation

@limit_extraction_threads
def rref_mod(A, p, panel=None):
    """RREF of an int64 matrix mod p. Returns (R, pivot columns).

    BLOCKED, and the reason it is correct is worth stating because getting it wrong is easy and quiet.

    Gauss-Jordan only ever adds multiples of a PIVOT row to other rows, and a pivot row is itself only ever modified
    by other pivot rows.  So the whole transformation a panel performs stays inside the span of the panel's original
    pivot rows: it is `I + U E`, where E selects those rows and U is (rows x panel width).  Matrices of that shape
    are closed under multiplication for a fixed E, which is what makes the composite of the panel's elementary
    operations have the same shape, and `(I + U E) X = X + U (E X)` is what makes the deferred update exact.  Both
    facts are proved in lean/JonesRec/BlockedElimination.lean.

    U cannot be read off the reduced pivot columns afterwards -- that is the wrong version, and it still produces
    plausible output -- so it is accumulated here as `Z` alongside the panel, by applying every operation to it too.
    Row swaps are not of that shape, so they are applied to the whole row immediately and never deferred.

    The payoff is that the O(n^3) work moves out of one full-array numpy pass per pivot and into one GEMM per panel."""
    kernel = active_kernel()
    if kernel is not None:
        return kernel.rref(A, p)
    A = A % p; R, C = A.shape; piv = []; r = 0; col = 0
    nb = panel or PANEL
    # Accumulating Z costs about as much per pivot as the elimination it defers, so blocking only pays once the
    # trailing block it saves is large.  Measured crossover is near 300x300; below it the simple loop wins.
    if panel is None and R * C < BLOCK_MIN:
        for c in range(C):
            if r >= R: break
            nz = np.nonzero(A[r:, c])[0]
            if nz.size == 0: continue
            pr = r + nz[0]
            if pr != r: A[[r, pr]] = A[[pr, r]]
            A[r] = A[r] * pow(int(A[r, c]), p - 2, p) % p
            rows = np.nonzero(A[:, c])[0]; rows = rows[rows != r]
            if rows.size: A[rows] = (A[rows] - np.outer(A[rows, c], A[r])) % p
            piv.append(c); r += 1
        return A, piv
    while col < C and r < R:
        top = min(col + nb, C); prow = r; width = top - col
        Z = np.zeros((R, width), dtype=np.int64)     # new_row_i = old_row_i + Z[i] . (original pivot rows)
        prows = []
        for c in range(col, top):
            if r >= R: break
            nz = np.nonzero(A[r:, c])[0]
            if nz.size == 0: continue
            pr = r + nz[0]
            if pr != r:                               # a swap is a permutation, not of the form I + U E: do it now,
                A[[r, pr]] = A[[pr, r]]; Z[[r, pr]] = Z[[pr, r]]      # to the whole row, and to the record
            k = len(prows); prows.append(r)
            sc = pow(int(A[r, c]), p - 2, p)
            A[r, col:top] = A[r, col:top] * sc % p
            Z[r] = Z[r] * sc % p; Z[r, k] = (Z[r, k] + sc - 1) % p    # scaling row r: I + (s-1) e_r e_r^T
            rows = np.nonzero(A[:, c])[0]; rows = rows[rows != r]
            if rows.size:
                m = A[rows, c].copy()
                A[np.ix_(rows, np.arange(col, top))] = (A[np.ix_(rows, np.arange(col, top))]
                                                        - np.outer(m, A[r, col:top])) % p
                Z[rows] = (Z[rows] - np.outer(m, Z[r])) % p
                Z[rows, k] = (Z[rows, k] - m) % p
            piv.append(c); r += 1
        if prows and top < C:                          # ---- one product brings the trailing block up to date
            S = A[np.array(prows), top:].copy()        # still the ORIGINAL rows: nothing deferred has touched them
            A[:, top:] = (A[:, top:] + gemm_mod(p, Z[:, :len(prows)], S)) % p
        col = top
    return A, piv

def rank_mod(A, p):
    """Rank only.  `dim_at` asked for a whole nullspace BASIS and then read its shape to learn a dimension."""
    return len(rref_mod(A, p)[1])

def nullspace_mod(A, p):
    """Right nullspace basis of A mod p as an (dim, C) array."""
    R, piv = rref_mod(A, p); C = A.shape[1]; pset = set(piv); free = [c for c in range(C) if c not in pset]
    B = np.zeros((len(free), C), dtype=np.int64)
    for t, f in enumerate(free):
        B[t, f] = 1
        for i, pc in enumerate(piv): B[t, pc] = (-R[i, f]) % p
    return B

@limit_extraction_threads
def batched_nullspace(M, p):
    """Nullspaces of a batch of matrices M (K, R, C) mod p sharing the generic pivot profile (points are generic).
    Returns (basis (K, dim, C), alive mask): points whose profile deviates from the majority are marked dead."""
    kernel = active_kernel()
    if kernel is not None:
        return kernel.batched_nullspace(M, p)
    A = M % p; K, R, C = A.shape; piv = []; r = 0; alive = np.ones(K, dtype=bool); idx = np.arange(K)
    if R * C < BLOCK_MIN:                      # maintaining the record costs about what it defers; see rref_mod
        for col in range(C):
            if r >= R: break
            nz = A[:, r:, col] != 0; has = nz.any(axis=1)
            # the majority is taken over the LIVE points: a point already marked dead carries garbage from the
            # profile it deviated on, and must not sway the vote for the others
            if (has & alive).sum() * 2 <= alive.sum(): alive &= ~has; continue
            alive &= has
            pr = r + nz.argmax(axis=1); pr = np.where(has, pr, r)
            rowr = A[idx, r].copy(); A[idx, r] = A[idx, pr]; A[idx, pr] = rowr
            inv = inv_mod(np.where(A[:, r, col] == 0, 1, A[:, r, col]), p)
            A[:, r] = A[:, r] * inv[:, None] % p
            f = A[:, :, col].copy(); f[:, r] = 0
            A = (A - f[:, :, None] * A[:, r, :][:, None, :]) % p
            piv.append(col); r += 1
        pset = set(piv); free = [c for c in range(C) if c not in pset]
        B = np.zeros((K, len(free), C), dtype=np.int64)
        for t, fcol in enumerate(free):
            B[:, t, fcol] = 1
            for i, pc in enumerate(piv): B[:, t, pc] = (-A[:, i, fcol]) % p
        return B, alive
    nb = PANEL
    col0 = 0
    while col0 < C and r < R:
        top = min(col0 + nb, C); Z = np.zeros((K, R, top - col0), dtype=np.int64); prows = []
        for col in range(col0, top):
            if r >= R: break
            nz = A[:, r:, col] != 0; has = nz.any(axis=1)
            if (has & alive).sum() * 2 <= alive.sum():              # majority of the LIVE points: free column
                alive &= ~has; continue
            alive &= has
            pr = r + nz.argmax(axis=1); pr = np.where(has, pr, r)
            # the swap is a permutation, and a different one per point: apply it now, to the whole row and to the
            # record, so that everything deferred below is of the form I + U E (see BlockedElimination.lean)
            rowr = A[idx, r].copy(); A[idx, r] = A[idx, pr]; A[idx, pr] = rowr
            zrow = Z[idx, r].copy(); Z[idx, r] = Z[idx, pr]; Z[idx, pr] = zrow
            k = len(prows); prows.append(r)
            inv = inv_mod(np.where(A[:, r, col] == 0, 1, A[:, r, col]), p)
            A[:, r, col0:top] = A[:, r, col0:top] * inv[:, None] % p
            Z[:, r] = Z[:, r] * inv[:, None] % p; Z[:, r, k] = (Z[:, r, k] + inv - 1) % p
            f = A[:, :, col].copy(); f[:, r] = 0
            A[:, :, col0:top] = (A[:, :, col0:top] - f[:, :, None] * A[:, r, col0:top][:, None, :]) % p
            Z = (Z - f[:, :, None] * Z[:, r, :][:, None, :]) % p
            Z[:, :, k] = (Z[:, :, k] - f) % p
            piv.append(col); r += 1
        if prows and top < C:                    # one batched product brings the trailing block up to date
            S = A[:, np.array(prows), top:]
            A[:, :, top:] = (A[:, :, top:] + gemm_mod(p, Z[:, :, :len(prows)], S)) % p
        col0 = top
    pset = set(piv); free = [c for c in range(C) if c not in pset]
    B = np.zeros((K, len(free), C), dtype=np.int64)
    for t, fcol in enumerate(free):
        B[:, t, fcol] = 1
        for i, pc in enumerate(piv): B[:, t, pc] = (-A[:, i, fcol]) % p
    return B, alive

def point_matrices(seq, cells, q, p, nrows, zero_cells=()):
    """Annihilation matrices at the points: M[k, c] = q^{j k} seq[k + b] for cells c = (b, j), windows k < nrows,
    followed by one row per cell in zero_cells forcing that coefficient to vanish.  seq: (>= nrows + max b, K).
    Returns (K, nrows + |zero_cells|, ncell)."""
    if seq.ndim == 3:
        # Generic module observations: (observation, action, parameter point).
        # Here a cell (i, 0) labels action i, without any shift interpretation.
        if any(j != 0 for _, j in cells):
            raise ValueError("generic action cells must have zero second coordinate")
        M = np.ascontiguousarray(seq[:nrows, [i for i, _ in cells], :].transpose(2, 0, 1))
        if zero_cells:
            extra = np.zeros((len(q), len(zero_cells), len(cells)), dtype=np.int64)
            for r, c in enumerate(zero_cells): extra[:, r, c] = 1
            M = np.concatenate([M, extra], axis=1)
        return M
    K = q.shape[0]; jmax = max(j for (_, j) in cells)
    qk = np.empty((nrows, K), dtype=np.int64); qk[0] = 1
    for k in range(1, nrows): qk[k] = qk[k - 1] * q % p
    qkj = [np.ones((nrows, K), dtype=np.int64)]
    for j in range(1, jmax + 1): qkj.append(qkj[-1] * qk % p)
    M = np.empty((nrows + len(zero_cells), len(cells), K), dtype=np.int64)
    for c, (b, j) in enumerate(cells): M[:nrows, c, :] = qkj[j] * seq[b:b + nrows] % p
    for r, c in enumerate(zero_cells): M[nrows + r] = 0; M[nrows + r, c, :] = 1
    return np.ascontiguousarray(M.transpose(2, 0, 1))

INTERP_METHOD = 'rref'   # 'rref' or 'fast' (von zur Gathen & Gerhard); see the note below

def _vandermonde_fast(q, Y, es, p):
    """The same solve by subproduct-tree interpolation.  The exponents es are consecutive, so dividing the values by
    q^es[0] turns the Laurent fit into an ordinary polynomial interpolation of degree < len(es)."""
    from . import _fastinterp as fi
    n = len(es); u = q[:n] % p
    if n > 1 and any(es[i + 1] - es[i] != 1 for i in range(n - 1)):
        raise ValueError("_vandermonde_fast needs consecutive exponents: it divides by q^es[0] to turn the Laurent "
                         "fit into an ordinary interpolation of degree < len(es)")
    ntt = fi._ntt_for(p)
    shift = inv_mod(pow_mod(u, es[0], p) if es[0] >= 0 else pow_mod(inv_mod(u, p), -es[0], p), p)
    tree = fi.subproduct_tree(u, p, ntt); si = fi.interp_weights(u, p, ntt, tree)
    X = np.zeros((n, Y.shape[1]), dtype=np.int64)
    for c in range(Y.shape[1]):
        co = fi.interpolate(u, Y[:n, c] * shift % p, p, ntt, tree, si)
        X[:len(co), c] = co
    ok = True
    if q.shape[0] > n:                                    # the same held-out check the elimination does
        V = np.empty((q.shape[0] - n, n), dtype=np.int64); qi = inv_mod(q[n:], p)
        for t, e in enumerate(es): V[:, t] = pow_mod(q[n:], e, p) if e >= 0 else pow_mod(qi, -e, p)
        pred = np.zeros((q.shape[0] - n, Y.shape[1]), dtype=np.int64)
        for t in range(n): pred = (pred + V[:, t:t + 1] * X[t:t + 1, :] % p) % p
        ok = np.array_equal(pred, Y[n:] % p)
    return X, ok

def vandermonde_solve(q, Y, es, p, method=None):
    """Coefficients c_e (e in es) with sum_e c_e q_i^e = Y[i, :] for the first len(es) points; returns (len(es), ncell)
    and the residual check on the remaining points (True if they all agree).

    Two methods, and the default was chosen by measurement, not by asymptotics.  `rref` solves the augmented
    Vandermonde system by one Gaussian elimination, O(n^2 (n + ncell)).  `fast` uses subproduct-tree interpolation
    (aj/fastinterp.py, von zur Gathen & Gerhard Alg. 10.11), O(M(n) log n) per column, with a real NTT underneath.
    The asymptotics favour `fast`; the measurements do not, at any shape this solver meets:

        points  cells   rref    fast
           128     64   0.01s   0.19s
           256    128   0.05s   0.78s
           512    256   0.39s   3.17s
           760    400   1.31s   7.43s

    The reason is that this caller has ONE point set and MANY right-hand sides.  The elimination shares its whole
    cost across all of them and runs as vectorised numpy over the columns, while tree interpolation shares only the
    tree and pays a Python-level recursion of small polynomial products per column.  `fast` would need the tree
    operations batched across columns to compete, and at 1.3 seconds against an engine that runs for hours there is
    nothing here worth winning.  Kept as an option, and tested, so the conclusion can be rechecked rather than
    retold."""
    if (method or INTERP_METHOD) == 'fast':
        # The optional NTT/PM-Basis backend can call its own BLAS routines.
        with operation_native_limits():
            return _vandermonde_fast(q, Y, es, p)
    n = len(es); qi = inv_mod(q, p)
    V = np.empty((q.shape[0], n), dtype=np.int64)
    if n and all(es[t + 1] - es[t] == 1 for t in range(n - 1)):
        # consecutive exponents, which is every band this solver builds: one running product, not one
        # exponentiation per column
        V[:, 0] = pow_mod(q, es[0], p) if es[0] >= 0 else pow_mod(qi, -es[0], p)
        for t in range(1, n): V[:, t] = V[:, t - 1] * q % p
    else:
        for t, e in enumerate(es): V[:, t] = pow_mod(q, e, p) if e >= 0 else pow_mod(qi, -e, p)
    R, piv = rref_mod(np.concatenate([V[:n], Y[:n] % p], axis=1), p)
    if piv[:n] != list(range(n)): raise RuntimeError("singular Vandermonde system")
    X = R[:n, n:]
    ok = True
    if q.shape[0] > n:
        pred = np.zeros((q.shape[0] - n, Y.shape[1]), dtype=np.int64)
        for t in range(n): pred = (pred + V[n:, t:t + 1] * X[t:t + 1, :] % p) % p
        ok = np.array_equal(pred, Y[n:] % p)
    return X, ok


# ----------------------------------------------------------------------------------------------------------------------
# CRT, rational reconstruction, exact certificates
# ----------------------------------------------------------------------------------------------------------------------
def crt_pair(residues, primes):
    """residues: list over primes of int arrays (same shape) -> Python-int object array of CRT lifts in [0, prod)."""
    from functools import reduce
    Mtot = reduce(lambda a, b: a * b, primes); out = np.zeros(residues[0].shape, dtype=object)
    for r, pi in zip(residues, primes):
        Mi = Mtot // pi; yi = pow(Mi, pi - 2, pi); out = out + r.astype(object) * (Mi * yi)
    return out % Mtot, Mtot

def ratrec(x, Mtot):
    """Rational reconstruction of x mod Mtot (Wang): (num, den) with |num|, den <= isqrt(Mtot/2) and den prime to
    Mtot, or None.

    The gcd check is von zur Gathen–Gerhard's (Modern Computer Algebra, Theorem 5.26).  Without it a success need
    not represent x: ratrec(36, 105) returned (3, 3), and 3/3 = 1 is not 36 mod 105, because 3 is not invertible
    mod 105.  In the pipeline this needs three or more 31-bit primes (then the bound exceeds each prime, so a prime
    can divide the denominator).  The exact certificate downstream would have rejected any operator built on such
    a value; with the check it is a reconstruction failure instead, which adds a prime.  This function is mirrored
    line for line by `ratrec` in lean/JonesRec/Recon/Wang.lean, where it is proved sound and complete (Wang's
    theorem), and aj/test_qhsolve.py checks the two against each other."""
    bound = math.isqrt(Mtot // 2)
    r0, r1 = Mtot, x % Mtot; t0, t1 = 0, 1
    while r1 > bound:
        qq = r0 // r1; r0, r1 = r1, r0 - qq * r1; t0, t1 = t1, t0 - qq * t1
    if t1 == 0 or abs(t1) > bound: return None
    if math.gcd(t1, Mtot) != 1: return None
    if t1 < 0: t1, r1 = -t1, -r1
    return r1, t1

def reconstruct_operator(residues, primes, cells, es):
    """residues: list over primes of (len(es), ncell) int64 arrays -> ({(b, j, e): integer coefficient} primitive,
    number of failures, certainty bound).

    The third value is the largest |numerator| * denominator reconstructed, with the modulus, for the log.  It is
    NOT a stopping criterion, and an earlier version that used `2 * bound < M` as one was testing a tautology:
    Wang's EEA only ever returns |a|, b <= sqrt(M/2), so 2|a|b <= M holds for every success.  What is true is
    weaker and worth stating exactly.  A rational a/b with 2|a|b < M is the unique rational of that height with its
    residue, and a success is the only small rational consistent with the residues; but a coefficient whose true
    value is LARGER than the ceiling reconstructs to some wrong small rational about 61% of the time (measured on
    200 000 random residues at one prime and at two), so success on every coefficient says nothing about a
    coefficient that is too big for the modulus.  The extraction therefore stops adding primes only when the
    reconstruction from k primes AGREES with the one from k-1, k >= MIN_PRIMES: two moduli returning the same wrong
    rational is what is negligible.  The exact certificate downstream decides correctness either way.

    The ceiling is worth knowing concretely: one 31-bit prime reaches |a|, b < 32639, two reach about 1.5e9, and
    8_5 has 2047 coefficients beyond the two-prime ceiling."""
    from math import gcd
    X, Mtot = crt_pair(residues, primes); fails = 0; num = {}; bound = 0
    for t, e in enumerate(es):
        for c, bj in enumerate(cells):
            x = int(X[t, c])
            if x == 0: continue
            rr = ratrec(x, Mtot)
            if rr is None: fails += 1; rr = ((x - Mtot) if x > Mtot // 2 else x, 1)
            else: bound = max(bound, abs(rr[0]) * rr[1])
            num[(bj[0], bj[1], e)] = rr
    if fails:
        # The coefficients share ONE denominator: the clearing makes the reference cell's polynomial monic, so every
        # value is a_i / lc with lc that polynomial's leading coefficient, and when lc is large its bits are paid
        # twice in every coefficient.  Ratios do not see it: x_i / x_0 = a_i / a_0 for any nonzero base x_0, with
        # height |a_i| |a_0| -- so relative to a SMALL base coefficient the primitive integer vector reconstructs
        # with about the bits of the a_i alone.  Which base is small is not known, but a small one is the one
        # with the fewest failures, so a handful of candidates are screened on a sample and the best is used.
        # 8_6: 38% of 180 000 coefficients failed at 273 bits in the lc gauge.  A wrong answer is impossible to
        # accept here: every coefficient must reconstruct, and the exact certificate decides afterwards.
        import random
        pos = [(t, c) for t in range(len(es)) for c in range(len(cells)) if int(X[t, c])]
        rng = random.Random(len(pos)); sample = rng.sample(pos, min(600, len(pos)))
        bases = rng.sample(pos, min(400, len(pos)))     # a small coefficient is rare; screen many, cheaply
        best = None
        for (t0, c0) in bases:
            inv = pow(int(X[t0, c0]), -1, Mtot)
            bad = sum(1 for (t, c) in sample if ratrec(int(X[t, c]) * inv % Mtot, Mtot) is None)
            if best is None or bad < best[0]: best = (bad, t0, c0, inv)
        bad, t0, c0, inv = best
        if bad * 4 < len(sample):                         # the base is worth a full pass
            rfails = 0; rnum = {}
            for t, e in enumerate(es):
                for c, bj in enumerate(cells):
                    x = int(X[t, c])
                    if x == 0: continue
                    rr = ratrec(x * inv % Mtot, Mtot)
                    if rr is None: rfails += 1; rr = (x, 1)
                    rnum[(bj[0], bj[1], e)] = rr
            if rfails == 0:
                L = 1
                for _, d in rnum.values(): L = L * d // gcd(L, d)
                u = {k: n * (L // d) for k, (n, d) in rnum.items()}
                g = 0
                for v in u.values(): g = gcd(g, abs(v))
                if g > 1: u = {k: v // g for k, v in u.items()}
                if u[min(u)] < 0: u = {k: -v for k, v in u.items()}    # a fixed sign, so rounds can be compared
                return u, 0, (max(abs(n) * d for n, d in rnum.values()), Mtot)
            fails = min(fails, rfails)
        # NOT the common denominator of failed reconstructions: those that "succeed" when the modulus is too small
        # are wrong small rationals with RANDOM denominators, and the lcm of hundreds of thousands of them is an
        # integer of millions of bits -- one process reached 38 GB on 8_6.  The callers only need the exponent
        # support and the failure count in this case, so return the numerators.
        return {k: n for k, (n, d) in num.items()}, fails, (bound, Mtot)
    den_l = 1
    for _, d in num.values(): den_l = den_l * d // gcd(den_l, d)
    u = {k: n * (den_l // d) for k, (n, d) in num.items()}
    g = 0
    for v in u.values(): g = gcd(g, abs(v))
    if g > 1: u = {k: v // g for k, v in u.items()}
    if u and u[min(u)] < 0: u = {k: -v for k, v in u.items()}
    return u, fails, (bound, Mtot)

def _laurent_base(coefficients, base_ring=None):
    """Choose exact ZZ/QQ without truncating rational or approximate values."""
    from sage.all import ZZ, QQ
    rational = False
    for c in coefficients:
        if isinstance(c, (int, np.integer)):
            continue
        if isinstance(c, (float, complex, np.floating, np.complexfloating)):
            raise TypeError('Laurent validation requires exact coefficients.')
        parent = getattr(c, 'parent', None)
        if callable(parent) and not parent().is_exact():
            try:
                exact = c.is_exact()
            except (AttributeError, TypeError, ValueError):
                exact = False
            if not exact: raise TypeError('Laurent validation requires exact coefficients.')
        denominator = getattr(c, 'denominator', None)
        if denominator is None:
            raise TypeError('Laurent validation requires integer or rational coefficients.')
        denominator = denominator() if callable(denominator) else denominator
        rational |= denominator != 1
    if base_ring is not None:
        if base_ring not in (ZZ, QQ): raise ValueError('Use ZZ or QQ for Laurent validation.')
        if rational and base_ring == ZZ: raise TypeError('Nonintegral coefficients require QQ.')
        return base_ring
    return QQ if rational else ZZ


def to_sage(u, base_ring=None):
    """{(b, j, e): c} -> {(b, j): Laurent polynomial over ZZ or QQ}."""
    from sage.all import LaurentPolynomialRing
    base_ring = _laurent_base(u.values(), base_ring)
    Rq = LaurentPolynomialRing(base_ring, 'q'); cells = {}
    for (b, j, e), c in u.items(): cells[(b, j)] = cells.get((b, j), Rq(0)) + Rq(c) * Rq.gen() ** e
    return Rq, cells

def annihilates_exactly(u, seq_polys, *, start=0, base_ring=None):
    """Grouped exact Laurent check over ZZ/QQ at the supplied starting index.

    Factored by shift before multiplying.  The identity is

        sum_{b,j} u_{bj}(q) q^{jk} s(k+b)  =  sum_b [ sum_j u_{bj}(q) q^{jk} ] s(k+b),

    and the bracket is a sum of EXPONENT SHIFTS of the coefficient polynomials, which costs no multiplication at
    all.  So a window needs one big product per shift b, not one per cell (b, j) -- a factor of (degree + 1) fewer,
    which for an operator of qN-degree 81 is 82x fewer.  Measured 4-10x end to end on the certificates for 7_4,
    8_1, 6_3, 7_3 and 8_3, with identical verdicts.  Exactness is untouched: this is the same sum, reassociated."""
    u = {key: c for key, c in u.items() if c}
    if not u: raise ValueError('A nonzero operator is required.')
    if isinstance(start, bool) or int(start) != start: raise ValueError('Starting index must be an integer.')
    start = int(start)
    seq_polys = list(seq_polys)
    from itertools import chain
    base_ring = _laurent_base(chain(u.values(), (c for d in seq_polys for c in d.values())), base_ring)
    Rq, cells = to_sage(u, base_ring=base_ring); r = max(b for (b, _) in cells)
    if len(seq_polys) <= r: return False, 0
    S = [Rq(d) for d in seq_polys]
    byb = {}
    for (b, j), ub in cells.items(): byb.setdefault(b, []).append((j, ub))
    for k in range(len(S) - r):
        acc = Rq(0)
        for b, terms in byb.items():
            a = Rq(0)
            for j, ub in terms: a += ub.shift(j * (k + start)) if j * (k + start) else ub
            if a: acc += a * S[k + b]
        if acc != 0: return False, k
    return True, len(S) - r


# ----------------------------------------------------------------------------------------------------------------------
# the band, predicted from the sequence's own degrees
# ----------------------------------------------------------------------------------------------------------------------
def extend_degrees(vals, need):
    """Continue a degree array past the terms actually held, or None if it cannot be done honestly.

    The q-degrees of a q-holonomic sequence are eventually quadratic in the index, so the second differences settle to a
    constant.  Once they have, continuing the array is exact arithmetic on that recurrence rather than a fit -- 8_5's
    top degrees have second difference 4 and its bottom degrees 0, both exactly, over every term held.  When they have
    NOT settled this returns None, so the caller falls back rather than extrapolating a shape the data does not show."""
    v = [int(x) for x in vals]
    if len(v) < 6 or need <= len(v): return v[:max(need, len(v))]
    d2 = [v[k + 2] - 2 * v[k + 1] + v[k] for k in range(len(v) - 2)]
    if len(set(d2[-4:])) != 1: return None
    c = d2[-1]
    while len(v) < need: v.append(2 * v[-1] - v[-2] + c)
    return v

def envelope(polys, cells, nwin, slack=GUARD):
    """Per-cell q-exponent bounds for the operator's coefficients, from the sequence's degrees alone.

    The term u_{bj} q^e (q^k)^j s(k+b) contributes q-powers in [e + kj + bot(k+b), e + kj + top(k+b)] at window k.  The
    windows cancel, so no term may stick out above every other term's top, nor below every other's bottom; taking the
    best such constraint over the windows gives a tropical (min-plus) envelope, per cell.  Returns (top, bot) dicts
    keyed by cell, or None when the degrees cannot be continued far enough to evaluate it.

    This is an ESTIMATE, not a bound, and it estimates the band of the LAURENT ratios u_c / u_ref -- i.e. it assumes
    the reference cell is a monomial.  When it is not, the quantity actually interpolated is D * u_c / u_ref for the
    cleared denominator D, wider by deg D on each side: 8_5's plan said 777 points and the extraction settled at
    993, the whole difference being a degree-108 denominator, not the envelope.  Measured against the certified
    rec_C operators in the store the width it predicts runs from 1.00x to 4.60x of the truth, median 2.04x, with
    ONE genuine shortfall: 8_6, predicted 463 wide against a true span of 1084 (0.44x).  8_6 has 30 honest values,
    and the degree arrays are continued from however many terms are held, so the prediction is only as good as
    the honest data it extrapolates -- 8_5, with 70, was not short.  The extraction therefore keeps widening
    reactively, and the estimate only decides where the search starts.  Its absolute position is meaningful only relative to a reference cell, because the
    operator's overall power of q is a gauge that normalisation fixes.

    `slack` is added on each side because the extraction only accepts a fit whose reconstructed support clears the band
    edges by GUARD; a band predicted exactly to the support would be rejected by that test and widened anyway."""
    if not polys or any(not d for d in polys): return None
    order = max(b for (b, _) in cells)
    tops = extend_degrees([max(d) for d in polys], nwin + order + 1)
    bots = extend_degrees([min(d) for d in polys], nwin + order + 1)
    if tops is None or bots is None or len(tops) < nwin + order + 1: return None
    Gt = [max(k * j + tops[k + b] for (b, j) in cells) for k in range(nwin)]
    Gb = [min(k * j + bots[k + b] for (b, j) in cells) for k in range(nwin)]
    top = {(b, j): min(Gt[k] - k * j - tops[k + b] for k in range(nwin)) + slack for (b, j) in cells}
    bot = {(b, j): max(Gb[k] - k * j - bots[k + b] for k in range(nwin)) - slack for (b, j) in cells}
    return top, bot

def predict_band(polys, cells, nwin, ref=None, fallback=FALLBACK_BAND, verbose=False):
    """A starting interpolation band for the extraction.

    Without a reference cell the envelope's own span is used, which is the band on some unknown normalisation.  Once the
    extraction has chosen its reference cell the band sharpens to the exponents the ratios u_c / u_ref can occupy,
    [min_c bot(c) - top(ref), max_c top(c) - bot(ref)], which is the quantity actually interpolated.  Returns the
    fallback band when the degrees will not support a prediction."""
    env = envelope(polys, cells, nwin) if polys else None
    if env is None: return (-fallback, fallback), False
    top, bot = env
    if ref is None: return (min(bot.values()), max(top.values())), True
    return (min(bot.values()) - top[ref], max(top.values()) - bot[ref]), True


# ----------------------------------------------------------------------------------------------------------------------
# detection: the corner (order, degree) by kernel dimensions at two points
# ----------------------------------------------------------------------------------------------------------------------
def corner_search(seq2, q2, p, max_order, max_degree, verbose=True):
    """seq2: (nvals, 2) values of the sequence at the two points q2.  Finds the first box whose kernel is a LINE,
    scanning orders upward and, within an order, degrees upward.

    A kernel of dimension > 1 is not a corner (see the module docstring): cutting the degree window short is what
    manufactures such boxes -- 6_3's (8, 31) dim-2 corner came of stopping at degree 36, and scanning further gives
    (7, 38) with dimension 1.  So a dim > 1 box is recorded and the search goes on; only the data horizon ends it.

    Within an order the dimension is non-decreasing in the degree, because a wider box contains the narrower one.  So
    one test at the widest degree the data allows settles the whole order when it comes back empty, and otherwise a
    binary search finds the smallest degree that holds anything -- a few tests per order instead of hundreds.

    Returns (corner or None, proved_empty, gaps, multi)."""
    nvals = seq2.shape[0]; proved, gaps, multi = [], [], []
    cap = max_degree if max_degree > 0 else nvals

    def dim_at(r, d):
        cells = [(b, j) for b in range(r + 1) for j in range(d + 1)]; W = nvals - r
        Ms = point_matrices(seq2, cells, q2, p, W)
        # the kernel DIMENSION is all this needs; building the basis and reading its shape was pure waste
        a, b = (len(cells) - rank_mod(Ms[i], p) for i in range(2))
        return a if a == b else None

    for r in range(1, max_order + 1):
        hi = min(cap, (nvals - r - MARGIN) // (r + 1) - 1)
        if hi < 0:
            gaps.append((r, 0, (r + 1) + r + MARGIN)); continue
        if hi < cap: gaps.append((r, hi + 1, (r + 1) * (hi + 2) + r + MARGIN))   # the data stops here, not the box
        top = dim_at(r, hi)
        if top is None: gaps.append((r, hi, 'point disagreement')); continue
        if top == 0:
            proved.append((r, hi)); continue
        lo = 0                                           # degree 0 is a box like any other: an operator need not
        while lo < hi:                                   # involve the index at all.  Smallest degree that holds anything:
            mid = (lo + hi) // 2
            dm = dim_at(r, mid)
            if dm is None: gaps.append((r, mid, 'point disagreement')); break
            if dm >= 1: hi = mid
            else: lo = mid + 1
        else:
            dm = dim_at(r, lo)
            if dm == 1:
                if verbose: print("  corner: order %d, degree %d, kernel dim 1 (orders %s empty)" % (r, lo, proved), flush=True)
                return (r, lo, 1), proved, gaps, multi
            multi.append((r, lo, dm))
            if verbose: print("  order %d, degree %d: kernel dimension %s, so the minimal operator has lower order and"
                              " higher degree: continuing" % (r, lo, dm), flush=True)
    return None, proved, gaps, multi


# ----------------------------------------------------------------------------------------------------------------------
# extraction at points: nullspaces, normalisation, interpolation
# ----------------------------------------------------------------------------------------------------------------------
class KernelNotALine(RuntimeError):
    """The box holds a whole module, not one operator: the bounds are wrong and must be adapted, not worked around."""
    def __init__(self, dim): super().__init__("kernel dimension %d at a point" % dim); self.dim = dim

def ref_candidates(basis, cells, p):
    """Reference cells for normalisation, in order of preference: corners of the (b, j) support (extreme j first, then
    extreme b), nonzero in the kernel vector.  Only a MONOMIAL cell gives Laurent ratios u_c / u_ref (the monomial-corner
    lesson); that cannot be seen at a point, so the extraction tries the candidates in turn until the interpolation of
    the ratios extends to spare points."""
    order = max(b for (b, _) in cells); jmax = max(j for (_, j) in cells); jmin = min(j for (_, j) in cells)
    pref = sorted(range(len(cells)), key=lambda c: (-max(cells[c][1] - jmin, jmax - cells[c][1]), -max(cells[c][0], order - cells[c][0]), cells[c]))
    out = [c for c in pref if basis[c] % p]
    if not out: raise RuntimeError("kernel vector vanishes on every cell")
    return out

def best_reference(p, qg, V, cands, cells, verbose=True, nwit=3, hold=24):   # see the note on the cap at the call site
    """Choose the reference cell by MEASURING it rather than by its position.  This is the default: on 8_2 it found a
    reference needing a degree-136 denominator and finished in 146 s, where taking the first positional candidate and
    clearing whatever denominator it carried had not converged at 681 points per prime.  Normalising the kernel vector by cell c
    makes every other cell the ratio u_x / u_c, which is a Laurent polynomial exactly when u_c is a monomial -- and that
    cannot be read off the box, only measured.  So for each candidate a few witness ratios are rationally reconstructed
    from the values at the points and verified on held-out ones: a monomial reference gives denominators of degree 0,
    and the candidate with the smallest denominator wins."""
    from sage.all import GF, PolynomialRing
    R = PolynomialRing(GF(p), 'x'); x = R.gen(); n = qg.shape[0]; fit = n - hold
    if fit < 12: return cands[0], None
    ncell = V.shape[1]; step = max(1, ncell // 11)
    best = (cands[0], None)
    for c in cands:
        col = V[:, c] % p
        if int((col != 0).sum()) < n - 2: continue
        Y = V * inv_mod(np.where(col == 0, 1, col), p)[:, None] % p
        wit = [w for w in range(0, ncell, step) if w != c and np.count_nonzero(Y[:, w]) > 0.9 * n][:nwit]
        degs = []; complete = True
        for w in wit:
            pts = [(int(qg[i]), int(Y[i, w])) for i in range(fit)]
            try:
                num, den = R.lagrange_polynomial(pts).rational_reconstruction(R.prod(x - aa for aa, _ in pts), fit // 2 - 1, fit // 2 - 1)
            except Exception: complete = False; break
            if all(den(int(qg[i])) != 0 and num(int(qg[i])) == Y[i, w] * den(int(qg[i])) for i in range(fit, n)):
                degs.append(int(den.degree()))
            else: complete = False; break
        # A witness that does not reconstruct within reach is not "no evidence": it is a ratio whose numerator or
        # denominator exceeds what these points can show, which is exactly what a non-monomial reference produces.
        # Scoring a candidate on the witnesses that happened to be small called 8_5's reference a monomial
        # (degree 0) on 809 points where 901 had measured 114, and the extraction then trusted it.  A candidate
        # is measured only when every witness reconstructs and verifies; otherwise it is unknown.
        if not complete or not degs: continue
        score = max(degs)
        if best[1] is None or score < best[1]: best = (c, score)
        if score == 0: break
    if verbose:
        print("  measured reference (b, j) = %s: witness denominators of degree %s" % (cells[best[0]], best[1]), flush=True)
    return best

# The share of the machine's RAM any pool of workers may take, engine or solver, and a forked worker's own baseline.
# 30% is what ran without swap-ins on the 24 GB machine while the user's other applications held ten gigabytes; the
# engine's batches used to be sized to 55% and the two together paged the machine into a state where nothing else ran.
# JR_MEM_FRAC overrides it for a machine that is otherwise idle.
MEM_FRAC = float(os.environ.get('JR_MEM_FRAC', '0.30')); MEM_WORKER = 0.3 * 2 ** 30

def points_per_task(nrows, ncell, procs, cap=16):
    """Points one pool task solves at once, sized so `procs` workers fit in MEM_FRAC of the machine's RAM.

    A task materialises the K point matrices (K x nrows x ncell, int64), reduces a copy, and the elimination keeps
    two more full-size temporaries: five matrices per point, measured on Ahat(8_8) (7 points of 20 MB each: 1.0 GB
    a worker, 0.3 GB of which is the worker's baseline).  A flat K = 16 was fine while systems had a few hundred
    cells; at 1590 cells it put sixteen workers at 2 GB each on a 24 GB machine, swap full and every worker at a
    tenth of a core.  Half of RAM was tried next and still paged, because the rest of the machine is not empty:
    30% is the share that ran without swap-ins.  Below the cap the count follows the budget."""
    try: ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (ValueError, OSError, AttributeError): return cap
    per_point = 5 * nrows * ncell * 8
    return max(1, min(cap, int((MEM_FRAC * ram / max(1, procs) - MEM_WORKER) / per_point)))

def solve_points_task(args):
    """Pool task: RAW kernel vectors at a chunk of points, on the common scaling that the shared pivot profile fixes.
    Normalising by a reference cell happens in the caller, which may first measure which cell to use.  An optional
    sixth argument (the points' indices in the caller's arrays) is handed back so results can arrive in any order."""
    p, q, seq, cells, nrows = args[:5]
    B, alive = batched_nullspace(point_matrices(seq, cells, q, p, nrows), p)
    v = np.zeros((q.shape[0], len(cells)), dtype=np.int64)
    if B.shape[1] == 1: v[:] = B[:, 0, :]        # batched_nullspace fixes one pivot profile, so the scaling is common
    else: alive[:] = False
    return p, q, v, alive, B.shape[1], (args[5] if len(args) > 5 else None)


# ----------------------------------------------------------------------------------------------------------------------
# the kernel store: no solved point is computed twice
# ----------------------------------------------------------------------------------------------------------------------
# A point's nullspace is the expensive unit of every extraction -- for Ahat, whose terms come cheaply from rec_C, it
# is nearly the whole cost (Ahat(8_12): 38 minutes a prime) -- and it used to live only in memory, so an interrupted
# reconstruction lost every point it had solved.  Now every point is looked up in a store before it is solved and
# written to it as its task completes.
#
# THE CALLER DECIDES where that store lives, since only the caller knows whether its sequence is worth keeping: the
# solver knows nothing about knots and does not choose a location.  A store is anything with
#     get_many(keys) -> {key: (alive, vector)}     put_many([(key, p, alive, vector), ...])     describe() -> str
# and two are provided here: `MemoryKernelStore`, the default, which lasts one run (and still saves re-solving across
# the band widenings of that run), and `SQLiteKernelStore(path)`, permanent, for any sequence.  JonesRec's callers
# hand the solver the `kernels` table of the rec_C store itself (jdb.JDB.kernel_store), whose schema is KERNEL_SCHEMA.
#
# Keys are CONTENT addresses, so a store can be shared by every sequence and can never serve a stale answer: a hash
# of everything the point's matrix is built from -- the prime, the point, the cells, the number of windows, and the
# sequence's terms at the point (seq[:nrows + order]).  Different data cannot share a key, and the same data always
# has the same kernel.  What is stored is the RAW kernel vector: its scaling depends on the pivot profile of the batch
# it was solved in, which is harmless because every point is normalised by a reference cell before it is used (the
# chunks of one run already mix profiles the same way).  Dead points -- off the generic profile -- are stored too.
KERNEL_SCHEMA = """
CREATE TABLE IF NOT EXISTS kernels (key BLOB PRIMARY KEY, p INTEGER NOT NULL, ncell INTEGER NOT NULL,
                                    alive INTEGER NOT NULL, vec BLOB NOT NULL, created REAL NOT NULL);
CREATE TRIGGER IF NOT EXISTS kernels_ao_u BEFORE UPDATE ON kernels BEGIN SELECT RAISE(ABORT, 'append-only'); END;
CREATE TRIGGER IF NOT EXISTS kernels_ao_d BEFORE DELETE ON kernels BEGIN SELECT RAISE(ABORT, 'append-only'); END;
"""
KERNEL_STATS = {'reused': 0, 'solved': 0}


class MemoryKernelStore:
    """Temporary: this process only."""
    def __init__(self): self._d = {}
    def get_many(self, keys): return {k: self._d[k] for k in keys if k in self._d}
    def put_many(self, rows):
        for k, p, a, v in rows: self._d[k] = (bool(a), np.asarray(v, dtype=np.int64).copy())
    def describe(self): return "temporary (this run only)"


class SQLiteKernelStore:
    """Permanent: an append-only SQLite table, WAL so any number of processes read and write at once, INSERT OR IGNORE
    so a point solved by two runs is simply the same row.  `path` may be a database that holds other tables too."""
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with self._con() as con: con.executescript(KERNEL_SCHEMA)

    def _con(self):
        import sqlite3
        con = sqlite3.connect(self.path, timeout=300)
        con.execute("PRAGMA journal_mode=WAL"); con.execute("PRAGMA synchronous=NORMAL"); con.execute("PRAGMA busy_timeout=300000")
        return con

    def get_many(self, keys):
        if not keys: return {}
        out = {}
        with self._con() as con:
            for s in range(0, len(keys), 500):
                chunk = keys[s:s + 500]
                for k, alive, vec in con.execute("SELECT key, alive, vec FROM kernels WHERE key IN (%s)" % ','.join('?' * len(chunk)), chunk):
                    out[bytes(k)] = (bool(alive), np.frombuffer(vec, dtype=np.uint32).astype(np.int64))
        return out

    def put_many(self, rows):
        """One transaction per call, so a batch is stored whole or not at all."""
        if not rows: return
        now = time.time()
        with self._con() as con:
            con.executemany("INSERT OR IGNORE INTO kernels (key, p, ncell, alive, vec, created) VALUES (?,?,?,?,?,?)",
                            [(k, int(p), int(np.asarray(v).shape[0]), int(a), np.asarray(v, dtype=np.uint32).tobytes(), now)
                             for k, p, a, v in rows])

    def describe(self): return "permanent (%s, table kernels)" % os.path.relpath(self.path)


def point_keys(p, q, seq, cells, nrows):
    """The content key of each point: everything point_matrices reads for it."""
    import hashlib
    if seq.ndim == 3:
        matrix = point_matrices(seq, cells, q, p, nrows)
        head = repr(("hologuess-action-kernel-v1", int(p), int(nrows), cells)).encode()
        return [hashlib.sha256(head + int(q[i]).to_bytes(8, 'little') + matrix[i].tobytes()).digest()
                for i in range(len(q))]
    order = max(b for (b, _) in cells)
    head = hashlib.sha256(b'hg-q-kernel-v2|%d|%d|%s' % (int(p), int(nrows), repr(cells).encode())).digest()
    cols = np.ascontiguousarray(np.asarray(seq[:nrows + order], dtype=np.int64).T)
    return [hashlib.sha256(head + int(q[i]).to_bytes(8, 'little') + cols[i].tobytes()).digest() for i in range(q.shape[0])]


def solve_points(p, q, seq, cells, nrows, procs, K, store, verbose=False):
    """Kernel vectors and live mask at every point, reading the store first and writing each task as it completes."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    from ._resources import limit_worker_threads, native_operation
    n, ncell = q.shape[0], len(cells)
    keys = point_keys(p, q, seq, cells, nrows)
    got = store.get_many(keys)
    V = np.zeros((n, ncell), dtype=np.int64); alive = np.zeros(n, dtype=bool); have = np.zeros(n, dtype=bool)
    for i, k in enumerate(keys):
        if k in got: alive[i], V[i] = got[k]; have[i] = True
    todo = np.flatnonzero(~have)
    KERNEL_STATS['reused'] += int(have.sum()); KERNEL_STATS['solved'] += int(todo.shape[0])
    if verbose and have.any():
        print("  p=%d: %d of %d points already solved (kernel store), %d to solve" % (p, int(have.sum()), n, todo.shape[0]), flush=True)
    if todo.shape[0]:
        def task_chunks():
            for s in range(0, todo.shape[0], K):
                indices = todo[s:s + K]
                yield p, q[indices], seq[..., indices], cells, nrows, indices
        def retain(result):
            _, _, vc, al, kdim, idx = result
            if kdim != 1: raise KernelNotALine(kdim)
            V[idx] = vc; alive[idx] = al
            store.put_many([(keys[i], p, al[t], vc[t]) for t, i in enumerate(idx)])
        with native_operation():
            if procs == 1:
                # Identical kernels/chunks and per-task persistence, without
                # creating a process and rescanning its libraries for each prime.
                for task in task_chunks(): retain(solve_points_task(task))
            else:
                tasks = list(task_chunks())
                # BrokenProcessPool still propagates instead of silently hanging.
                with operation_native_limits():
                    with ProcessPoolExecutor(min(procs, len(tasks)), mp_context=get_context('fork'), initializer=limit_worker_threads) as ex:
                        futs = [ex.submit(solve_points_task, t) for t in tasks]
                        for fut in as_completed(futs):
                            try:
                                retain(fut.result())
                            except BaseException:
                                for f in futs: f.cancel()
                                raise
    return V, alive

def clear_denominator(p, qg, Y, ref, hold=24, verbose=True, nwit=8, nkeep=4):
    """When no cell of the operator is a monomial, the ratios u_c / u_ref are rational functions and no band fits.
    Reconstruct their common denominator D (u_ref up to a unit and the content) by rational reconstruction of a few
    witness cells (Cauchy interpolation, verified on held-out points), and multiply it out: (Y * D(q)) is then made of
    Laurent polynomials.  Returns (Y cleared, deg D) or None."""
    from sage.all import GF, PolynomialRing, lcm
    skip = ref if isinstance(ref, (set, frozenset)) else {ref}
    R = PolynomialRing(GF(p), 'x'); x = R.gen(); n = qg.shape[0]; fit, held = n - hold, hold
    if fit < 8: return None
    ncell = Y.shape[1]; step = max(1, ncell // (nwit + 1))
    witness = [c for c in range(0, ncell, step) if c not in skip and np.count_nonzero(Y[:, c]) > 0.9 * n][:nwit]
    D = R(1); used = 0
    for c in witness:
        pts = [(int(qg[i]), int(Y[i, c])) for i in range(fit)]
        F = R.lagrange_polynomial(pts); M = R.prod(x - a for a, _ in pts)
        try: num, den = F.rational_reconstruction(M, fit // 2 - 1, fit // 2 - 1)
        except Exception: continue
        okh = all(den(int(qg[i])) != 0 and num(int(qg[i])) == Y[i, c] * den(int(qg[i])) for i in range(fit, n))
        if not okh: continue
        D = lcm(D, den.monic()); used += 1
        if used >= nkeep: break
    if used == 0: return None
    Dv = np.array([int(D(int(g))) for g in qg], dtype=np.int64)
    if verbose: print("  p=%d: common denominator of degree %d cleared (%d witness cells verified on %d held-out points)" % (p, D.degree(), used, held), flush=True)
    return Y * Dv[:, None] % p, int(D.degree())

def probe_dim(seq_at, cells, p, nrows):
    q, seq = seq_at(p, 2)
    return nullspace_mod(point_matrices(seq[..., :1], cells, q[:1], p, nrows)[0], p).shape[0]

def support_cells(seq_at, cells, p, nrows, npts=4, verbose=True):
    """The cells the operator actually occupies.  A cell whose coefficient is a nonzero Laurent polynomial cannot vanish
    at several independent points, so the cells that are zero in the kernel vector at all of them are zero in the
    operator.  Restricting to this support matters twice over: it shrinks the system, and it makes the reference cell a
    corner of the real Newton polygon (hence a monomial, whose ratios stay Laurent) instead of a corner of the box."""
    q, seq = seq_at(p, npts)
    live = np.zeros(len(cells), dtype=bool)
    for i in range(min(npts, q.shape[0])):
        B = nullspace_mod(point_matrices(seq[..., i:i + 1], cells, q[i:i + 1], p, nrows)[0], p)
        if B.shape[0] != 1: return None
        live |= (B[0] % p) != 0
    keep = [c for c in range(len(cells)) if live[c]]
    if verbose and len(keep) < len(cells):
        print("  support: %d of %d cells carry the operator" % (len(keep), len(cells)), flush=True)
    return keep

def extract(seq_at, cells, primes, band, procs, verbose=True, label='', nrows=None, ref_mode='corner', polys=None,
            min_primes=MIN_PRIMES, max_points=MAX_POINTS, store=None):
    """The annihilator of the sequence inside the given cells, which must be a line at a point.

    A kernel of dimension > 1 is never extracted from (see the module docstring): picking an element of such a box
    returns a non-minimal operator -- 6_3's (8, 31) box gives a 35394-monomial generator where the answer is the
    8988-monomial operator at (7, 38).  The cure is to adapt the bounds, which is what the callers do."""
    nrows = nrows or len(cells) + NUSE_EXTRA
    dim = probe_dim(seq_at, cells, primes[0], nrows)
    if dim > 1: raise KernelNotALine(dim)
    keep = support_cells(seq_at, cells, primes[0], nrows, verbose=verbose)
    if keep and len(keep) < len(cells): cells = [cells[c] for c in keep]
    return extract_one(seq_at, cells, primes, band, procs, verbose, label, nrows, ref_mode, polys, min_primes, max_points,
                       store=store)

@limit_extraction_threads
def extract_one(seq_at, cells, primes, band, procs, verbose=True, label='', nrows=None, ref_mode='corner', polys=None,
                min_primes=MIN_PRIMES, max_points=MAX_POINTS, store=None):
    """seq_at(p, npts) -> (q (npts,), seq (n, npts)) at fresh points for the prime; cells: list of (b, j).
    Returns ({(b, j, e): c} primitive integer operator, info).

    `band` may be None, in which case the starting band is predicted from the sequence's degrees in `polys`; once the
    reference cell has been chosen the prediction is sharpened to that cell and the band jumps there in one step rather
    than crawling outward.  Prediction never replaces the reactive widening below, which is what makes the result
    correct: the envelope is an estimate and can fall short."""
    ncell = len(cells); order = max(b for (b, _) in cells); nrows = nrows or ncell + NUSE_EXTRA
    predicted = False
    store = store if store is not None else MemoryKernelStore()      # the caller chooses; see the kernel store
    if verbose: print("  %s: kernel store %s" % (label, store.describe()), flush=True)
    if band is None:
        band, predicted = predict_band(polys, cells, nrows, ref=None)
        if verbose: print("  %s: starting band [%d, %d] %s" % (label, band[0], band[1],
                          "predicted from the sequence's degrees" if predicted else "(degrees give no prediction)"), flush=True)
    sharpened = not predicted
    ELO, EHI = band; ref = None; refs = None; fails = 0; band0 = band; clear = False; clear_tries = 0; wide_cleared = 0
    primes = list(primes); res_cache = {}; last_band = None
    while True:
        es = list(range(ELO, EHI + 1)); npts = len(es) + 8
        if (ELO, EHI) != last_band: res_cache.clear(); last_band = (ELO, EHI)   # residues are band-specific
        # ... and reference- and denominator-specific: a residue is Y normalised by ONE cell, with or without its
        # denominator cleared, so the cache key carries all of that, not the prime alone
        ckey = lambda pp: (pp, ELO, EHI, ref, clear)
        if npts > max_points:
            # A band this wide with THIS reference means a large denominator; try the next candidate.  `max_points`
            # is the caller's trade-off between one wide round and a round per candidate: for rec_C the points cost
            # engine time and the default is modest; for Ahat they are rec_C extensions and the caller raises it --
            # Ahat(8_6), 1302 cells, needed 2649 points with its first reference and cycled candidates at 2400.
            if clear:
                # the band overflowed with a denominator already cleared: the numerators are what is wide, and they are
                # wide for every reference cell (8_14's rec_C: 433 of 433).  Two such cells settle it; cycling through
                # the rest at full width costs hours and cannot succeed
                wide_cleared += 1
                if wide_cleared >= 2:
                    raise RuntimeError("the numerators need more than %d points per prime with a verified denominator "
                                       "(%d reference cells): raise max_points (--max-points)" % (max_points, wide_cleared))
            if refs is None or len(refs) <= 1: raise RuntimeError("band exceeds %d points with every reference cell" % max_points)
            refs = refs[1:]; ref = refs[0]; ELO, EHI = band0; es = list(range(ELO, EHI + 1)); npts = len(es) + 8; fails = 0
            if verbose: print("  switching the reference cell to (b, j) = %s" % (cells[ref],), flush=True)
        if verbose: print("  %s: %d cells, %d windows, band [%d, %d] (%d points per prime), primes %s" % (label, ncell, nrows, ELO, EHI, npts, primes), flush=True)
        residues = []; t0 = time.time(); widen = False; prev_cand = None; settled = None
        for p in primes:
            if ckey(p) in res_cache: residues.append(res_cache[ckey(p)]); continue   # this band, this normalisation
            # Points beyond the band's need.  The interpolation holds out eight; the rest are for best_reference
            # and clear_denominator, which reconstruct witness RATIOS u_w / u_ref as rational functions and can only
            # reach a numerator of degree about half the point count.  Those numerators span up to the band, so the
            # headroom has to scale with the band: a flat 32 (tried in v1.4.1) left the witnesses that carried 8_5's
            # denominator out of reach, and the measurement, seeing only the small ones, called the reference a
            # monomial.  15% is what the measurements that worked had.
            q, seq = seq_at(p, npts + max(32, int(npts * OVERSAMPLE)))
            if seq.shape[0] < nrows + (order if seq.ndim == 2 else 0): raise ValueError("need %d terms at the points, have %d" % (nrows + order, seq.shape[0]))
            K = points_per_task(nrows, ncell, procs)
            if K < 16 and verbose and p == primes[0]: print("  %d points per task: %d workers at %d x %d need about %.2f GB each" % (
                K, procs, nrows, ncell, (5 * K * nrows * ncell * 8 + MEM_WORKER) / 2 ** 30), flush=True)
            # every point is looked up in the kernel store first and written to it as its task completes, so an
            # interrupted extraction resumes where it stopped instead of re-solving (see KernelStore)
            Vall, alive = solve_points(p, q, seq, cells, nrows, procs, K, store, verbose=verbose)
            qg = q[alive]; V = Vall[alive]
            if qg.shape[0] < npts: raise RuntimeError("only %d/%d good points at p=%d" % (qg.shape[0], npts, p))
            if ref is None:
                cands = ref_candidates(V[0], cells, p)
                if ref_mode == 'measured':
                    # Look past the first sixteen candidates when none of them is a monomial.  A reference whose
                    # witness denominators have degree 0 IS a monomial of the operator, and it makes the ratios
                    # Laurent, which avoids denominator clearing and the band oscillation that follows it -- 7_7's
                    # Ahat settles for a degree-1 reference and spends most of its time on exactly that.  The scan
                    # stops at the first degree-0 candidate, so widening the list is free when one is found early
                    # and bounded when it is not.
                    ref, sc = best_reference(p, qg, V, cands[:16], cells, verbose)
                    if sc: ref, sc = best_reference(p, qg, V, cands[:64], cells, verbose)
                    # a reference measured to give Laurent ratios (denominator degree 0) is a monomial of the operator:
                    # there is then nothing to gain from switching, and every switch throws the band away and restarts
                    refs = [ref] if sc == 0 else [ref] + [c for c in cands if c != ref]
                    if sc:
                        # The measurement already says this reference is NOT a monomial: its witness ratios share a
                        # denominator of degree sc.  Solving at the plan's band cannot succeed (the ratios are not
                        # Laurent), and the old flow answered that failure with a blind 25% widening before clearing
                        # anything -- 8_5 computed 194 surplus points, two hours of engine time, in exactly that round.
                        # Clear the denominator from the start and make room for the numerators, which is the band
                        # the extraction settles on anyway; and make sure there are enough points to reconstruct a
                        # denominator of that degree at all.
                        clear = True
                        lo, hi = min(ELO, band0[0] - sc), max(EHI, band0[1] + sc)
                        short = 2 * sc + 8 - (hi - lo + 1)
                        if short > 0: lo -= (short + 1) // 2; hi += (short + 1) // 2
                        if (lo, hi) != (ELO, EHI):
                            ELO, EHI = lo, hi; widen = 'reset'
                            if verbose: print("  measured denominator of degree %d: clearing it and starting at band [%d, %d] (%d points per prime)"
                                              % (sc, ELO, EHI, EHI - ELO + 9), flush=True)
                            break
                else:
                    refs = cands; ref = refs[0]
                    if verbose: print("  reference cell (b, j) = %s" % (cells[ref],), flush=True)
                if ref_mode == 'rational': clear = True
            if not sharpened:
                # the band the ratios really occupy depends on which cell normalises them, and that is now known
                sharpened = True
                nb, ok_pred = predict_band(polys, cells, nrows, ref=cells[ref])
                if ok_pred and (nb[0] < ELO or nb[1] > EHI):
                    ELO, EHI = min(ELO, nb[0]), max(EHI, nb[1]); band0 = (ELO, EHI); widen = 'reset'
                    if verbose: print("  band sharpened for reference (b, j) = %s: [%d, %d] (%d points per prime)"
                                      % (cells[ref], ELO, EHI, EHI - ELO + 9), flush=True)
                    break
            keep = V[:, ref] % p != 0
            qg = qg[keep]; V = V[keep]
            if qg.shape[0] < npts: raise RuntimeError("the reference cell vanishes at too many points at p=%d" % p)
            Y = V * inv_mod(V[:, ref], p)[:, None] % p
            if clear:
                cl = clear_denominator(p, qg, Y, ref, verbose=verbose)
                if cl is None:
                    if verbose: print("  p=%d: denominator reconstruction failed" % p, flush=True)
                    widen = True; fails += 1; break
                Y, degD = cl
                if degD > EHI - band0[1]:           # first time (or a larger denominator): make room for the numerators
                    ELO, EHI = band0[0] - degD, band0[1] + degD; widen = 'reset'; break
            # Eight held-out points, not the whole oversample.  Passing every solved point makes the residual
            # check stricter, but a wrong fit already survives eight random points with probability about p^-8,
            # which is 10^-75 here -- there is no confidence left to buy, and the extra residual arithmetic is
            # not free.  (Measured: it was not the cause of the 7_7 slowdown either, so it buys nothing at all.)
            X, ok = vandermonde_solve(qg[:npts], Y[:npts], es, p)
            if not ok and not clear:
                # The ratios are not Laurent with this reference cell -- it is not a monomial of the operator, or the
                # band is short.  Try to reconstruct the denominator they share from the points in hand and clear it
                # BEFORE any widening: a widening costs engine points, this costs seconds, and when it succeeds the
                # band is reset to exactly where the numerators live.  (It used to wait for a second failure, and
                # the blind 25% widening in between computed 194 surplus points for 8_5.)  If it fails, the points
                # in hand cannot show the denominator -- reconstruction needs about twice its degree plus the
                # numerator's -- and widening is the only way to get more.
                cl = clear_denominator(p, qg, Y, ref, verbose=verbose)
                if cl is not None:
                    clear = True; _, degD = cl; ELO, EHI = band0[0] - degD, band0[1] + degD; widen = 'reset'; fails = 0; break
                if clear_tries < 3:
                    clear_tries += 1; w = (EHI - ELO); ELO, EHI = ELO - w, EHI + w; widen = True
                    if verbose: print("  p=%d: the denominator does not fit in %d points: doubling the band" % (p, npts), flush=True)
                    break
            if not ok:
                fails += 1
                # With the denominator cleared -- reconstructed from the points and verified on held-out ones -- a
                # failed interpolation means the numerators are wider than the band, which another reference cell does
                # not change: keep widening, up to max_points.  Switching here sent 8_14's rec_C through all 433 cells
                # at bands of at most 5798 points for 9.5 hours, when its numerators needed more.
                if fails >= 3 and not clear and refs is not None and len(refs) > 1:
                    # this reference cell is hopeless: next candidate, band and denominator state reset
                    refs = refs[1:]; ref = refs[0]; ELO, EHI = band0; fails = 0; clear = False; clear_tries = 0
                    if verbose: print("  p=%d: interpolation keeps failing: switching the reference cell to (b, j) = %s" % (p, cells[ref]), flush=True)
                    widen = 'reset'; break
                if verbose: print("  p=%d: interpolation does not extend to the spare points: widening the band" % p, flush=True)
                widen = True; break
            residues.append(X); res_cache[ckey(p)] = X
            if verbose: print("  p=%d: %d points solved and interpolated (%.0fs)" % (p, npts, time.time() - t0), flush=True)
            # Early termination.  Solving one prime's points is the dominant cost of a round, so stop as soon as the
            # answer has stopped changing rather than always paying for the requested number.  The test: rational
            # reconstruction succeeds on every coefficient, the support clears the band guard, and the operator is
            # IDENTICAL to the one reconstructed from one prime fewer, with at least min_primes primes.  A single
            # prime cannot be trusted here (see reconstruct_operator: a coefficient beyond the ceiling reconstructs
            # to a wrong small rational 61% of the time, and a run that stopped on it ended with a failed
            # certificate and no way to retry); two moduli agreeing on the same wrong rational is what is
            # negligible.  The caller still certifies the operator exactly afterwards.
            cand, cf, (bd, Mt) = reconstruct_operator(residues, primes[:len(residues)], cells, es)
            sp = [e for (_, _, e) in cand]
            if cand and cf == 0 and min(sp) >= ELO + GUARD and max(sp) <= EHI - GUARD and \
                    len(residues) >= min_primes and prev_cand == cand:
                settled = (cand, cf)
                if verbose and len(residues) < len(primes):
                    print("  reconstruction settled: %d primes agree (height 2^%.0f of a 2^%.0f modulus); the exact"
                          " certificate still decides" % (len(residues), math.log2(max(bd, 1)), math.log2(Mt)), flush=True)
                break
            prev_cand = cand
        if not widen:
            # Interpolation succeeded for every prime this round, so the reference cell is doing its job: the
            # strike count starts again.  `fails` counts CONSECUTIVE failed rounds, and it drives two escalations
            # (clear the denominator at 1, abandon the reference cell at 3) that must not be reached by adding up
            # unrelated failures from before a widening that fixed them.  The old code got this reset by accident,
            # via reconstruct_operator's unrelated return value landing in the same name; removing that clobber
            # without restoring the reset cost 10-39x on 7_5, 7_6 and 7_7, and scoping the counter per band
            # instead removed the escalation altogether and made 7_7 widen without limit.
            fails = 0
            if settled is not None: u, rrfails = settled     # the agreeing reconstruction, not recomputed
            else: u, rrfails, _ = reconstruct_operator(residues, primes[:len(residues)], cells, es)  # NOT `fails`:
            supp = [e for (_, _, e) in u]                     # that counts interpolation strikes, read next round
            if u and min(supp) >= ELO + GUARD and max(supp) <= EHI - GUARD and rrfails == 0: break
            if rrfails and u and min(supp) >= ELO + GUARD and max(supp) <= EHI - GUARD:
                # Rational reconstruction failed but the band is fine.  These are different failures and the cure
                # is different: ratrec needs |numerator| and |denominator| below sqrt(M/2), so it fails when the
                # CRT modulus is too small for the coefficients -- MORE PRIMES, not more points.  Widening the band
                # here (which is what this did) buys nothing and costs a whole round of point solving.
                nxt = [pp for pp in PRIMES if pp not in primes]
                if nxt:
                    primes.append(nxt[0])
                    if verbose: print("  %d rational reconstructions failed with %d primes (%.0f bits): adding p=%d"
                                      % (rrfails, len(primes) - 1, sum(math.log2(x) for x in primes[:-1]), nxt[0]), flush=True)
                    continue
                # Widening the band cannot help a reconstruction that fails for size, and re-solves every prime.
                raise RuntimeError("%d rational reconstructions still fail with all %d primes (%.0f bits): the "
                                   "coefficients are larger than the prime list reaches, or the residues are on "
                                   "different normalisations; extend PRIMES or inspect the normalisation"
                                   % (rrfails, len(primes), sum(math.log2(x) for x in primes)))
            if verbose: print("  support [%d, %d] touches the band guard or %d rational reconstructions failed: widening" % (min(supp) if supp else 0, max(supp) if supp else 0, rrfails), flush=True)
        if widen == 'reset': continue
        # with a reference known to be a monomial the only thing missing is width, so grow faster than the cautious step
        w = max(16, int(len(es) * (0.45 if (refs is not None and len(refs) == 1) else 0.25)))
        ELO, EHI = ELO - w, EHI + w
    info = {'band': [ELO, EHI], 'points_per_prime': npts, 'primes': primes[:len(residues)], 'ref': cells[ref], 'terms': len(u),
            'denominator_cleared': clear, 'ref_mode': ref_mode, 'band_predicted': predicted}
    return u, info


# ----------------------------------------------------------------------------------------------------------------------
# the sequence the solver is handed, and the four stages run over it
# ----------------------------------------------------------------------------------------------------------------------
class Sequence:
    """What the solver needs of a q-holonomic sequence.  Subclasses decide where the values come from; the solver never
    asks what they mean.

      at(p, npts)  -> (q (K,), s (n, K)) values at K >= npts points over GF(p), each column one specialisation
      terms        -> how many terms s(0..terms-1) are currently reachable at a point
      grow(n)      -> make n terms reachable and return how many were achieved (None if the source cannot grow)
      cap(n)       -> stop producing more than n terms per point (the extraction usually needs far fewer than
                      detection reached, and for a costly source that difference is the whole bill)
      polys        -> the exact Laurent polynomials {e: c} of the terms held, or None
    """
    def at(self, p, npts): raise NotImplementedError
    @property
    def terms(self): raise NotImplementedError
    def grow(self, n): return None
    def cap(self, n): pass
    @property
    def polys(self): return None


class ArraySequence(Sequence):
    """A sequence backed by a callable, for callers that already have one."""
    def __init__(self, fn, terms, polys=None, grow=None):
        self._fn = fn; self._terms = terms; self._polys = polys; self._grow = grow; self._cap = terms
    def at(self, p, npts): return self._fn(p, npts)
    @property
    def terms(self): return self._terms
    def grow(self, n):
        if self._grow is None: return None
        self._terms = self._grow(n); self._cap = self._terms; return self._terms
    def cap(self, n): self._cap = min(self._terms, max(n, 2))
    @property
    def polys(self): return self._polys


def detect(seq, p, max_order=12, max_degree=0, grow_to=0, verbose=True):
    """Stage one: the corner (order, degree) whose kernel is a line, growing the sequence when the search runs out of
    terms.  Detection needs two points only, so growth here is cheap: it extends those two, not the whole point set.
    Returns (corner, q2, s2, proved, gaps, multi)."""
    while True:
        q2, s2 = seq.at(p, 2)
        corner, proved, gaps, multi = corner_search(s2[:, :2], q2[:2], p, max_order, max_degree, verbose)
        if corner is not None or not grow_to: return corner, q2, s2, proved, gaps, multi
        need = min((g[2] for g in gaps if isinstance(g[2], int)), default=0)
        if need <= seq.terms or need > grow_to:
            if verbose: print("  term-limited: the next untested box needs %d terms%s" % (need, '' if need <= grow_to else
                              " (beyond the cap of %d)" % grow_to), flush=True)
            return corner, q2, s2, proved, gaps, multi
        step = max(need + 20, int(seq.terms * 1.25))
        if verbose: print("  term-limited at %d: the next untested box needs %d; growing to %d" % (seq.terms, need, step), flush=True)
        if seq.grow(step) is None: return corner, q2, s2, proved, gaps, multi


def restrict(corner, q2, s2, p, terms, windows_extra=NUSE_EXTRA, verbose=True):
    """Stage two: the cells inside the corner box that the operator actually occupies, and the system size that follows.

    The corner is a rectangle but the operator occupies only part of it, and it is the occupied part that decides how
    many windows the system needs -- hence how many terms EVERY point must be carried to, which is the whole cost of the
    extraction.  The two detection points already reach far enough to see the support, so it is found here, before
    anything is sized: 8_5's box of 703 cells holds an operator on 392 of them, the difference between needing 781 terms
    at every point and needing about 470.  Returns (cells, nrows, terms_needed)."""
    r, d = corner[0], corner[1]
    cells = [(b, j) for b in range(r + 1) for j in range(d + 1)]
    full_rows = min(terms - r, len(cells) + windows_extra)
    live = np.zeros(len(cells), dtype=bool); ok = True
    for i in range(min(2, q2.shape[0])):
        B = nullspace_mod(point_matrices(s2[:, i:i + 1], cells, q2[i:i + 1], p, full_rows)[0], p)
        if B.shape[0] != 1: ok = False; break
        live |= (B[0] % p) != 0
    if ok and 0 < int(live.sum()) < len(cells):
        cells = [cells[c] for c in range(len(cells)) if live[c]]
        if verbose: print("  support: %d of %d cells carry the operator" % (len(cells), (r + 1) * (d + 1)), flush=True)
    nrows = min(terms - r, len(cells) + windows_extra)
    return cells, nrows, min(terms, nrows + r + 2)


def plan(seq, cells, nrows, verbose=True):
    """Stage three: what the extraction will cost, before it is paid.

    The band is predicted from the sequence's own degrees (see `envelope`), which turns the number of points from
    something discovered by repeated widening -- each widening recomputing terms at more points -- into something known
    up front.  It is an estimate, so this reports rather than binds."""
    band, predicted = predict_band(seq.polys, cells, nrows, ref=None)
    npts = band[1] - band[0] + 9
    if verbose:
        if predicted: print("  plan: %d cells, %d windows, predicted band [%d, %d] -> about %d points per prime"
                            % (len(cells), nrows, band[0], band[1], npts), flush=True)
        else: print("  plan: %d cells, %d windows; the degrees give no band prediction, starting at +-%d"
                    % (len(cells), nrows, FALLBACK_BAND), flush=True)
    return band, npts, predicted


def solve(seq, primes, procs=8, max_order=12, max_degree=0, band=None, ref_mode='measured',
          windows_extra=NUSE_EXTRA, grow_to=0, verbose=True, label='', min_primes=MIN_PRIMES, max_points=MAX_POINTS,
          store=None):
    """The whole pipeline over one sequence: detect the corner, restrict to the support, plan the band, extract.

    Returns (u, info) with u = {(b, j, e): integer coefficient}, or (None, info) when no corner is found, in which case
    info carries what was proved empty and where the data ran out."""
    p = primes[0]
    corner, q2, s2, proved, gaps, multi = detect(seq, p, max_order, max_degree, grow_to, verbose)
    if corner is None:
        return None, {'corner': None, 'proved_empty': proved, 'gaps': gaps, 'higher_dimensional': multi}
    r, d, _ = corner
    cells, nrows, terms_needed = restrict(corner, q2, s2, p, seq.terms, windows_extra, verbose)
    if terms_needed < seq.terms:
        if verbose: print("  the extraction needs %d terms per point, not %d" % (terms_needed, seq.terms), flush=True)
        seq.cap(terms_needed)
    pband, npts, predicted = plan(seq, cells, nrows, verbose)
    u, info = extract_one(seq.at, cells, primes, band if band is not None else pband, procs, verbose,
                          label or 'operator order %d degree %d' % (r, d), nrows, ref_mode, seq.polys, min_primes, max_points,
                          store=store)
    # the band WAS predicted, by plan() above; extract_one only sets this when it predicts one itself.  (Letting
    # extract_one re-sharpen on the reference cell was measured and is a wash or worse: 7_5 went 72.6s -> 96.0s.)
    info['band_predicted'] = bool(predicted)
    info |= {'corner': [r, d], 'cells': len(cells), 'windows': nrows, 'terms_needed': terms_needed,
             'proved_empty': proved, 'predicted_points': npts}
    return u, info
