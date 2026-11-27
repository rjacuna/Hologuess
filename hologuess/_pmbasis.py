#!/usr/bin/env python3
"""
Recurrence fitting as a Hermite-Pade / minimal approximant basis problem over GF(p)  (PM-Basis in numpy).

The ansatz  sum_{(b,j) in cells} u_{bj}(q) q^{jk} J[k+b](q) = 0  for windows k = 0..W-1  says that the vector
(u_c(q))_c of polynomials is a left kernel vector of the m x W polynomial matrix F with F[c, k] = q^{jk} J[k+b]
(each column normalised to non-negative exponents).  A minimal approximant basis P (m x m) of F at order
sigma > deg F + deg u contains the exact kernel vectors of small degree as its lowest-degree rows, with no
guess about the q-support of the u_c needed (Giorgi-Jeannerod-Villard M-Basis / PM-Basis; NTT-based polynomial
matrix products with 16-bit-limb float64 BLAS matmuls; NTT primes below 2^31).

    sage -python pmbasis.py JSON ORDER --dmax=D [--cells=FILE] --windows=W [--sigma-extra=E] [--prime=i] [--out=FILE]
"""
import sys, json, time, re, os
import numpy as np

# NTT primes p = c 2^k + 1 < 2^31 with k >= 21 and a primitive root
NTT_PRIMES = [(2013265921, 31), (1811939329, 13), (469762049, 3), (2113929217, 5), (1711276033, 29),
              (167772161, 3), (754974721, 11), (1224736769, 3), (998244353, 3), (897581057, 3), (985661441, 3),
              (943718401, 7), (935329793, 3), (924844033, 5), (1004535809, 3)]

def inv_vec(x, p):
    e = p - 2; r = np.ones_like(x); b = x % p
    while e:
        if e & 1: r = (r * b) % p
        b = (b * b) % p; e >>= 1
    return r

# ----------------------------------------------------------------------------------------------
# backend: numpy (threads over chunks) or cupy (PM_BACKEND=cupy; batched fp64 GEMMs on cuBLAS, elementwise NTT on the GPU,
# the stored right-factor transforms kept resident on the device when they fit PM_GPU_MEM bytes)
BACKEND = os.environ.get('PM_BACKEND', 'numpy').lower()
MLXB = BACKEND == 'mlx'
if BACKEND == 'cupy':
    import cupy as xp
    GPU = True
    def dev(a): return a if isinstance(a, xp.ndarray) else xp.asarray(a)
    def host(a): return xp.asnumpy(a) if isinstance(a, xp.ndarray) else a
elif MLXB:
    import mlx.core as xp
    GPU = True                                   # device-resident products; the Apple GPU (unified memory)
    def dev(a): return a if isinstance(a, xp.array) else xp.array(np.ascontiguousarray(a))
    def host(a): return np.array(a) if isinstance(a, xp.array) else a
else:
    xp = np; GPU = False
    def dev(a): return a
    def host(a): return a
GPU_MEM = int(float(os.environ.get('PM_GPU_MEM', '6e9')))       # bytes of device memory a resident transform may take
NTHREADS = int(os.environ.get('PM_THREADS', '8'))
MEM_ENTRIES = int(float(os.environ.get('PM_MEM', '2.5e8')))      # int64 entries a resident (right-factor) transform may take
WORK_ENTRIES = int(float(os.environ.get('PM_WORK', '4e7')))       # int64 entries per streamed chunk (left factor / column chunk)
from concurrent.futures import ThreadPoolExecutor
_POOL = [None]
def pool():
    if _POOL[0] is None: _POOL[0] = ThreadPoolExecutor(NTHREADS)
    return _POOL[0]
def budget_entries():
    """entries (8 bytes) a resident transform buffer may take on the compute device"""
    return GPU_MEM // 8 if GPU else MEM_ENTRIES

def _mm(p, X, Y, mod=np):
    """(…,a,b)@(…,b,c) over GF(p), p < 2^31, b <= 2^12: three exact float64 matmuls on 16-bit limbs (numpy or cupy arrays)."""
    Xh = (X >> 16).astype(mod.float64); Xl = (X & 0xFFFF).astype(mod.float64)
    Yh = (Y >> 16).astype(mod.float64); Yl = (Y & 0xFFFF).astype(mod.float64)
    Phh = Xh @ Yh; Pll = Xl @ Yl; Pm = (Xh + Xl) @ (Yh + Yl) - Phh - Pll
    Phh = Phh.astype(mod.int64) % p; Pll = Pll.astype(mod.int64) % p; Pm = Pm.astype(mod.int64) % p
    return (Phh * ((1 << 32) % p) + Pm * 65536 + Pll) % p

def matmul_mod(p, X, Y, lchunk=None):
    """Host (numpy) batched (L,a,b)@(L,b,c) (or 2-D, or 2-D x 3-D) over GF(p); L-chunks in parallel threads."""
    if X.ndim == 2 and Y.ndim == 2:
        return _mm(p, X, Y)
    L = X.shape[0] if X.ndim == 3 else Y.shape[0]
    if lchunk is None: lchunk = max(1, int(2e7 // (X.shape[-2] * X.shape[-1] + Y.shape[-2] * Y.shape[-1] + 1)))
    out = np.empty((L, X.shape[-2], Y.shape[-1]), dtype=np.int64)
    def work(s):
        xs = X[s:s + lchunk] if X.ndim == 3 else X
        ys = Y[s:s + lchunk] if Y.ndim == 3 else Y
        out[s:s + lchunk] = _mm(p, xs, ys)
    list(pool().map(work, range(0, L, lchunk)))
    return out

def _mm_mlx(p, X, Y):
    """(L,a,b)@(L,b,c) over GF(p), p < 2^31, on the Apple GPU: 8-bit limbs multiplied in fp32 with the inner dimension
    blocked at 256 (partial sums < 2^24, exact), recombined mod p in int64."""
    mx = xp; b = X.shape[-1]
    Xl = [((X >> (8 * i)) & 255).astype(mx.float32) for i in range(4)]
    Yl = [((Y >> (8 * j)) & 255).astype(mx.float32) for j in range(4)]
    acc = None
    for k0 in range(0, b, 256):
        k1 = min(k0 + 256, b)
        for i in range(4):
            for j in range(4):
                P = (Xl[i][..., :, k0:k1] @ Yl[j][..., k0:k1, :]).astype(mx.int64) % p
                term = (P * ((1 << (8 * (i + j))) % p)) % p
                acc = term if acc is None else (acc + term) % p
    mx.eval(acc)
    return acc

def mm_dev(p, X, Y):
    """Device-side batched (L,a,b)@(L,b,c): cupy arrays on the GPU backend (chunked over L to bound temporaries),
    mlx arrays with the fp32 limb product on the mlx backend, the threaded host product on the numpy backend."""
    if not GPU: return matmul_mod(p, X, Y)
    if MLXB:
        L = X.shape[0]; a, b = X.shape[1], X.shape[2]; c = Y.shape[2]
        lchunk = max(1, int(budget_entries() // (12 * (a * b + b * c + a * c))))
        parts = [_mm_mlx(p, X[s:s + lchunk], Y[s:s + lchunk]) for s in range(0, L, lchunk)]
        out = parts[0] if len(parts) == 1 else xp.concatenate(parts, axis=0)
        xp.eval(out); return out
    L = X.shape[0]; a, b = X.shape[1], X.shape[2]; c = Y.shape[2]
    lchunk = max(1, int(budget_entries() // (6 * (a * b + b * c + a * c))))
    out = xp.empty((L, a, c), dtype=xp.int64)
    for s in range(0, L, lchunk):
        out[s:s + lchunk] = _mm(p, X[s:s + lchunk], Y[s:s + lchunk], xp)
    return out

NTT_MODE = os.environ.get('PM_NTT', 'fused')     # fused: CUDA kernel on cupy, compiled C kernel on numpy; 'mlx': Apple GPU; 'elementwise': plain array ops
_NTT_CUDA_SRC = r"""
extern "C" __global__
void ntt_stage(long long* a, const long long* ws, const unsigned long long p, const unsigned long long mu,
               const int M, const int half, const int step, const unsigned long long n)
{
    unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int col = (int)(idx % (unsigned long long)M);
    unsigned long long pair = idx / (unsigned long long)M;
    unsigned long long blk = pair / (unsigned long long)half;
    int j = (int)(pair % (unsigned long long)half);
    unsigned long long lo = (blk * 2ULL * (unsigned long long)half + (unsigned long long)j) * (unsigned long long)M + (unsigned long long)col;
    unsigned long long hi = lo + (unsigned long long)half * (unsigned long long)M;
    unsigned long long w = (unsigned long long)ws[(unsigned long long)j * (unsigned long long)step];
    unsigned long long x = (unsigned long long)a[hi];
    unsigned long long prod = x * w;                       /* < 2^62 for p < 2^31 */
    unsigned long long q = __umul64hi(prod, mu);           /* Barrett: mu = floor(2^64 / p) */
    unsigned long long t = prod - q * p;
    if (t >= p) t -= p;
    if (t >= p) t -= p;
    unsigned long long u = (unsigned long long)a[lo];
    unsigned long long s = u + t; if (s >= p) s -= p;
    unsigned long long d = u + p - t; if (d >= p) d -= p;
    a[lo] = (long long)s; a[hi] = (long long)d;
}
extern "C" __global__
void scale_mod(long long* a, const unsigned long long c, const unsigned long long p, const unsigned long long mu, const unsigned long long n)
{
    unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned long long prod = (unsigned long long)a[idx] * c;
    unsigned long long q = __umul64hi(prod, mu);
    unsigned long long t = prod - q * p;
    if (t >= p) t -= p;
    if (t >= p) t -= p;
    a[idx] = (long long)t;
}
"""

_NTT_C_SRC = r"""
#include <stdint.h>
typedef unsigned __int128 u128;
static inline uint64_t mulmod(uint64_t x, uint64_t w, uint64_t p, uint64_t mu) {
    uint64_t prod = x * w;                                   /* < 2^62 for p < 2^31 */
    uint64_t q = (uint64_t)(((u128)prod * mu) >> 64);        /* Barrett: mu = floor(2^64 / p) */
    uint64_t t = prod - q * p;
    if (t >= p) t -= p;
    if (t >= p) t -= p;
    return t;
}
/* one butterfly stage on columns [c0, c1) of an (L, M) row-major int64 array: block length 2*half, twiddle stride step */
void ntt_stage(int64_t* a, const int64_t* ws, uint64_t p, uint64_t mu, int64_t L, int64_t M, int64_t half, int64_t step, int64_t c0, int64_t c1) {
    for (int64_t blk = 0; blk < L / (2 * half); blk++) {
        for (int64_t j = 0; j < half; j++) {
            uint64_t w = (uint64_t)ws[j * step];
            int64_t* lo = a + (blk * 2 * half + j) * M;
            int64_t* hi = lo + half * M;
            for (int64_t c = c0; c < c1; c++) {
                uint64_t t = mulmod((uint64_t)hi[c], w, p, mu);
                uint64_t u = (uint64_t)lo[c];
                uint64_t s = u + t; if (s >= p) s -= p;
                uint64_t d = u + p - t; if (d >= p) d -= p;
                lo[c] = (int64_t)s; hi[c] = (int64_t)d;
            }
        }
    }
}
void scale_mod(int64_t* a, uint64_t c, uint64_t p, uint64_t mu, int64_t n0, int64_t n1) {
    for (int64_t i = n0; i < n1; i++) a[i] = (int64_t)mulmod((uint64_t)a[i], c, p, mu);
}
"""
_MLX = [None]
def _load_mlx():
    """MLX (Apple GPU) butterfly kernels; None if mlx is not importable or no GPU."""
    if _MLX[0] is not None: return _MLX[0] or None
    try:
        import mlx.core as mx
        header = r"""
        inline ulong mulhi64(ulong a, ulong b) {
            ulong a0 = a & 0xffffffffUL, a1 = a >> 32, b0 = b & 0xffffffffUL, b1 = b >> 32;
            ulong p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
            ulong mid = (p00 >> 32) + (p01 & 0xffffffffUL) + (p10 & 0xffffffffUL);
            return p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
        }
        inline ulong mulmod64(ulong x, ulong w, ulong p, ulong mu) {
            ulong prod = x * w;                       /* < 2^62 for p < 2^31 */
            ulong q = mulhi64(prod, mu);              /* Barrett, mu = floor(2^64 / p) */
            ulong t = prod - q * p;
            if (t >= p) t -= p;
            if (t >= p) t -= p;
            return t;
        }
        """
        stage_src = r"""
        uint idx = thread_position_in_grid.x;
        ulong n = (ulong)prm[5];
        if ((ulong)idx >= n) return;
        ulong p = (ulong)prm[0]; ulong mu = (ulong)prm[1]; ulong hf = (ulong)prm[2]; ulong stp = (ulong)prm[3]; ulong mc = (ulong)prm[4];
        ulong col = idx % mc; ulong pair = idx / mc; ulong blk = pair / hf; ulong j = pair % hf;
        ulong lo = (blk * 2 * hf + j) * mc + col; ulong hi = lo + hf * mc;
        ulong w = (ulong)ws[j * stp];
        ulong t = mulmod64((ulong)a[hi], w, p, mu);
        ulong u = (ulong)a[lo];
        ulong s = u + t; if (s >= p) s -= p;
        ulong d = u + p - t; if (d >= p) d -= p;
        out[lo] = (int64_t)s; out[hi] = (int64_t)d;
        """
        scale_src = r"""
        uint idx = thread_position_in_grid.x;
        ulong n = (ulong)prm[5];
        if ((ulong)idx >= n) return;
        out[idx] = (int64_t)mulmod64((ulong)a[idx], (ulong)prm[2], (ulong)prm[0], (ulong)prm[1]);
        """
        stage = mx.fast.metal_kernel(name="ntt_stage", input_names=["a", "ws", "prm"], output_names=["out"], source=stage_src, header=header)
        scale = mx.fast.metal_kernel(name="ntt_scale", input_names=["a", "prm"], output_names=["out"], source=scale_src, header=header)
        _MLX[0] = (mx, stage, scale)
    except Exception as ex:
        _MLX[0] = False
        if os.environ.get('PM_DEBUG'): print("MLX NTT kernel unavailable:", ex)
    return _MLX[0] or None

_NTT_C = [None]
def _load_ntt_c():
    """Compile (once, cached in ~/.cache/jonesrec) and load the C butterfly kernel; None if no compiler."""
    if _NTT_C[0] is not None: return _NTT_C[0] or None
    try:
        import ctypes, hashlib, subprocess, tempfile
        cache = os.path.join(os.path.expanduser('~'), '.cache', 'jonesrec'); os.makedirs(cache, exist_ok=True)
        h = hashlib.sha1(_NTT_C_SRC.encode()).hexdigest()[:12]
        so = os.path.join(cache, 'ntt_stage_%s.so' % h)
        if not os.path.exists(so):
            csrc = os.path.join(cache, 'ntt_stage_%s.c' % h); open(csrc, 'w').write(_NTT_C_SRC)
            subprocess.run(['cc', '-O3', '-march=native', '-shared', '-fPIC', '-o', so, csrc], check=True, capture_output=True)
        lib = ctypes.CDLL(so)
        i64p = ctypes.POINTER(ctypes.c_int64)
        lib.ntt_stage.argtypes = [i64p, i64p, ctypes.c_uint64, ctypes.c_uint64] + [ctypes.c_int64] * 6; lib.ntt_stage.restype = None
        lib.scale_mod.argtypes = [i64p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int64, ctypes.c_int64]; lib.scale_mod.restype = None
        _NTT_C[0] = lib
    except Exception as ex:
        _NTT_C[0] = False
        if os.environ.get('PM_DEBUG'): print("C NTT kernel unavailable:", ex)
    return _NTT_C[0] or None

def barrett_stage_reference(a, ws, p, half, step):
    """Pure-Python mirror of the fused kernel's index and Barrett arithmetic, for testing on the CPU: a (L, M) int64."""
    L, M = a.shape; mu = (1 << 64) // p; n = (L // 2) * M
    out = a.copy()
    for idx in range(n):
        col = idx % M; pair = idx // M; blk = pair // half; j = pair % half
        lo = (blk * 2 * half + j) * M + col; hi = lo + half * M
        w = int(ws[j * step]); x = int(a.flat[hi]); prod = x * w
        q = (prod * mu) >> 64; t = prod - q * p
        if t >= p: t -= p
        if t >= p: t -= p
        u = int(a.flat[lo]); s = u + t
        if s >= p: s -= p
        d = u + p - t
        if d >= p: d -= p
        out.flat[lo] = s; out.flat[hi] = d
    return out

class NTT(object):
    def __init__(self, p, g):
        self.p = p; self.g = g; self.cache = {}
        self.mu = (1 << 64) // p
        self.kern = None
        if BACKEND == 'cupy' and NTT_MODE == 'fused':
            mod = xp.RawModule(code=_NTT_CUDA_SRC)
            self.kern = (mod.get_function('ntt_stage'), mod.get_function('scale_mod'))
    def roots(self, L, inverse):
        key = (L, inverse)
        if key not in self.cache:
            p = self.p
            if (p - 1) % L: raise ValueError(
                "NTT length %d does not divide p-1 for p=%d (2-adic valuation %d): the root would be wrong and the "
                "transform would return silently incorrect values" % (L, p, (p - 1 & -(p - 1)).bit_length() - 1))
            w = pow(self.g, (p - 1) // L, p)
            if inverse: w = pow(w, p - 2, p)
            ws = np.empty(L, dtype=np.int64); ws[0] = 1
            for i in range(1, L): ws[i] = (ws[i - 1] * w) % p
            rev = np.zeros(L, dtype=np.int64); bits = L.bit_length() - 1
            for i in range(L):
                r = 0; x = i
                for _ in range(bits): r = (r << 1) | (x & 1); x >>= 1
                rev[i] = r
            self.cache[key] = (dev(ws), dev(rev))
        return self.cache[key]
    def transform_dev(self, a, inverse=False):
        """NTT along axis 0 of an (L, ...) array on the compute device (numpy: host, column blocks in threads)."""
        shp = a.shape[1:]; L = a.shape[0]
        if MLXB:
            return self._transform_mlx_dev(dev(a), inverse).reshape(L, *shp)
        a2 = a.reshape(L, -1); M = a2.shape[1]
        if not GPU and ((NTT_MODE in ('fused', 'c') and _load_ntt_c() is not None) or (NTT_MODE == 'mlx' and _load_mlx() is not None)):
            return self._transform1(a2, inverse).reshape(L, *shp)          # the compiled kernels parallelise themselves
        if not GPU and M >= 4 * NTHREADS and L * M > 200000:
            ch = (M + NTHREADS - 1) // NTHREADS
            parts = list(pool().map(lambda s: self._transform1(np.ascontiguousarray(a2[:, s:s + ch]), inverse), range(0, M, ch)))
            return np.concatenate(parts, axis=1).reshape(L, *shp)
        return self._transform1(a2, inverse).reshape(L, *shp)
    def transform(self, a, inverse=False):
        """Host in, host out; on the GPU backend the columns are streamed through the device in blocks."""
        if not GPU: return self.transform_dev(a, inverse)
        if MLXB: return host(self.transform_dev(a, inverse))
        shp = a.shape[1:]; L = a.shape[0]; a2 = a.reshape(L, -1); M = a2.shape[1]
        cols = max(1, int(budget_entries() // (4 * L)))
        out = np.empty_like(a2)
        for s in range(0, M, cols):
            out[:, s:s + cols] = host(self._transform1(dev(np.ascontiguousarray(a2[:, s:s + cols])), inverse))
        return out.reshape(L, *shp)
    def _transform_mlx_dev(self, d, inverse):
        """Device in, device out (mlx backend): bit-reversal, fused stages and scaling on the Apple GPU."""
        mx, stage, scale = _load_mlx(); p = self.p; L = d.shape[0]; M = int(d.size // L)
        ws, rev = self.roots(L, inverse)
        key = ('mlxrev', L)
        if key not in self.cache: self.cache[key] = mx.array(np.ascontiguousarray(host(rev), dtype=np.int32))
        wkey = ('mlxws', L, inverse)
        if wkey not in self.cache: self.cache[wkey] = mx.array(np.ascontiguousarray(host(ws), dtype=np.int64))
        d = mx.take(d.reshape(L, M).astype(mx.int64), self.cache[key], axis=0).reshape(-1)
        wsd = self.cache[wkey]; n = (L // 2) * M; tg = 256
        length = 2
        while length <= L:
            half = length // 2; step = L // length
            prm = mx.array(np.array([p, self.mu, half, step, M, n], dtype=np.int64))
            d = stage(inputs=[d, wsd, prm], grid=(n, 1, 1), threadgroup=(tg, 1, 1), output_shapes=[d.shape], output_dtypes=[mx.int64])[0]
            length *= 2
        if inverse:
            prm = mx.array(np.array([p, self.mu, pow(L, p - 2, p), 0, M, L * M], dtype=np.int64))
            d = scale(inputs=[d, prm], grid=(L * M, 1, 1), threadgroup=(tg, 1, 1), output_shapes=[d.shape], output_dtypes=[mx.int64])[0]
        mx.eval(d)
        return d.reshape(L, M)
    def _transform_mlx(self, a, inverse, rev):
        """Fused Metal butterflies on the Apple GPU (one kernel launch per stage, Barrett reduction); a is the
        (L, M) int64 host array in natural order; bit-reversal, stages and scaling run on the device; returns a host array."""
        mx, stage, scale = _load_mlx(); p = self.p; L, M = a.shape
        ws, _ = self.roots(L, inverse)
        key = ('mlxrev', L)
        if key not in self.cache: self.cache[key] = mx.array(np.ascontiguousarray(host(rev), dtype=np.int32))
        d = mx.take(mx.array(a), self.cache[key], axis=0).reshape(-1)
        wkey = ('mlxws', L, inverse)
        if wkey not in self.cache: self.cache[wkey] = mx.array(np.ascontiguousarray(host(ws), dtype=np.int64))
        wsd = self.cache[wkey]
        n = (L // 2) * M; tg = 256
        length = 2
        while length <= L:
            half = length // 2; step = L // length
            prm = mx.array(np.array([p, self.mu, half, step, M, n], dtype=np.int64))
            d = stage(inputs=[d, wsd, prm], grid=(n, 1, 1), threadgroup=(tg, 1, 1), output_shapes=[d.shape], output_dtypes=[mx.int64])[0]
            length *= 2
        if inverse:
            prm = mx.array(np.array([p, self.mu, pow(L, p - 2, p), 0, M, L * M], dtype=np.int64))
            d = scale(inputs=[d, prm], grid=(L * M, 1, 1), threadgroup=(tg, 1, 1), output_shapes=[d.shape], output_dtypes=[mx.int64])[0]
        mx.eval(d)
        return np.array(d, dtype=np.int64).reshape(L, M)
    def _transform_c(self, a, inverse, ws):
        """Fused C butterflies (one pass per stage, Barrett reduction), column blocks in parallel threads."""
        import ctypes
        lib = _load_ntt_c(); p = self.p; L, M = a.shape
        a = np.ascontiguousarray(a, dtype=np.int64); wsn = np.ascontiguousarray(host(ws), dtype=np.int64)
        ap = a.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)); wp = wsn.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        nth = max(1, min(NTHREADS, M // 64)); ch = (M + nth - 1) // nth
        cols = [(c, min(c + ch, M)) for c in range(0, M, ch)]
        length = 2
        while length <= L:
            half = length // 2; step = L // length
            if nth > 1:
                list(pool().map(lambda cc: lib.ntt_stage(ap, wp, p, self.mu, L, M, half, step, cc[0], cc[1]), cols))
            else:
                lib.ntt_stage(ap, wp, p, self.mu, L, M, half, step, 0, M)
            length *= 2
        if inverse:
            n = L * M; c = pow(L, p - 2, p); chn = (n + nth - 1) // nth
            list(pool().map(lambda s: lib.scale_mod(ap, c, p, self.mu, s, min(s + chn, n)), range(0, n, chn)))
        return a
    def _transform1(self, a, inverse):
        """In-place iterative radix-2 NTT on an (L, M) array of the backend's type."""
        p = self.p; L = a.shape[0]; ws, rev = self.roots(L, inverse)
        if not GPU and NTT_MODE == 'mlx' and _load_mlx() is not None:
            return self._transform_mlx(a, inverse, rev)              # Apple GPU path (opt-in: copies to/from the CPU eat its gain)
        a = xp.ascontiguousarray(a[rev])
        M = a.shape[1]
        if not GPU and NTT_MODE in ('fused', 'c') and _load_ntt_c() is not None:
            return self._transform_c(a, inverse, ws)
        if self.kern is not None:
            stage, scale = self.kern
            n = (L // 2) * M; threads = 256; blocks = (n + threads - 1) // threads
            length = 2
            while length <= L:
                half = length // 2; step = L // length
                stage((blocks,), (threads,), (a, ws, np.uint64(p), np.uint64(self.mu), np.int32(M), np.int32(half), np.int32(step), np.uint64(n)))
                length *= 2
            if inverse:
                nn = L * M; blocks2 = (nn + threads - 1) // threads
                scale((blocks2,), (threads,), (a, np.uint64(pow(L, p - 2, p)), np.uint64(p), np.uint64(self.mu), np.uint64(nn)))
            return a
        length = 2
        while length <= L:
            half = length // 2; step = L // length
            w = ws[::step][:half]
            v = a.reshape(L // length, 2, half, M)
            t = v[:, 1] * w[None, :, None]
            xp.remainder(t, p, out=t)
            u = v[:, 0]
            xp.subtract(u, t, out=v[:, 1]); xp.remainder(v[:, 1], p, out=v[:, 1])
            xp.add(u, t, out=v[:, 0]); xp.remainder(v[:, 0], p, out=v[:, 0])
            length *= 2
        if inverse:
            xp.multiply(a, pow(L, p - 2, p), out=a); xp.remainder(a, p, out=a)
        return a

def nextpow2(n):
    L = 1
    while L < n: L *= 2
    return L

def fold(A, L, p):
    """A (d+1, ...) host -> A mod (x^L - 1) as an (L, ...) host array."""
    out = np.zeros((L,) + A.shape[1:], dtype=np.int64)
    for s in range(0, A.shape[0], L):
        blk = A[s:s + L]; out[:blk.shape[0]] = (out[:blk.shape[0]] + blk) % p
    return out

def _resident_transform(ntt, Bf, L, k, n):
    """Transform of a folded/padded right factor (L, k, n) stored as int32 on the compute device, built in column blocks."""
    cch = max(1, int(WORK_ENTRIES // (L * k)))
    if MLXB:
        parts = [ntt.transform_dev(dev(np.ascontiguousarray(Bf[:, :, s:s + cch]))).astype(xp.int32) for s in range(0, n, cch)]
        Bt = parts[0] if len(parts) == 1 else xp.concatenate(parts, axis=2)
        xp.eval(Bt); return Bt
    Bt = xp.empty((L, k, n), dtype=xp.int32)
    for s in range(0, n, cch):
        Bt[:, :, s:s + cch] = ntt.transform_dev(dev(np.ascontiguousarray(Bf[:, :, s:s + cch]))).astype(xp.int32)
    return Bt

def _resident_transform_padded(ntt, B, L, k, n, dB):
    """Transform of the (dB+1, k, n) host array B zero-padded (or folded) to length L, stored int32 on the device, built in
    column blocks without materialising the padded array."""
    cch = max(1, int(WORK_ENTRIES // (L * k)))
    parts = []
    for s in range(0, n, cch):
        blk = np.zeros((L, k, min(cch, n - s)), dtype=np.int64)
        for t in range(0, dB + 1, L):
            piece = B[t:t + L, :, s:s + cch]; blk[:piece.shape[0]] = (blk[:piece.shape[0]] + piece) % ntt.p
        parts.append(ntt.transform_dev(dev(blk)).astype(xp.int32))
    if MLXB:
        Bt = parts[0] if len(parts) == 1 else xp.concatenate(parts, axis=2); xp.eval(Bt); return Bt
    Bt = xp.empty((L, k, n), dtype=xp.int32)
    for s, part in zip(range(0, n, cch), parts): Bt[:, :, s:s + cch] = part
    return Bt

def polymul(ntt, A, B, out_lo=0, out_hi=None, achunk=None):
    """A: (dA+1, m, k) dense or a GPoly; B: (dB+1, k, n) dense host polynomial matrix; coefficients out_lo..out_hi-1 of A*B.
    Middle product per row group of A: only B[out_lo - dA_g : out_hi] can contribute to the wanted range, so the group's
    transform length is nextpow2(dA_g + out_hi - out_lo) and its right factor is transformed once and kept resident."""
    p = ntt.p
    if not isinstance(A, GPoly): A = GPoly.from_dense(A)
    m, k = A.m, A.n; dB, n = B.shape[0] - 1, B.shape[2]
    D = A.degree() + dB + 1
    if out_hi is None: out_hi = D
    out_hi = min(out_hi, D)
    out = np.zeros((out_hi - out_lo, m, n), dtype=np.int64)
    for rows, C in A.groups:
        dA = C.shape[0] - 1; Dg = dA + dB + 1
        if out_lo >= Dg: continue
        hi = min(out_hi, Dg)
        b0 = max(0, out_lo - dA)                   # B coefficients below b0 cannot reach output index out_lo
        Bs = B[b0:hi]; dBs = Bs.shape[0] - 1       # product A * Bs has output index t' = t - b0
        L = nextpow2(dA + dBs + 1)
        Bt = _resident_transform_padded(ntt, Bs, L, k, n, dBs)
        ach = achunk or max(1, int(WORK_ENTRIES // (L * k)))
        for s in range(0, len(rows), ach):
            Ii = rows[s:s + ach]
            Ap = np.zeros((L, len(Ii), k), dtype=np.int64); Ap[:dA + 1] = C[:, s:s + ach, :]
            At = ntt.transform_dev(dev(Ap))
            Cg = host(ntt.transform_dev(mm_dev(p, At, Bt), inverse=True))
            out[:hi - out_lo, Ii, :] = Cg[out_lo - b0:hi - b0]
        del Bt
    return out

def polymul_rows(ntt, A, B, out_deg, rows=None, dense_out=False):
    """C = A*B for A (m x k) and B (k x n) polynomial matrices (dense (d+1, ., .) arrays or GPoly), using that row i of C
    has degree <= out_deg[i]: each row group is computed with a cyclic convolution of length nextpow2(out_deg+1)
    (the aliasing of the individual products cancels in the sum).  rows: subset of rows to compute (default all).
    Returns a GPoly grouped by out_deg (or, with dense_out, a dense (max deg + 1, m, n) array with other rows 0)."""
    p = ntt.p
    if not isinstance(A, GPoly): A = GPoly.from_dense(A)
    if not isinstance(B, GPoly): B = GPoly.from_dense(B)
    m, k, n = A.m, A.n, B.n
    rows = list(range(m)) if rows is None else list(rows)
    groups = {}
    for i in rows: groups.setdefault(nextpow2(int(out_deg[i]) + 1), []).append(i)
    BE = budget_entries()
    pieces = []
    for L in sorted(groups):
        I = np.array(groups[L]); Dg = min(L, max(int(out_deg[i]) for i in I) + 1)
        Cout = np.zeros((Dg, len(I), n), dtype=np.int64)
        if L * k * n // 2 <= BE:
            Bt = _resident_transform(ntt, B.folded_cols(L, p, 0, n), L, k, n)
            ach = max(1, int(WORK_ENTRIES // (L * k)))
            for s in range(0, len(I), ach):
                Ii = I[s:s + ach]
                At = ntt.transform_dev(dev(fold(A.rows_dense(Ii), L, p)))
                Cg = host(ntt.transform_dev(mm_dev(p, At, Bt), inverse=True))
                Cout[:, s:s + ach, :] = Cg[:Dg]
            del Bt
        elif L * k * len(I) // 2 <= BE:
            # few left rows (the high-degree group): keep their transform resident as int32 and stream the right
            # factor's columns, each transformed exactly once
            ach = max(1, int(WORK_ENTRIES // (L * k)))
            Ats = []
            for s in range(0, len(I), ach):
                Ats.append((s, ntt.transform_dev(dev(fold(A.rows_dense(I[s:s + ach]), L, p))).astype(xp.int32)))
            cch = max(1, int(WORK_ENTRIES // (L * k)))
            for t in range(0, n, cch):
                Bt = ntt.transform_dev(dev(B.folded_cols(L, p, t, min(t + cch, n))))
                for s, At in Ats:
                    Cg = host(ntt.transform_dev(mm_dev(p, At, Bt), inverse=True))
                    Cout[:, s:s + ach, t:t + cch] = Cg[:Dg]
            del Ats
        else:
            # neither operand fits: right factor's columns outermost (transformed once), left rows re-transformed inside
            cch = max(1, int(WORK_ENTRIES // (L * k))); ach = max(1, int(WORK_ENTRIES // (L * k)))
            for t in range(0, n, cch):
                Bt = ntt.transform_dev(dev(B.folded_cols(L, p, t, min(t + cch, n))))
                for s in range(0, len(I), ach):
                    At = ntt.transform_dev(dev(fold(A.rows_dense(I[s:s + ach]), L, p)))
                    Cg = host(ntt.transform_dev(mm_dev(p, At, Bt), inverse=True))
                    Cout[:, s:s + ach, t:t + cch] = Cg[:Dg]
        while Cout.shape[0] > 1 and not Cout[-1].any(): Cout = Cout[:-1]
        pieces.append((I, Cout))
    G = GPoly(pieces, m, n)
    return G.to_dense() if dense_out else G

class GPoly(object):
    """Polynomial matrix stored by row groups: groups = [(rows: int array, C: (d_g+1, |rows|, n) int64 host array)].
    Rows of very different degrees (the W non-kernel rows of an approximant basis reach the order, the others stay
    small) are kept only to their own degree, which bounds memory and shortens the transforms of the short rows."""
    def __init__(self, groups, m, n):
        self.groups = [(np.asarray(r, dtype=np.int64), C) for r, C in groups if len(r)]; self.m = m; self.n = n
    @staticmethod
    def from_dense(P):
        return GPoly([(np.arange(P.shape[1]), P)], P.shape[1], P.shape[2])
    def degree(self): return max(C.shape[0] - 1 for _, C in self.groups)
    def row_degrees(self):
        d = np.zeros(self.m, dtype=np.int64)
        for r, C in self.groups: d[r] = C.shape[0] - 1
        return d
    def rows_dense(self, I):
        """(dmax+1, |I|, n) dense array of the rows I (in the given order)"""
        I = np.asarray(I); pos = {int(i): t for t, i in enumerate(I)}
        d = max(C.shape[0] for r, C in self.groups if np.isin(r, I).any())
        out = np.zeros((d, len(I), self.n), dtype=np.int64)
        for r, C in self.groups:
            sel = [t for t, i in enumerate(r) if int(i) in pos]
            if sel: out[:C.shape[0], [pos[int(r[t])] for t in sel], :] = C[:, sel, :]
        return out
    def folded_cols(self, L, p, c0, c1):
        """(L, m, c1-c0): all rows, columns c0..c1-1, reduced mod x^L - 1"""
        out = np.zeros((L, self.m, c1 - c0), dtype=np.int64)
        for r, C in self.groups:
            for s in range(0, C.shape[0], L):
                blk = C[s:s + L, :, c0:c1]; out[:blk.shape[0], r, :] = (out[:blk.shape[0], r, :] + blk) % p
        return out
    def to_dense(self):
        d = self.degree() + 1; out = np.zeros((d, self.m, self.n), dtype=np.int64)
        for r, C in self.groups: out[:C.shape[0], r, :] = C
        return out
    def nbytes(self): return sum(C.nbytes for _, C in self.groups)

def regroup(rows_and_arrays, m, n, rdeg):
    """Build a GPoly from (rows, dense array) pieces, splitting rows by nextpow2(rdeg+1) and trimming each group's
    trailing zero coefficients."""
    groups = {}
    for r, C in rows_and_arrays:
        for t, i in enumerate(r):
            groups.setdefault(nextpow2(int(rdeg[i]) + 1), ([], []))
            groups[nextpow2(int(rdeg[i]) + 1)][0].append(int(i)); groups[nextpow2(int(rdeg[i]) + 1)][1].append(C[:, t, :])
    out = []
    for L in sorted(groups):
        idx, cols = groups[L]; d = max(c.shape[0] for c in cols)
        C = np.zeros((d, len(idx), n), dtype=np.int64)
        for t, c in enumerate(cols): C[:c.shape[0], t, :] = c
        while C.shape[0] > 1 and not C[-1].any(): C = C[:-1]
        out.append((np.array(idx), C))
    return GPoly(out, m, n)

def rref_mod(M, p):
    """Reduced row echelon form of M (r x c) over GF(p) (vectorised); returns (R, pivot columns)."""
    R = M.copy() % p; r, c = R.shape; piv = []; row = 0
    for col in range(c):
        if row >= r: break
        nz = np.nonzero(R[row:, col])[0]
        if len(nz) == 0: continue
        i = row + nz[0]
        if i != row: R[[row, i]] = R[[i, row]]
        R[row] = (R[row] * pow(int(R[row, col]), p - 2, p)) % p
        others = np.nonzero(R[:, col])[0]; others = others[others != row]
        if len(others):
            R[others] = (R[others] - R[others, col][:, None] * R[row][None, :]) % p
        piv.append(col); row += 1
    return R, piv

def mbasis(F, sigma, rdeg, p):
    """Iterative M-Basis: F (>= sigma, m, n) mod x^sigma, rdeg (m,) shifts. Returns P (dP+1, m, m), new rdeg."""
    m, n = F.shape[1], F.shape[2]
    P = np.zeros((sigma + 1, m, m), dtype=np.int64); P[0] = np.eye(m, dtype=np.int64); dP = 0
    Fr = np.array(F[:sigma], dtype=np.int64)     # residual P*F mod x^sigma (coefficients < k are zero after step k)
    rdeg = rdeg.copy()
    for k in range(sigma):
        R = Fr[k]                               # (m, n) coefficient of x^k of P*F
        if not R.any(): continue
        perm = np.argsort(rdeg, kind='stable')
        Rt = np.ascontiguousarray(R[perm].T)    # (n, m) columns in sorted row order
        E, pivcols = rref_mod(Rt, p)
        r = len(pivcols); pivset = set(pivcols)
        pivot_rows = perm[pivcols]              # original indices
        nonpiv_sorted = [i for i in range(m) if i not in pivset]
        nonpivot_rows = perm[nonpiv_sorted]
        C = E[:r][:, nonpiv_sorted].T           # (m - r, r): R[nonpiv i] = sum_l C[i, l] R[pivot l]
        if C.any():
            P[:dP + 1, nonpivot_rows, :] = (P[:dP + 1, nonpivot_rows, :] - matmul_mod(p, C, P[:dP + 1, pivot_rows, :])) % p
            Fr[k:, nonpivot_rows, :] = (Fr[k:, nonpivot_rows, :] - matmul_mod(p, C, Fr[k:, pivot_rows, :])) % p
        # pivot rows: multiply by x
        P[1:dP + 2, pivot_rows, :] = P[:dP + 1, pivot_rows, :]; P[0, pivot_rows, :] = 0; dP += 1
        Fr[k + 1:, pivot_rows, :] = Fr[k:-1, pivot_rows, :]; Fr[k, pivot_rows, :] = 0
        rdeg[pivot_rows] += 1
    while dP > 0 and not P[dP].any(): dP -= 1
    return P[:dP + 1].copy(), rdeg

T0 = [time.time()]
CKPT = [None]          # checkpoint file prefix (set from --ckpt in main)

def pmbasis(ntt, F, sigma, rdeg, base=64, depth=0, need_deg=None, path=''):
    """Divide-and-conquer approximant basis at order sigma; F (>= sigma, m, n) int64 mod p.
    Returns (P, rdeg): P a GPoly (rows grouped by degree), or at depth 0 with need_deg a dense array of the candidate rows."""
    p = ntt.p
    if sigma <= base:
        P, r = mbasis(F, sigma, rdeg, p)
        return regroup([(np.arange(P.shape[1]), P)], P.shape[1], P.shape[2], r), r
    s1 = sigma // 2
    ck = CKPT[0] if depth <= 1 else None                       # checkpoint the completed halves at depth 0 and 1
    tag = '%s_%s_s%d' % (ck, path or 'root', sigma) if ck else None
    if tag and os.path.exists(tag + '_rdeg1.npy'):
        rdeg1 = np.load(tag + '_rdeg1.npy'); gr = np.load(tag + '_P1.npz')
        P1 = GPoly([(gr['r%d' % i], gr['C%d' % i]) for i in range(int(gr['ngroups']))], F.shape[1], F.shape[1])
        print("    depth %d: first half order %d loaded from checkpoint %s" % (depth, s1, tag), flush=True)
    else:
        P1, rdeg1 = pmbasis(ntt, F[:s1], s1, rdeg, base, depth + 1, path=path + 'a')
        if tag:
            np.savez(tag + '_P1.npz', ngroups=len(P1.groups), **{'r%d' % i: r for i, (r, C) in enumerate(P1.groups)}, **{'C%d' % i: C for i, (r, C) in enumerate(P1.groups)})
            np.save(tag + '_rdeg1.npy', rdeg1)
    print("    depth %d: first half order %d done, P1 degree %d (%.1f GB grouped), %.0fs" % (depth, s1, P1.degree(), P1.nbytes() / 1e9, time.time() - T0[0]), flush=True) if depth <= 1 else None
    d1 = P1.degree()
    lo = max(0, s1 - d1)
    G = polymul(ntt, P1, F[lo:sigma], out_lo=s1 - lo, out_hi=sigma - lo)     # coefficients s1..sigma-1 of P1*F
    P2, rdeg2 = pmbasis(ntt, G, sigma - s1, rdeg1, base, depth + 1, path=path + 'b')
    del G
    rows = None
    if depth == 0 and need_deg is not None:
        rows = [i for i in range(len(rdeg2)) if rdeg2[i] <= need_deg]      # only candidate kernel rows at the top
        print("    top level: computing %d candidate rows (rdeg <= %d) of %d" % (len(rows), need_deg, len(rdeg2)), flush=True)
    P = polymul_rows(ntt, P2, P1, rdeg2, rows, dense_out=(rows is not None))
    del P1, P2
    if depth <= 1: print("    pmbasis depth %d order %d done: rdeg min/max %d/%d, %.0fs" % (depth, sigma, rdeg2.min(), rdeg2.max(), time.time() - T0[0]), flush=True)
    return P, rdeg2

def parse_J(s):
    d = {}
    for term in s.split(' + '):
        term = term.strip()
        mm = re.match(r'^(-?\d+)\*q\^(-?\d+)$', term)
        if mm: d[int(mm.group(2))] = int(mm.group(1))
        else: d[0] = int(term)
    return d

def build_F(Jseq, cells, W, p):
    """F[c, k] = q^{j k} J[k+b], per window normalised to min exponent 0; returns (F array (deg+1, m, W) int64, shifts).
    Vectorised per (window, order): the cells sharing an order b place the same polynomial J[k+b] at different shifts."""
    m = len(cells)
    jmin = [min(J) for J in Jseq]; jmax = [max(J) for J in Jseq]
    mins = [min(j * k + jmin[k + b] for (b, j) in cells) for k in range(W)]
    D = max(max(j * k + jmax[k + b] for (b, j) in cells) - mins[k] for k in range(W))
    F = np.zeros((D + 1, m, W), dtype=np.int64)
    by_b = {}
    for c, (b, j) in enumerate(cells): by_b.setdefault(b, []).append((c, j))
    for k in range(W):
        for b, cj in by_b.items():
            J = Jseq[k + b]; es = np.fromiter(J.keys(), dtype=np.int64, count=len(J)); vs = np.fromiter((v % p for v in J.values()), dtype=np.int64, count=len(J))
            for c, j in cj: F[es + j * k - mins[k], c, k] = vs
    return F, mins

def evaluate_rows(P_rows, x0, p):
    """Horner: P_rows (d+1, r, m) -> (r, m) values at x0."""
    acc = np.zeros(P_rows.shape[1:], dtype=np.int64)
    for d in range(P_rows.shape[0] - 1, -1, -1): acc = (acc * x0 + P_rows[d]) % p
    return acc

def main():
    args = sys.argv[1:]
    path, ORDER = args[0], int(args[1])
    def opt(name, default, typ=int):
        for a in args[2:]:
            if a.startswith(name): return typ(a[len(name):])
        return default
    DMAX = opt('--dmax=', ORDER + 2); W = opt('--windows=', 0); SIGX = opt('--sigma-extra=', 800)
    PI = opt('--prime=', 0); OUT = opt('--out=', None, str); CELLS = opt('--cells=', None, str); BASE = opt('--base=', 64)
    PILOT = opt('--cells-from=', None, str)      # take the cells and qN-degree from a finished run's output (another prime)
    CK = opt('--ckpt=', None, str)               # checkpoint prefix: completed halves at depth 0/1 are saved and reused on restart
    if CK: CKPT[0] = CK; os.makedirs(os.path.dirname(CK) or '.', exist_ok=True)
    p, g = NTT_PRIMES[PI]
    data = json.load(open(path))['values']
    NJ = max(int(k) for k in data); Jseq = [parse_J(data[str(N)]['J_normalised_q']) for N in range(1, NJ + 1)]
    if PILOT:
        pil = json.load(open(PILOT))
        allowed = {(b, j) for b, j, e, c in pil['terms']}
        DMAX = max(j for b, j in allowed)
        cells = sorted(allowed)
        print("cells taken from the pilot %s: %d cells, qN-degree %d" % (PILOT, len(cells), DMAX), flush=True)
    elif CELLS:
        allowed = set(tuple(c) for c in json.load(open(CELLS)))
        cells = [(b, j) for b in range(ORDER + 1) for j in range(DMAX + 1) if (b, j) in allowed]
    else:
        cells = [(b, j) for b in range(ORDER + 1) for j in range(DMAX + 1)]
    if W == 0: W = NJ - ORDER - 4
    Wall = NJ - ORDER
    t0 = time.time()
    F, mins = build_F(Jseq, cells, W, p)
    degF = F.shape[0] - 1; sigma = degF + SIGX
    m = len(cells)
    print("%s: J(1..%d); %d cells, %d fit windows (+%d held out); deg F = %d, sigma = %d, p = %d; generic row degree ~ %.0f" % (
        path, NJ, m, W, Wall - W, degF, sigma, p, W * sigma / m), flush=True)
    ntt = NTT(p, g)
    Fpad = np.zeros((sigma, m, W), dtype=np.int64); Fpad[:degF + 1] = F; del F
    if CKPT[0]:   # key the checkpoints by the input so a rerun with other cells/windows/prime cannot pick up stale halves
        import hashlib; CKPT[0] += '_' + hashlib.md5(Fpad[:2].tobytes() + Fpad[degF].tobytes() + str((sigma, m, W, p)).encode()).hexdigest()[:8]
    T0[0] = time.time()
    P, rdeg = pmbasis(ntt, Fpad, sigma, np.zeros(m, dtype=np.int64), base=BASE, need_deg=sigma - degF - 1)
    if isinstance(P, GPoly): P = P.to_dense()
    del Fpad
    print("approximant basis: P degree %d, sum rdeg %d (n*sigma = %d), %.0fs" % (P.shape[0] - 1, rdeg.sum(), W * sigma, time.time() - t0), flush=True)
    # actual row degrees
    # row degrees by one vectorised reverse-argmax, not a Python double loop over (degree, row):
    # measured 58x faster at the sizes this runs at (0.012s against 0.69s at degree 20000, 52 rows)
    nzrow = P.any(axis=2)                                     # (deg+1, m): does row i have any nonzero at degree d
    nzdeg = np.where(nzrow.any(axis=0), P.shape[0] - 1 - np.argmax(nzrow[::-1], axis=0), -1)
    order_rows = np.argsort(nzdeg)
    print("lowest row degrees:", sorted(nzdeg)[:12], " ... generic ~", int(np.median(nzdeg)))
    exact = [i for i in range(m) if 0 <= nzdeg[i] and nzdeg[i] + degF < sigma]
    print("rows that are exact kernel vectors (deg + deg F < sigma): %d, degrees %s" % (len(exact), sorted(nzdeg[exact])[:12]))
    if not exact:
        print("no exact kernel vector: increase --windows or --sigma-extra"); return
    # verify on all windows (fit + held out) at random points
    rng = np.random.RandomState(1)
    Fall, minsall = build_F(Jseq, cells, Wall, p)
    xs = [int(x) for x in rng.randint(2, p - 1, size=3)]
    Fvs = [evaluate_rows(Fall, x0, p) for x0 in xs]                          # (m, Wall) per point, once
    del Fall
    good = []
    for i in exact:
        ok = True
        for x0, Fv in zip(xs, Fvs):
            u = evaluate_rows(P[:, i:i + 1, :], x0, p)[0]                    # (m,)
            if (((u[:, None] * Fv) % p).sum(axis=0) % p).any(): ok = False; break
        good.append((i, ok))
    print("held-out verification:", [(int(i), int(nzdeg[i]), ok) for i, ok in good][:12])
    # minimal element in M-degree among the exact rows: constant combinations killing the cells with j > jm
    ex = [i for i, ok in good if ok]
    if not ex: print("no exact row survives held-out verification"); return
    dmin = min(nzdeg[ex]); lowrows = [i for i in ex if nzdeg[i] == dmin]
    jm = None
    def left_kernel_mod(rows_list):
        """basis of {v : v * Mtx = 0} over GF(p) for Mtx given as a list of rows (numpy int64), via rref of the transpose"""
        Mt = np.array(rows_list, dtype=np.int64).T % p                      # columns = rows of Mtx
        R, piv = rref_mod(Mt, p)
        free = [j for j in range(Mt.shape[1]) if j not in piv]
        basis = []
        for f in free:
            v = np.zeros(Mt.shape[1], dtype=np.int64); v[f] = 1
            for r_i, pc in enumerate(piv): v[pc] = (-R[r_i, f]) % p
            basis.append(v)
        return basis
    for jtry in range(DMAX + 1):
        cols_hi = [c for c, (b, j) in enumerate(cells) if j > jtry]
        if not cols_hi: jm = jtry; comb = [np.array([1] + [0] * (len(lowrows) - 1), dtype=np.int64)]; break
        rows_list = [[int(P[d, i, c]) for c in cols_hi for d in range(dmin + 1)] for i in lowrows]
        K = left_kernel_mod(rows_list)
        if K: jm = jtry; comb = K; break
    print("minimal M-degree among %d lowest-degree exact rows: qN-deg %d, combination space dim %d" % (len(lowrows), jm, len(comb)))
    vec = np.zeros((dmin + 1, m), dtype=np.int64)
    for t, i in enumerate(lowrows):
        vec = (vec + int(comb[0][t]) * P[:dmin + 1, i, :]) % p
    terms = [(cells[c][0], cells[c][1], d, int(vec[d, c])) for c in range(m) for d in range(dmin + 1) if vec[d, c]]
    used = {(b, j) for b, j, d, v in terms}
    print("operator mod p: order %d, qN-deg %d, q-degree %d, %d terms in %d cells; total %.0fs" % (
        max(b for b, j in used), max(j for b, j in used), dmin, len(terms), len(used), time.time() - t0), flush=True)
    if OUT:
        json.dump({'p': p, 'order': ORDER, 'cells': cells, 'qdeg': int(dmin), 'Mdeg': int(jm), 'terms': terms,
                   'row_degrees_sorted': sorted(int(x) for x in nzdeg)[:50], 'windows': W, 'sigma': int(sigma)}, open(OUT, 'w'))
        print("wrote", OUT)

if __name__ == '__main__':
    main()
