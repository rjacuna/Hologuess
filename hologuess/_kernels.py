"""Operation-scoped selection of the optional bulk FLINT matrix bridge."""

from contextlib import contextmanager
from contextvars import ContextVar
import ctypes
from functools import lru_cache
from operator import index
import os
from pathlib import Path
import sys

import numpy as np


_LIBRARY_ENV = 'HOLOGUESS_FLINT_LIBRARY'
_BUILD_GUIDE = 'Build the bridge with python -m hologuess.build_flint --help.'
_BRIDGE_ABI = 1
_ACTIVE_KERNEL = ContextVar('hologuess_active_kernel', default=None)


def _signed_matrix(values, dimensions):
    array = np.asarray(values)
    if array.ndim != dimensions:
        raise ValueError(f'Expected a {dimensions}-dimensional modular matrix array.')
    if array.dtype.kind != 'i' or array.dtype.itemsize > 8:
        raise TypeError('Modular matrix entries must be exact signed integers up to 64 bits.')
    return np.require(array, dtype=np.int64, requirements=('C', 'A'))


class _FlintKernel:
    def __init__(self, library, path, version):
        self._library = library
        self.path = path
        self.version = version
        self._rref = library.hologuess_flint_rref
        matrix_buffer = np.ctypeslib.ndpointer(
            dtype=np.int64, ndim=3, flags=('C_CONTIGUOUS', 'ALIGNED'))
        mask_buffer = np.ctypeslib.ndpointer(
            dtype=np.uint8, ndim=2, flags=('C_CONTIGUOUS', 'ALIGNED'))
        self._rref.argtypes = [matrix_buffer, matrix_buffer, mask_buffer,
                              ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                              ctypes.c_uint32]
        self._rref.restype = ctypes.c_int

    def _reduce(self, matrices, p):
        if isinstance(p, (bool, np.bool_)):
            raise TypeError('The modulus must be an integer prime.')
        p = index(p)
        if p < 2 or p >= 2**31:
            raise ValueError('Bulk FLINT requires a prime below 2**31.')
        count, rows, cols = matrices.shape
        reduced = np.empty_like(matrices)
        masks = np.zeros((count, cols), dtype=np.uint8)
        status = self._rref(matrices, reduced, masks, count, rows, cols, p)
        if status == 1:
            raise ValueError('Bulk FLINT requires a prime below 2**31.')
        if status == 2:
            raise OverflowError('Matrix dimensions exceed the FLINT bridge limits.')
        if status != 0:
            raise RuntimeError(f'Unexpected FLINT bridge status {status}.')
        return reduced, masks.view(np.bool_)

    def rref(self, A, p):
        reduced, masks = self._reduce(_signed_matrix(A, 2)[None], p)
        return reduced[0], np.flatnonzero(masks[0]).tolist()

    def batched_nullspace(self, M, p):
        reduced, pivot_masks = self._reduce(_signed_matrix(M, 3), p)
        count, cols = pivot_masks.shape
        alive = np.ones(count, dtype=bool)
        profile, free = [], []
        live_count = count
        # Every surviving point agrees on previous pivot decisions. Its
        # canonical RREF therefore gives the same next pivot as qhsolve's
        # shared elimination. Reproduce the live majority, including ties.
        for col in range(cols):
            has_pivot = pivot_masks[:, col]
            pivot_count = np.count_nonzero(alive & has_pivot)
            if 2 * pivot_count > live_count:
                alive &= has_pivot
                profile.append(col)
                live_count = pivot_count
            else:
                alive &= ~has_pivot
                free.append(col)
                live_count -= pivot_count

        basis = np.zeros((count, len(free), cols), dtype=np.int64)
        free_indices = np.arange(len(free))
        for point in np.flatnonzero(alive):
            basis[point, free_indices, free] = 1
            if free and profile:
                block = reduced[point, :len(profile)][:, free]
                basis[point][:, profile] = -block.T % p
        return basis, alive


@lru_cache(maxsize=8)
def _load_flint(path):
    try:
        library = ctypes.CDLL(os.fspath(path))
        abi = library.hologuess_flint_abi
        abi.argtypes = []
        abi.restype = ctypes.c_uint32
        if abi() != _BRIDGE_ABI:
            raise ValueError('incompatible Hologuess bridge ABI')
        header = library.hologuess_flint_header_version
        runtime = library.hologuess_flint_runtime_version
        for version in (header, runtime):
            version.argtypes = []
            version.restype = ctypes.c_char_p
        linked_version = runtime()
        if not linked_version or header() != linked_version:
            raise ValueError('FLINT headers and linked runtime have different versions')
        return _FlintKernel(library, path, linked_version.decode('ascii'))
    except (AttributeError, OSError, ValueError) as exc:
        raise RuntimeError(f'Cannot load the Hologuess FLINT bridge at {path}: {exc}. {_BUILD_GUIDE}') from exc


def bridge_suffix():
    if sys.platform == 'darwin':
        return '.dylib'
    if sys.platform.startswith('linux'):
        return '.so'
    return None


def default_flint_path():
    """Return the platform's user-install location for the bundled bridge ABI."""
    suffix = bridge_suffix()
    if suffix is None:
        return None
    cache_home = os.environ.get('XDG_CACHE_HOME')
    cache = Path(cache_home).expanduser() if cache_home else Path.home() / '.cache'
    return cache / 'hologuess' / f'flint-abi{_BRIDGE_ABI}' / f'libhologuess_flint{suffix}'


def _flint_path(path):
    if path is not None:
        return path
    configured = os.environ.get(_LIBRARY_ENV)
    if configured:
        return configured
    installed = default_flint_path()
    if installed is not None:
        try:
            installed.lstat()
        except FileNotFoundError:
            return None
    return installed


def load_flint(path=None):
    """Load and validate an installed bridge, with no effect on selection."""
    selected = _flint_path(path)
    if not selected:
        raise RuntimeError(f'Install a bridge, supply flint_library or set {_LIBRARY_ENV}. {_BUILD_GUIDE}')
    return _load_flint(Path(selected).expanduser().resolve())


def active_kernel():
    """Return the selected native kernel, or None for the NumPy implementation."""
    return _ACTIVE_KERNEL.get()


@contextmanager
def kernel_scope(kernel_backend='auto', flint_library=None):
    """Select kernels for one reconstruction and restore the caller's scope."""
    if kernel_backend == 'numpy':
        if flint_library is not None:
            raise ValueError('flint_library requires kernel_backend="auto" or "flint".')
        selected = None
    elif kernel_backend == 'flint':
        selected = load_flint(flint_library)
    elif kernel_backend == 'auto':
        path = _flint_path(flint_library)
        selected = None if path is None else load_flint(path)
    else:
        raise ValueError('kernel_backend must be "auto", "numpy" or "flint".')
    token = _ACTIVE_KERNEL.set(selected)
    try:
        yield selected
    finally:
        _ACTIVE_KERNEL.reset(token)
