"""Selection and exact compatibility tests for Hologuess's FLINT kernels."""

from math import factorial
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hologuess import _kernels, _qhsolve as core
from hologuess import guess, guess_qsequence, interpolate_problem, solve_module
from hologuess.module import ModuleProblem


LIBRARY = _kernels._flint_path(None)


def generic_matrix(a, p):
    """Use Sage's generic field algorithms as an independent exact oracle."""
    from sage.all import GF, matrix

    return matrix(GF(p), *a.shape, [int(value) for value in a.flat],
                  implementation='generic')


class KernelSelectionTests(unittest.TestCase):
    def test_explicit_numpy_does_not_load_a_configured_library(self):
        with patch.dict(os.environ, HOLOGUESS_FLINT_LIBRARY='unusable-bridge'):
            with patch.object(_kernels, 'load_flint', side_effect=AssertionError('unexpected FLINT load')):
                with _kernels.kernel_scope('numpy'):
                    self.assertIsNone(_kernels.active_kernel())
                    reduced, pivots = core.rref_mod(np.array([[1, 2], [2, 4]]), 101)
        self.assertEqual(pivots, [0])
        np.testing.assert_array_equal(reduced, [[1, 2], [0, 0]])

    def test_auto_falls_back_when_no_bridge_is_installed(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, XDG_CACHE_HOME=directory):
            os.environ.pop('HOLOGUESS_FLINT_LIBRARY', None)
            problem = ModuleProblem([[1, 2, 3], [2, 4, 6]], training_rows=2)
            result = solve_module(problem)
        self.assertEqual(result.metadata['kernel_backend'], 'numpy')
        self.assertTrue(result.metadata['exact_holdout'])

    def test_auto_validates_a_configured_library(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / 'never-created.so')
            with patch.dict(os.environ, HOLOGUESS_FLINT_LIBRARY=missing):
                with self.assertRaisesRegex(RuntimeError, 'Cannot load the Hologuess FLINT bridge'):
                    with _kernels.kernel_scope():
                        self.fail('missing configured bridge reached the body')
            with self.assertRaisesRegex(RuntimeError, 'Cannot load the Hologuess FLINT bridge'):
                with _kernels.kernel_scope(flint_library=missing):
                    self.fail('missing explicit bridge reached the body')

    def test_invalid_selection_and_missing_library_are_explicit(self):
        with self.assertRaisesRegex(ValueError, 'backend'):
            with _kernels.kernel_scope('foreign', None):
                self.fail('invalid backend reached the body')
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / 'never-created.so'
            with self.assertRaisesRegex(ValueError, 'flint'):
                with _kernels.kernel_scope('numpy', missing):
                    self.fail('unexpected library path reached the body')
            with self.assertRaisesRegex(RuntimeError, 'FLINT|flint'):
                with _kernels.kernel_scope('flint', missing):
                    self.fail('missing bridge reached the body')
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, XDG_CACHE_HOME=directory):
            os.environ.pop('HOLOGUESS_FLINT_LIBRARY', None)
            with self.assertRaisesRegex(RuntimeError, 'HOLOGUESS_FLINT_LIBRARY'):
                _kernels.load_flint()
        self.assertIsNone(_kernels.active_kernel())


@unittest.skipUnless(LIBRARY, 'install the FLINT bridge or set HOLOGUESS_FLINT_LIBRARY')
class FlintArithmeticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # An explicitly supplied but broken library is a test failure.
        cls.backend = _kernels.load_flint(LIBRARY)

    def flint_options(self):
        return {'kernel_backend': 'flint', 'flint_library': LIBRARY}

    def assert_rref(self, a, p):
        result, pivots = self.backend.rref(a, p)
        oracle = generic_matrix(a, p).echelon_form()
        expected = np.array([int(value) for value in oracle.list()], dtype=np.int64)
        np.testing.assert_array_equal(result, expected.reshape(a.shape))
        self.assertEqual(tuple(pivots), oracle.pivots())
        self.assertEqual(len(pivots), oracle.rank())

    def assert_kernel(self, matrix, basis, p):
        field_matrix = generic_matrix(matrix, p)
        pivots = field_matrix.echelon_form().pivots()
        free = [i for i in range(matrix.shape[1]) if i not in pivots]
        kernel_matrix = generic_matrix(basis, p)
        self.assertEqual(len(basis), len(free))
        np.testing.assert_array_equal(basis[:, free], np.eye(len(free), dtype=np.int64))
        self.assertTrue((field_matrix * kernel_matrix.transpose()).is_zero())

    def test_exact_rrefs_at_solver_primes_and_noncontiguous_input(self):
        rng = np.random.default_rng(7402)
        for p in (2, 101, core.PRIMES[0], 2147483647):
            with self.subTest(p=p):
                storage = rng.integers(-2*p, 2*p, (8, 14), dtype=np.int64)
                a = storage[::-1, ::2]
                a[0, 0] = np.iinfo(np.int64).min
                a[1] = a[0]
                a[:, 1] = a[:, 4]
                snapshot = a.copy()
                self.assert_rref(a, p)
                np.testing.assert_array_equal(a, snapshot)

    def test_empty_rank_zero_and_invalid_arithmetic(self):
        for shape in ((0, 4), (4, 0), (3, 4)):
            with self.subTest(shape=shape):
                a = np.zeros(shape, dtype=np.int64)
                self.assert_rref(a, 101)
                basis, alive = self.backend.batched_nullspace(a[None], 101)
                np.testing.assert_array_equal(alive, [True])
                np.testing.assert_array_equal(basis[0], np.eye(shape[1], dtype=np.int64))
        basis, alive = self.backend.batched_nullspace(np.empty((0, 3, 4), dtype=np.int64), 101)
        self.assertEqual(basis.shape, (0, 4, 4))
        self.assertEqual(alive.shape, (0,))
        for p in (1, 91, 2**31):
            with self.subTest(p=p), self.assertRaises(ValueError):
                self.backend.rref(np.array([[1]], dtype=np.int64), p)
        for dtype in (np.float64, np.uint64, object):
            with self.subTest(dtype=dtype), self.assertRaises(TypeError):
                self.backend.rref(np.array([[1]], dtype=dtype), 101)

    def test_mixed_pivots_use_ties_and_only_surviving_points(self):
        # The first column ties and keeps only the last two points. Their next
        # tie drops the third point even though column two wins over all points.
        batch = np.array([[[1, 0, 0, 0], [0, 0, 1, 0]],
                          [[1, 0, 0, 0], [0, 0, 1, 0]],
                          [[0, 1, 0, 0], [0, 0, 1, 0]],
                          [[0, 1, 7, 0], [0, 0, 0, 1]]], dtype=np.int64)
        p = core.PRIMES[0]
        basis, alive = self.backend.batched_nullspace(batch, p)
        np.testing.assert_array_equal(alive, [False, False, False, True])
        np.testing.assert_array_equal(basis[-1], [[1, 0, 0, 0], [0, p-7, 1, 0]])
        self.assert_kernel(batch[-1], basis[-1], p)

    def test_random_batched_kernels_match_numpy_and_independent_residuals(self):
        p = core.PRIMES[1]
        rng = np.random.default_rng(872)
        batch = rng.integers(-p, 2*p, size=(9, 8, 7), dtype=np.int64).swapaxes(1, 2)
        batch[:, :, -1] = 3*batch[:, :, 1] - 2*batch[:, :, 2]
        batch[:2, :, 0] = 0
        batch[0] = 0
        baseline, base_alive = core.batched_nullspace(batch, p)
        candidate, alive = self.backend.batched_nullspace(batch, p)
        self.assertGreater(int(alive.sum()), 0)
        self.assertFalse(bool(alive[0]))
        np.testing.assert_array_equal(alive, base_alive)
        np.testing.assert_array_equal(candidate[alive], baseline[base_alive])
        for i in np.flatnonzero(alive):
            self.assert_kernel(batch[i], candidate[i], p)

    def test_backend_context_restores_on_exit_and_exception(self):
        class Expected(Exception):
            pass

        self.assertIsNone(_kernels.active_kernel())
        with self.assertRaises(Expected):
            with _kernels.kernel_scope('flint', LIBRARY):
                self.assertIsNotNone(_kernels.active_kernel())
                with _kernels.kernel_scope('numpy', None):
                    self.assertIsNone(_kernels.active_kernel())
                self.assertIsNotNone(_kernels.active_kernel())
                raise Expected
        self.assertIsNone(_kernels.active_kernel())

    @unittest.skipUnless(_kernels.bridge_suffix(), 'requires a supported bridge install platform')
    def test_default_discovers_an_installed_bridge(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, XDG_CACHE_HOME=directory):
            os.environ.pop('HOLOGUESS_FLINT_LIBRARY', None)
            installed = _kernels.default_flint_path()
            installed.parent.mkdir(parents=True)
            installed.symlink_to(Path(LIBRARY).expanduser().resolve())
            problem = ModuleProblem([[1, 2, 3], [2, 4, 6]], training_rows=2)
            result = solve_module(problem)
            self.assertEqual(result.metadata['kernel_backend'], 'flint')
            self.assertTrue(result.metadata['exact_holdout'])
            self.assertIs(_kernels.load_flint(), self.backend)

    def test_q_and_differential_public_operators_agree_exactly(self):
        from sage.all import QQ, PolynomialRing
        from ore_algebra import OreAlgebra

        values = [{k*k: 1} for k in range(24)]
        options = {'max_order': 1, 'max_degree': 2, 'max_points': 128}
        current = guess_qsequence(values, **options, kernel_backend='numpy')
        with patch.dict(os.environ, HOLOGUESS_FLINT_LIBRARY=os.fspath(LIBRARY)):
            compiled = guess_qsequence(values, **options)
        self.assertEqual(compiled.coefficients, current.coefficients)
        self.assertEqual(compiled.metadata['kernel_backend'], 'flint')
        self.assertEqual(current.metadata['kernel_backend'], 'numpy')
        self.assertTrue(compiled.metadata['finite_window_valid'])
        self.assertEqual(compiled.order, 1)

        A = OreAlgebra(PolynomialRing(QQ, 'x'), 'Dx')
        taylor = [QQ(1) / factorial(k) for k in range(16)]
        current = guess(taylor, A, path=[(1, 0)], holdout=4, kernel_backend='numpy')
        with patch.dict(os.environ, HOLOGUESS_FLINT_LIBRARY=os.fspath(LIBRARY)):
            compiled = guess(taylor, A, path=[(1, 0)], holdout=4)
        self.assertEqual(compiled.operators, current.operators)
        expected = A.gen() - 1
        self.assertEqual(compiled.operator, compiled.operator.leading_coefficient() * expected)
        self.assertEqual(compiled.metadata['kernel_backend'], 'flint')
        self.assertTrue(compiled.metadata['exact_training'])
        self.assertTrue(compiled.metadata['exact_holdout'])

    def test_module_cache_is_shared_between_exact_backends(self):
        problem = ModuleProblem([[1, 2, 3], [2, 4, 6], [3, 6, 9]], training_rows=2)
        for first in ('numpy', 'flint'):
            with self.subTest(first=first):
                store = core.MemoryKernelStore()
                options = {'numpy': {'kernel_backend': 'numpy'}, 'flint': self.flint_options()}
                second = 'numpy' if first == 'flint' else 'flint'
                before = solve_module(problem, store=store, **options[first])
                after = solve_module(problem, store=store, **options[second])
                self.assertEqual(after.basis, before.basis)
                self.assertEqual(after.valid_basis, before.valid_basis)
                self.assertEqual(after.metadata['kernel_backend'], second)
                self.assertGreater(after.metadata['cache_hits'], 0)
                self.assertTrue(after.metadata['exact_holdout'])

    def test_parameter_interpolation_keeps_held_out_validation(self):
        from sage.all import QQ, PolynomialRing

        field = PolynomialRing(QQ, 't').fraction_field()
        t = field.gen()
        values = [(t+1)**n / (t-2) for n in range(10)]
        problem = ModuleProblem([values[:-1], values[1:]], training_rows=7)
        current = interpolate_problem(problem, t, band=(-3, 3), max_points=128,
                                      kernel_backend='numpy')
        with patch.dict(os.environ, HOLOGUESS_FLINT_LIBRARY='unusable-bridge'):
            compiled = interpolate_problem(problem, t, band=(-3, 3), max_points=128,
                                           flint_library=LIBRARY)
        self.assertEqual(compiled.coefficients, current.coefficients)
        coefficients = [field.zero(), field.zero()]
        for (column, exponent), value in compiled.coefficients.items():
            coefficients[column] += value*t**exponent
        self.assertNotEqual(coefficients[1], 0)
        self.assertEqual(coefficients[0] + (t+1)*coefficients[1], 0)
        self.assertEqual(compiled.metadata['kernel_backend'], 'flint')
        self.assertTrue(compiled.metadata['exact_all_observations'])
        self.assertEqual(compiled.metadata['holdout_observations'], 2)

    @unittest.skipUnless('fork' in multiprocessing.get_all_start_methods(), 'requires fork workers')
    def test_fork_workers_use_flint_and_numpy_reads_their_point_cache(self):
        p = core.PRIMES[0]
        points = np.arange(2, 10, dtype=np.int64)
        seq = np.repeat(np.array([pow(2, k, p) for k in range(9)], dtype=np.int64)[:, None],
                        len(points), axis=1)
        arguments = (p, points, seq, [(0, 0), (1, 0)], 8)
        store = core.MemoryKernelStore()
        default = core.solve_points(*arguments, procs=1, K=3, store=core.MemoryKernelStore())
        backend_type = type(self.backend)
        original = backend_type.batched_nullspace
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'compiled-calls'

            def record(instance, matrices, prime):
                with marker.open('a') as stream:
                    stream.write(f'{os.getpid()}\n')
                return original(instance, matrices, prime)

            with _kernels.kernel_scope('flint', LIBRARY):
                with patch.object(backend_type, 'batched_nullspace', record):
                    computed = core.solve_points(*arguments, procs=2, K=3, store=store)
            pids = [int(line) for line in marker.read_text().splitlines()]
            self.assertEqual(len(pids), 3)
            self.assertNotIn(os.getpid(), pids)
        with patch.object(core, 'solve_points_task', side_effect=AssertionError('point cache miss')):
            replay = core.solve_points(*arguments, procs=1, K=3, store=store)
        for current, compiled, cached in zip(default, computed, replay):
            np.testing.assert_array_equal(compiled, current)
            np.testing.assert_array_equal(cached, current)


if __name__ == '__main__':
    unittest.main()
