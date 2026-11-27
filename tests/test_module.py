"""Independent exact finite-matrix checks for module reconstruction."""
import unittest
from sage.all import QQ, GF, PolynomialRing, matrix, sqrt, NumberField
from hologuess.module import ModuleProblem, solve_module
from hologuess import _qhsolve as core


class ModuleTests(unittest.TestCase):
    def verify_full_kernel(self, rows, result, field=QQ):
        m = matrix(field, rows)
        b = matrix(field, len(result.basis), m.ncols(), sum((list(v) for v in result.basis), []))
        self.assertEqual(b.rank(), len(result.basis))
        self.assertEqual(m * b.transpose(), matrix(field, m.nrows(), b.nrows(), 0))
        self.assertEqual(len(result.basis), m.ncols() - m.rank())

    def test_full_multidimensional_kernel(self):
        rows = [[1, 2, 3, 0], [2, 4, 6, 0], [5, 10, 15, 0]]
        result = solve_module(ModuleProblem(zip(*rows), labels=('a', 'b', 'c', 'd')))
        self.assertEqual(result.status, 'finite_kernel')
        self.assertEqual(result.metadata['nullity'], 3)
        self.assertFalse(result.metadata['all_indices_proved'])
        self.verify_full_kernel(rows, result)

    def test_separate_holdout_failure(self):
        p = ModuleProblem([[1, 2, 3], [2, 4, 7]], training_rows=2)
        result = solve_module(p)
        self.assertEqual(result.status, 'holdout_failed')
        self.assertEqual(result.holdout_failures, (0,))
        self.assertTrue(result.metadata['exact_training'])
        self.assertFalse(result.metadata['exact_holdout'])
        self.assertEqual(result.valid_basis, ())

    def test_holdout_valid_linear_combination(self):
        # Both e_0 and e_1 fail individually; e_0-e_1 is nevertheless valid.
        result = solve_module(ModuleProblem([[0, 1], [0, 1]], training_rows=1))
        self.assertEqual(result.basis, ((QQ(1), QQ(0)), (QQ(0), QQ(1))))
        self.assertEqual(result.holdout_failures, (0, 1))
        self.assertEqual(result.status, 'holdout_failed')
        self.assertEqual(len(result.valid_basis), 1)
        a, b = result.valid_basis[0]
        self.assertNotEqual(a, 0)
        self.assertEqual(a + b, 0)
        self.assertTrue(result.metadata['holdout_used_to_select'])
        self.assertTrue(result.metadata['full_holdout_intersection'])

    def test_holdout_parameter_identity_intersection(self):
        R = PolynomialRing(QQ, 't'); t = R.gen()
        result = solve_module(ModuleProblem([[0, t], [0, 1], [0, t+1]], training_rows=1))
        self.assertEqual(result.holdout_failures, (0, 1, 2))
        self.assertEqual(len(result.valid_basis), 1)
        a, b, c = result.valid_basis[0]
        self.assertEqual(a*t + b + c*(t+1), 0)
        self.assertNotEqual(c, 0)

    def test_holdout_finite_extension_intersection(self):
        K = GF(9, 'a'); a = K.gen()
        result = solve_module(ModuleProblem([[K(0), a], [K(0), K(1)]], training_rows=1))
        self.assertEqual(result.holdout_failures, (0, 1))
        self.assertEqual(len(result.valid_basis), 1)
        u, v = result.valid_basis[0]
        self.assertEqual(a*u + v, 0)

    def test_true_holdout(self):
        result = solve_module(ModuleProblem([[1, 2, 3], [2, 4, 6]], training_rows=2))
        self.assertEqual(result.status, 'validated')
        self.assertTrue(result.metadata['exact_holdout'])

    def test_bad_first_prime_rank(self):
        rows = [[3, 0], [0, 1]]
        result = solve_module(ModuleProblem(zip(*rows)), primes=[3, 5, 7, 11])
        self.assertEqual(result.status, 'no_relations')
        self.assertEqual(result.metadata['profile_resets'], 1)
        self.verify_full_kernel(rows, result)

    def test_bad_first_prime_same_rank_different_pivot(self):
        rows = [[3, 1, 2]]
        result = solve_module(ModuleProblem(zip(*rows)), primes=[3, 5, 7, 11, 13, 17])
        self.assertEqual(result.status, 'finite_kernel')
        self.assertEqual(result.metadata['pivot_columns'], (0,))
        self.assertEqual(result.metadata['profile_resets'], 1)
        self.verify_full_kernel(rows, result)

    def test_denominator_prime_skipped(self):
        rows = [[QQ(1)/3, QQ(2)/3, 1]]
        result = solve_module(ModuleProblem(zip(*rows)), primes=[3, 5, 7, 11, 13])
        self.assertEqual(result.metadata['skipped_primes'][0], (3, 'denominator'))
        self.verify_full_kernel(rows, result)

    def test_polynomial_identity_expansion(self):
        R = PolynomialRing(QQ, 't'); t = R.gen()
        result = solve_module(ModuleProblem([[t + 1], [t - 1], [1]]))
        self.assertEqual(result.metadata['expanded_rows'], 2)
        self.assertEqual(result.basis, ((QQ(-1)/2, QQ(1)/2, QQ(1)),))

    def test_rational_parameter_identity(self):
        R = PolynomialRing(QQ, 't'); t = R.gen()
        result = solve_module(ModuleProblem([[1/(t + 1)], [t/(t + 1)], [1]]))
        self.assertEqual(result.basis, ((QQ(-1), QQ(-1), QQ(1)),))
        self.assertEqual(result.metadata['parameters'], ('t',))

    def test_multiple_and_nested_parameters(self):
        R = PolynomialRing(QQ, 't'); t = R.gen()
        S = PolynomialRing(R.fraction_field(), 'u'); u = S.gen()
        result = solve_module(ModuleProblem([[u/(t + 1)], [u*t/(t + 1)], [u]]))
        self.assertEqual(result.basis, ((QQ(-1), QQ(-1), QQ(1)),))
        self.assertEqual(result.metadata['parameters'], ('t', 'u'))

    def test_finite_extension_field(self):
        K = GF(9, 'a'); a = K.gen()
        rows = [[a, 1, a + 1], [a*a, a, a*a+a]]
        result = solve_module(ModuleProblem(zip(*rows)))
        self.assertEqual(result.metadata['backend'], 'exact_constant_field')
        self.verify_full_kernel(rows, result, K)

    def test_finite_field_parameter_identity(self):
        K = GF(7); R = PolynomialRing(K, 't'); t = R.gen()
        result = solve_module(ModuleProblem([[1/(t+1)], [t/(t+1)], [1]]))
        self.assertEqual(result.metadata['nullity'], 1)
        v = result.basis[0]
        self.assertEqual(tuple(c / v[-1] for c in v), (K(-1), K(-1), K(1)))

    def test_algebraic_constant_field(self):
        R = PolynomialRing(QQ, 'z'); K = NumberField(R.gen()**2 - 2, 'a'); a = K.gen()
        rows = [[a, 1, a + 1], [2, a, 2 + a]]
        result = solve_module(ModuleProblem(zip(*rows)))
        self.verify_full_kernel(rows, result, K)

    def test_zero_observations(self):
        result = solve_module(ModuleProblem([[0, 0], [0, 0]]))
        self.assertEqual(result.basis, ((QQ(1), QQ(0)), (QQ(0), QQ(1))))

    def test_persistent_kernel_store(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            store = core.SQLiteKernelStore(str(Path(directory)/'kernels.sqlite'))
            p = ModuleProblem([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
            first = solve_module(p, store=store)
            again = solve_module(p, store=store)
            self.assertEqual(first.basis, again.basis)
            self.assertGreater(again.metadata['cache_hits'], 0)
            changed = solve_module(ModuleProblem([[1, 2, 3], [2, 4, 7], [3, 6, 9]]), store=store)
            self.assertEqual(changed.metadata['cache_hits'], 0)
            self.assertNotEqual(first.basis, changed.basis)

    def test_corrupted_cache_cannot_assert_full_rank(self):
        import numpy as np
        from hologuess.module import _expand, _cache_key
        problem = ModuleProblem([[1, 2], [2, 4]])
        _, exact, _, _ = _expand(problem)
        store = core.MemoryKernelStore()
        # Structurally plausible rank-two/empty-kernel certificate, but rank=1.
        store.put_many([(_cache_key(exact, 101), 101, 1, np.array([1, 1]))])
        with self.assertRaisesRegex(ValueError, 'rank differs'):
            solve_module(problem, primes=[101, 103, 107], store=store)

    def test_corrupted_cache_same_rank_wrong_kernel_rejected(self):
        import numpy as np
        from hologuess.module import _expand, _cache_key
        problem = ModuleProblem([[1, 2], [2, 4]])
        _, exact, _, _ = _expand(problem)
        store = core.MemoryKernelStore()
        # Correct rank and free-coordinate identity, false residual for (0,1).
        store.put_many([(_cache_key(exact, 101), 101, 1, np.array([1, 0, 0, 1]))])
        with self.assertRaisesRegex(ValueError, 'modular residual'):
            solve_module(problem, primes=[101, 103, 107], store=store)

    def test_corrupted_cache_noncanonical_pivot_profile_rejected(self):
        import numpy as np
        from hologuess.module import _expand, _cache_key
        problem = ModuleProblem([[1], [1]])
        _, exact, _, _ = _expand(problem)
        store = core.MemoryKernelStore()
        # Correct kernel/rank but not the matrix's canonical RREF pivot profile.
        store.put_many([(_cache_key(exact, 101), 101, 1, np.array([0, 1, 1, 100]))])
        with self.assertRaisesRegex(ValueError, 'pivot profile differs'):
            solve_module(problem, primes=[101, 103, 107], store=store)

    def test_corrupted_cache_structure_rejected(self):
        import numpy as np
        from hologuess.module import _expand, _cache_key
        problem = ModuleProblem([[1, 2], [2, 4]])
        _, exact, _, _ = _expand(problem)
        # Nonbinary mask, wrong nullity and lost free identity respectively.
        for payload in ([2, 0, 99, 1], [1, 0], [1, 0, 99, 2]):
            store = core.MemoryKernelStore()
            store.put_many([(_cache_key(exact, 101), 101, 1, np.array(payload))])
            with self.assertRaises(ValueError):
                solve_module(problem, primes=[101, 103, 107], store=store)

    def test_insufficient_prime_height_is_explicit(self):
        result = solve_module(ModuleProblem([[1], [10**100 + 123]]), primes=[3,5,7], max_primes=3)
        self.assertEqual(result.status, 'reconstruction_exhausted')
        self.assertFalse(result.metadata['full_training_kernel'])

    def test_input_guards(self):
        for columns in ([], [[1], [1,2]], [[]]):
            with self.assertRaises(ValueError): ModuleProblem(columns)
        with self.assertRaises(ValueError): solve_module(ModuleProblem([[1]]), primes=[3,3])
        with self.assertRaises(ValueError): solve_module(ModuleProblem([[1]]), min_primes=1)
        with self.assertRaises(TypeError): solve_module(ModuleProblem([[1.5], [3.0]]))


if __name__ == '__main__':
    unittest.main()
