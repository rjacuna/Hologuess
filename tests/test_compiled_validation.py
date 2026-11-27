"""Independent kernel and rejection checks for compiled fitting validation."""
import unittest
from unittest.mock import patch

import numpy as np
from sage.all import QQ, GF, PolynomialRing, matrix

from hologuess import ModuleProblem, solve_module
from hologuess.module import _annihilates_matrix, _modular_image


class CompiledValidationTests(unittest.TestCase):
    def assert_kernel(self, rows, result, field=QQ):
        M = matrix(field, rows)
        B = matrix(field, len(result.basis), M.ncols(),
                   [c for v in result.basis for c in v])
        self.assertTrue((M*B.transpose()).is_zero())
        self.assertEqual(B.rank(), len(result.basis))
        self.assertEqual(len(result.basis), M.ncols()-M.rank())

    def test_rational_fitting_certificate_is_not_repeated(self):
        rows = [[QQ(1)/(k+1)**100, QQ(7)/(13*(k+1)**100), QQ(0)]
                for k in range(8)]
        problem = ModuleProblem(zip(*rows), training_rows=6)
        with patch('hologuess.module._annihilates_matrix', wraps=_annihilates_matrix) as check:
            result = solve_module(problem)
        self.assertEqual(check.call_count, 1)
        self.assertEqual(result.metadata['exact_training_checks'], 1)
        self.assertTrue(result.metadata['exact_holdout'])
        self.assert_kernel(rows, result)

    def test_compiled_product_rejects_corrupted_vector(self):
        M = matrix(QQ, [[QQ(1)/3, QQ(2)/5], [QQ(7)/3, QQ(14)/5]])
        self.assertTrue(_annihilates_matrix(M, ((QQ(-6)/5, QQ(1)),)))
        self.assertFalse(_annihilates_matrix(M, ((QQ(-6)/5+1, QQ(1)),)))

    def test_stable_but_incorrect_modular_candidates_are_not_certified(self):
        # Wrong images agree across primes and have a plausible rank/profile.
        # Exact fitting, rather than CRT stability alone, must reject them.
        wrong = ((0,), np.array([[0, 1]], dtype=np.int64), False)
        with patch('hologuess.module._modular_image', return_value=wrong):
            result = solve_module(ModuleProblem([[1, 2], [2, 4]]),
                                  primes=[101, 103, 107], max_primes=3)
        self.assertEqual(result.status, 'reconstruction_exhausted')
        self.assertFalse(result.metadata['full_training_kernel'])
        self.assertEqual(result.metadata['exact_training_checks'], 2)

    def test_parameter_expansion_certifies_same_rational_identity(self):
        R = PolynomialRing(QQ, 't'); t = R.gen()
        with patch('hologuess.module._annihilates_matrix', wraps=_annihilates_matrix) as check:
            result = solve_module(ModuleProblem([[1/(t+1)], [t/(t+1)], [1]]))
        self.assertEqual(check.call_count, 1)
        self.assertEqual(result.basis, ((QQ(-1), QQ(-1), QQ(1)),))
        a, b, c = result.basis[0]
        self.assertEqual(a/(t+1)+b*t/(t+1)+c, 0)

    def test_finite_extension_and_separate_holdout_intersection(self):
        K = GF(9, 'a'); a = K.gen()
        with patch('hologuess.module._annihilates_matrix', wraps=_annihilates_matrix) as check:
            result = solve_module(ModuleProblem([[K(0), a], [K(0), K(1)]], training_rows=1))
        self.assertEqual(check.call_count, 1)
        self.assertEqual(result.holdout_failures, (0, 1))
        self.assertEqual(len(result.valid_basis), 1)
        u, v = result.valid_basis[0]
        self.assertEqual(a*u+v, 0)
        self.assertTrue(result.metadata['holdout_used_to_select'])

    def test_full_rank_empty_kernel_retains_rank_certificate(self):
        rows = [[2, 0], [0, 3]]
        result = solve_module(ModuleProblem(zip(*rows)), primes=[101, 103])
        self.assertEqual(result.status, 'no_relations')
        self.assertEqual(result.metadata['exact_training_backend'], 'full_rank_certificate')
        self.assert_kernel(rows, result)

    def test_invalid_modular_array_cannot_supply_a_rank_certificate(self):
        M = matrix(QQ, [[1, 2], [2, 4]])
        for bad in (np.array([[1, 2]]),
                    np.array([[1., 2.], [2., 4.]]),
                    np.array([[1, -1], [2, 4]]),
                    np.array([[1, 101], [2, 4]])):
            with self.subTest(array=repr(bad)):
                with self.assertRaisesRegex(ValueError, 'Invalid input-derived'):
                    _modular_image(M, 101, None, lambda p: bad)


if __name__ == '__main__':
    unittest.main()
