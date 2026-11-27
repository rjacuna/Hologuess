"""Independent polynomial/rational action-evaluation and exact-check controls."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from sage.all import QQ, GF, PolynomialRing

from hologuess import ModuleProblem
from hologuess import parameter as implementation


class ParameterBatchingTests(unittest.TestCase):
    def setUp(self):
        self.R = PolynomialRing(QQ, 't')
        self.t = self.R.gen()
        self.K = self.R.fraction_field()

    def inspect(self, problem, callback):
        def intercepted(oracle, **options):
            callback(oracle)
            return SimpleNamespace(metadata={})
        with patch.object(implementation, 'interpolate_relation', intercepted):
            return implementation.interpolate_problem(problem, self.t)

    def test_integer_polynomial_matrix_uses_one_batch_and_integer_validator(self):
        t = self.t
        values = [(t+1)**n for n in range(8)]
        problem = ModuleProblem([values[:-1], values[1:]], training_rows=5)
        def check(oracle):
            with patch.object(implementation.core, 'eval_laurent',
                              wraps=implementation.core.eval_laurent) as evaluated:
                points, matrix = oracle.at(101, 4)
            self.assertEqual(evaluated.call_count, 1)
            for n in range(5):
                for j in range(2):
                    expected = [int(GF(101)(values[n+j](int(g)))) for g in points]
                    self.assertEqual(matrix[n,j].tolist(), expected)
            self.assertTrue(oracle.exact_validate({(0,0):1,(0,1):1,(1,0):-1}))
        result = self.inspect(problem, check)
        self.assertEqual(result.metadata['validation_ring'], 'Integer Ring')

    def test_rational_polynomial_coefficients_keep_rational_validator(self):
        t = self.t
        values = [(t+1)**n / 2 for n in range(8)]
        problem = ModuleProblem([values[:-1], values[1:]], training_rows=5)
        def check(oracle):
            points, matrix = oracle.at(101, 4)
            for n in range(5):
                self.assertEqual(matrix[n,0].tolist(),
                    [int(GF(101)(values[n](int(g)))) for g in points])
            self.assertTrue(oracle.exact_validate({(0,0):1,(0,1):1,(1,0):-1}))
        result = self.inspect(problem, check)
        self.assertEqual(result.metadata['validation_ring'], 'Rational Field')

    def test_shared_denominator_row_scaling_preserves_relations_and_point_prefix(self):
        t = self.K.gen()
        values = [(t+1)**n/(t-2) for n in range(8)]
        problem = ModuleProblem([values[:-1], values[1:]], training_rows=5)
        def check(oracle):
            points, matrix = oracle.at(101, 4)
            self.assertNotIn(2, points)
            for n in range(5):
                for j in range(2):
                    expected = [int(GF(101)((values[n+j]*(t-2))(int(g)))) for g in points]
                    self.assertEqual(matrix[n,j].tolist(), expected)
            longer, data = oracle.at(101, 7)
            self.assertEqual(points.tolist(), longer[:4].tolist())
            np.testing.assert_array_equal(matrix, data[:,:,:4])
            self.assertTrue(oracle.exact_validate({(0,0):1,(0,1):1,(1,0):-1}))
        result = self.inspect(problem, check)
        self.assertTrue(result.metadata['row_denominators_cleared'])

    def test_heterogeneous_denominators_remain_rational_without_lcm_expansion(self):
        t = self.K.gen()
        a, b = 1/(t-2), 1/(t-3)
        columns = [[(n+1)*a for n in range(7)],[(n+1)*b for n in range(7)],
                   [(n+1)*(a+b) for n in range(7)]]
        problem = ModuleProblem(columns, training_rows=5)
        def check(oracle):
            with patch.object(implementation.core, 'inv_mod',
                              wraps=implementation.core.inv_mod) as inversions:
                points, matrix = oracle.at(101, 4)
            self.assertLessEqual(inversions.call_count, 4)
            self.assertNotIn(2, points); self.assertNotIn(3, points)
            for n in range(5):
                for j in range(3):
                    expected = [int(GF(101)(columns[j][n](int(g)))) for g in points]
                    self.assertEqual(matrix[n,j].tolist(), expected)
            self.assertTrue(oracle.exact_validate({(0,0):1,(1,0):1,(2,0):-1}))
            self.assertFalse(oracle.exact_validate({(0,0):1,(1,0):1,(2,0):-2}))
        result = self.inspect(problem, check)
        self.assertFalse(result.metadata['row_denominators_cleared'])

    def test_heldout_pole_and_corruption_are_not_discarded(self):
        t = self.K.gen()
        left = [t+1]*6 + [(t+1)/(t-2)]
        right = [(t+1)**2]*6 + [(t+1)**2/(t-2)+1]
        def check(oracle):
            points, _ = oracle.at(101,4)
            self.assertNotIn(2,points)
            self.assertFalse(oracle.exact_validate({(0,0):1,(0,1):1,(1,0):-1}))
        self.inspect(ModuleProblem([left,right],training_rows=5),check)

    def test_bad_coefficient_prime_does_not_publish_partial_cache_state(self):
        t = self.t
        def check(oracle):
            for _ in range(2):
                with self.assertRaises(ZeroDivisionError): oracle.at(101,4)
            points, matrix = oracle.at(103,4)
            self.assertEqual(matrix.shape,(5,2,4))
            with self.assertRaises(ZeroDivisionError): oracle.at(101,4)
        values = [(t+1)**n/101 for n in range(6)]
        self.inspect(ModuleProblem([values[:-1],values[1:]]),check)

    def test_heterogeneous_row_keeps_constant_rational_entry(self):
        t = self.K.gen()
        data = [1/(t-2),1/(t-3),self.K(QQ(1)/3)]
        columns = [[(n+1)*value for n in range(7)] for value in data]
        def check(oracle):
            points, matrix = oracle.at(101,4)
            for n in range(5):
                for j in range(3):
                    self.assertEqual(matrix[n,j].tolist(),
                        [int(GF(101)(columns[j][n](int(g)))) for g in points])
            self.assertTrue(oracle.exact_validate({(0,0):-2,(0,1):1,
                                                  (1,0):-3,(1,1):1,(2,0):-6}))
        self.inspect(ModuleProblem(columns,training_rows=5),check)

    def test_heterogeneous_rational_line_reconstructs_end_to_end(self):
        t = self.K.gen()
        left = [1/(t-2) for n in range(9)]
        right = [(n+1)/(t-3) for n in range(9)]
        result = implementation.interpolate_problem(
            ModuleProblem([left,right,[a+b for a,b in zip(left,right)]],training_rows=6),
            t,band=(-3,3),max_points=256)
        weights = [self.K.zero() for _ in range(3)]
        for (j,e),c in result.coefficients.items(): weights[j] += c*t**e
        self.assertNotEqual(weights[2],0)
        self.assertEqual(weights[0],-weights[2])
        self.assertEqual(weights[1],-weights[2])
        self.assertFalse(result.metadata['row_denominators_cleared'])
        self.assertEqual(result.metadata['validation_mode'],'mixed_polynomial_rational')
        self.assertTrue(result.metadata['exact_all_observations'])


if __name__ == '__main__': unittest.main()
