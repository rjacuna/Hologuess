"""Independent action and end-to-end compatibility examples (Sage runtime)."""
import unittest
from math import factorial

try:
    from sage.all import QQ, GF, PolynomialRing, binomial
    from ore_algebra import OreAlgebra
    from hologuess.adapters import prepare, guess
    HAVE_SAGE = True
except ImportError:
    HAVE_SAGE = False


@unittest.skipUnless(HAVE_SAGE, 'SageMath and ore_algebra are required')
class OreAdapterTests(unittest.TestCase):
    def equivalent(self, actual, expected):
        self.assertTrue(actual)
        self.assertEqual(expected.leading_coefficient() * actual,
                         actual.leading_coefficient() * expected)

    def test_shift_factorial(self):
        R = PolynomialRing(QQ, 'n'); n = R.gen()
        A = OreAlgebra(R, 'Sn'); E = A.gen()
        found = guess([factorial(i) for i in range(24)], A, order=1, degree=1, holdout=3)
        self.equivalent(found.operator, E - (n + 1))

    def test_forward_difference_factorial(self):
        R = PolynomialRing(QQ, 'n'); n = R.gen()
        A = OreAlgebra(R, 'Fn'); delta = A.gen()
        found = guess([factorial(i) for i in range(24)], A, order=1, degree=1, holdout=3)
        self.equivalent(found.operator, delta - n)

    def test_taylor_exp(self):
        R = PolynomialRing(QQ, 'x'); A = OreAlgebra(R, 'Dx')
        found = guess([QQ(1) / factorial(i) for i in range(24)], A, order=1, degree=0, holdout=3)
        self.equivalent(found.operator, A.gen() - 1)

    def test_taylor_monomial_action(self):
        R = PolynomialRing(QQ, 'x'); x = R.gen(); A = OreAlgebra(R, 'Dx')
        p = prepare([2, 3, 5, 7, 11, 13], A, order=2, degree=2)
        column = p.columns[list(p.labels).index(x**2 * A.gen()**2)]
        # x² f'' has coefficients 0, 0, 2*a2, 6*a3.
        self.assertEqual(list(column), [0, 0, 10, 42])

    def test_algebraic_catalan(self):
        R = PolynomialRing(QQ, 'x'); x = R.gen(); A = OreAlgebra(R, 'C')
        data = [QQ(binomial(2*n, n)) / (n+1) for n in range(24)]
        found = guess(data, A, order=2, degree=1, holdout=3)
        self.equivalent(found.operator, x * A.gen()**2 - A.gen() + 1)

    def test_prime_field(self):
        F = GF(101); R = PolynomialRing(F, 'n'); A = OreAlgebra(R, 'Sn')
        found = guess([F(3)**n for n in range(24)], A, order=1, degree=0, holdout=3)
        self.equivalent(found.operator, A.gen() - 3)

    def test_multivariate_shift(self):
        R = PolynomialRing(QQ, ('n', 'k')); A = OreAlgebra(R, 'Sn', 'Sk')
        data = [[QQ(2)**n * QQ(3)**k for k in range(8)] for n in range(9)]
        found = guess(data, A, order=1, degree=0, holdout=4)
        self.assertGreaterEqual(len(found.operators), 2)
        for op in found.operators:
            # Evaluation on a distinct, unused grid point.
            val = sum(c * QQ(2)**(20+ex[0]) * QQ(3)**(21+ex[1]) for ex, c in op.polynomial().dict().items())
            self.assertEqual(val, 0)

    def test_mixed_shift_derivative(self):
        R = PolynomialRing(QQ, ('n', 'x')); A = OreAlgebra(R, 'Sn', 'Dx')
        data = [[QQ(2)**n / factorial(k) for k in range(9)] for n in range(8)]
        found = guess(data, A, order=1, degree=0, holdout=4)
        self.assertGreaterEqual(len(found.operators), 2)

    def test_parameterized_rational_sequence(self):
        K = PolynomialRing(QQ, 't').fraction_field(); t = K.gen()
        R = PolynomialRing(K, 'n'); n = R.gen(); A = OreAlgebra(R, 'Sn')
        data = [1 / (n0 + t) for n0 in range(20)]
        found = guess(data, A, order=1, degree=1, parameter_degree=1, holdout=3)
        self.equivalent(found.operator, (n+t+1) * A.gen() - (n+t))

    def test_qshift_parameterized(self):
        K = PolynomialRing(QQ, 'q').fraction_field(); q = K.gen()
        R = PolynomialRing(K, 'Q'); Q = R.gen()
        A = OreAlgebra(R, ('E', {Q:q*Q}, {}))
        data = [q**(n*n) for n in range(14)]
        found = guess(data, A, order=1, degree=2, parameter_degree=1, holdout=3)
        self.equivalent(found.operator, A.gen() - q*Q**2)

    def test_qshift_uses_current_qhsolve(self):
        K = PolynomialRing(QQ, 'q').fraction_field(); q = K.gen()
        R = PolynomialRing(K, 'Q'); Q = R.gen()
        A = OreAlgebra(R, ('E', {Q:q*Q}, {}))
        found = guess([q**(n*n) for n in range(32)], A, order=1, degree=2)
        self.assertEqual(found.metadata['backend'], 'qhsolve')
        self.assertEqual(found.operator.parent(), A)
        self.equivalent(found.operator, A.gen() - q*Q**2)
        degree_zero = guess([q**(n*n) for n in range(14)], A, order=1, degree=0)
        self.assertEqual(degree_zero.operators, ())
        parameter_zero = guess([q**(n*n) for n in range(14)], A, order=1, degree=2, parameter_degree=0)
        self.assertEqual(parameter_zero.operators, ())
        self.assertEqual(parameter_zero.metadata['parameter_degree_bound'], 0)

    def test_finite_field_parameter(self):
        K = PolynomialRing(GF(101), 't').fraction_field(); t = K.gen()
        R = PolynomialRing(K, 'n'); A = OreAlgebra(R, 'Sn')
        data = [t**n for n in range(14)]
        found = guess(data, A, order=1, degree=0, parameter_degree=1, holdout=3)
        self.equivalent(found.operator, A.gen() - t)

    def test_no_high_boundary_zero_extension(self):
        R = PolynomialRing(QQ, 'x'); A = OreAlgebra(R, 'Dx')
        p = prepare([1, 2, 3, 4, 5], A, order=2, degree=3)
        self.assertEqual(len(p.columns[0]), 3)
        with self.assertRaises(ValueError):
            prepare([1, 2], A, order=2)

    def test_rectangular_and_filtered_inputs(self):
        R = PolynomialRing(QQ, ('n', 'k')); A = OreAlgebra(R, 'Sn', 'Sk')
        with self.assertRaises(ValueError):
            prepare([[1, 2], [3]], A, order=0)
        with self.assertRaises(ValueError):
            prepare([[1, 2], [3, 4]], A, order=0, point_filter=lambda *n: False)

    def test_failed_holdout_is_not_returned(self):
        R = PolynomialRing(QQ, 'x'); A = OreAlgebra(R, 'Dx')
        data = [QQ(1) / factorial(i) for i in range(20)]
        data[-1] *= 2
        found = guess(data, A, order=1, degree=0, holdout=3)
        self.assertEqual(found.operators, ())
        self.assertEqual(found.status, 'holdout_failed')

    def test_reject_inexact_before_coercion(self):
        from sage.all import RR, CC
        R = PolynomialRing(QQ, 'n'); A = OreAlgebra(R, 'Sn')
        for data in ([1.0, 2.0, 3.0], [RR(1), RR(2), RR(3)], [CC(1), CC(2), CC(3)]):
            with self.assertRaises(TypeError):
                guess(data, A, order=1, degree=0)

    def test_start_and_callable(self):
        R = PolynomialRing(QQ, 'n'); n = R.gen(); A = OreAlgebra(R, 'Sn')
        found = guess(factorial, A, terms=24, start=3, order=1, degree=1, holdout=3)
        self.equivalent(found.operator, A.gen() - (n+1))


if __name__ == '__main__':
    unittest.main()
