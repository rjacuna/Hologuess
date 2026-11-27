import unittest
from sage.all import QQ, SR, PolynomialRing
from ore_algebra import OreAlgebra
from hologuess import guess, guess_qsequence, QSequence


class PublicInterface(unittest.TestCase):
    def test_kjj_exact_symbolic_rational_example(self):
        R = PolynomialRing(QQ,'n'); n = R.gen(); A = OreAlgebra(R,'Sn')
        result = guess([SR(1)/(i+1) for i in range(20)], A, order=1,degree=1)
        self.assertTrue(result.operators)
        expected = (n+2)*A.gen()-(n+1)
        self.assertEqual(result.operator.leading_coefficient()*expected,
                         expected.leading_coefficient()*result.operator)

    def test_explicit_backend_cannot_silently_fallback(self):
        A = OreAlgebra(PolynomialRing(QQ,'n'),'Sn')
        with self.assertRaises(ValueError):
            guess(list(range(20)), A, backend='qhsolve',path=[(1,1)])

    def test_bad_input_prime_is_skipped_for_rational_qdata(self):
        from fractions import Fraction
        result = guess_qsequence([{n:Fraction(1,101)} for n in range(20)],
            max_order=1,max_degree=0,primes=[101,103,107],max_points=80,band=(-4,4))
        self.assertEqual(result.metadata['skipped_input_primes'],[101])
        self.assertTrue(result.metadata['finite_window_valid'])

    def test_invalid_start_cannot_be_truncated(self):
        with self.assertRaises(ValueError): QSequence([{0:1},{1:1}],start=1.5)

    def test_single_variable_multivariate_parent(self):
        from hologuess import ModuleProblem, solve_module
        R = PolynomialRing(QQ,1,names=('t',)); t = R.gen()
        result = solve_module(ModuleProblem([[t+1],[t-1],[R(1)]]))
        self.assertEqual(len(result.basis),1)
        self.assertEqual(sum(c*f for c,f in zip(result.basis[0],[t+1,t-1,R(1)])),0)


if __name__ == '__main__': unittest.main()
