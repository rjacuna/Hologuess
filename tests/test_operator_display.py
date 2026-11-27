"""Notebook LaTeX preserves the exact native Ore operators."""

from math import factorial
import pickle
import unittest

from sage.all import LaurentPolynomialRing, PolynomialRing, QQ, ZZ, binomial, latex
from ore_algebra import OreAlgebra

from hologuess import guess, guess_qsequence, operator_latex, set_latex_symbols


class OperatorDisplayTests(unittest.TestCase):
    def setUp(self):
        set_latex_symbols()
        self.addCleanup(set_latex_symbols)

    def test_shift_keeps_native_type_text_arithmetic_and_pickle(self):
        ring = QQ['n']
        n = ring.gen()
        algebra = OreAlgebra(ring, 'Sn')
        native = (n + 2) * algebra.gen() - n - 1
        values = [1 / QQ(k + 1) for k in range(16)]
        operator = guess(values, algebra, path=[(1, 1)], holdout=3).operator

        self.assertIs(type(operator), type(native))
        self.assertIs(operator.parent(), algebra)
        self.assertEqual(str(operator), str(native))
        self.assertEqual(repr(operator), repr(native))
        self.assertEqual(operator.list(), native.list())
        self.assertEqual(operator * n, native * n)
        self.assertEqual(latex(operator), r'\left(n + 2\right) S - n - 1')
        self.assertEqual(latex(native), r'\left(n + 2\right) \mathit{Sn} - n - 1')

        restored = pickle.loads(pickle.dumps(operator))
        self.assertIs(type(restored), type(native))
        self.assertIs(restored.parent(), algebra)
        self.assertEqual(restored, native)
        self.assertEqual(latex(restored), latex(operator))

    def test_derivative_alias_binds_powers_without_changing_existing_result(self):
        algebra = OreAlgebra(QQ['x'], 'Dx')
        taylor = [0 if k % 2 else QQ((-1)**(k//2)) / factorial(k) for k in range(16)]
        operator = guess(taylor, algebra, path=[(2, 0)], holdout=3).operator
        self.assertEqual(str(operator), 'Dx^2 + 1')
        self.assertEqual(latex(operator), 'D^{2} + 1')

        set_latex_symbols(D=r'\partial_x')
        self.assertEqual(latex(operator), r'{\partial_x}^{2} + 1')
        self.assertEqual(operator_latex(operator, Dx=r'\mathcal{L}'), r'{\mathcal{L}}^{2} + 1')
        self.assertEqual(latex(operator), r'{\partial_x}^{2} + 1')
        self.assertEqual(str(operator), 'Dx^2 + 1')

        set_latex_symbols(D='D', Dx=r'\nabla')
        self.assertEqual(latex(operator), r'{\nabla}^{2} + 1')
        with self.assertRaises(ValueError):
            set_latex_symbols(D=' ')
        self.assertEqual(latex(operator), r'{\nabla}^{2} + 1')
        set_latex_symbols()
        self.assertEqual(latex(operator), 'D^{2} + 1')

    def test_repeated_shifts_distinguish_axes_and_keep_parameter(self):
        parameters = QQ['t'].fraction_field()
        t = parameters.gen()
        ring = PolynomialRing(parameters, ('n', 'k'))
        algebra = OreAlgebra(ring, 'Sn', 'Sk')
        data = [[t**n * QQ(2)**k for k in range(6)] for n in range(7)]
        result = guess(data, algebra, path=[((1, 1), 0)], parameter_degree=1, holdout=4)
        shift_n = next(op for op in result.operators if set(map(tuple, op.dict())) == {(1, 0), (0, 0)})
        shift_k = next(op for op in result.operators if set(map(tuple, op.dict())) == {(0, 1), (0, 0)})

        self.assertEqual(latex(shift_n).replace(' ', ''), 'S_{n}-t')
        self.assertEqual(latex(shift_k).replace(' ', ''), 'S_{k}-2')
        self.assertEqual(operator_latex(shift_n, Sn='T').replace(' ', ''), 'T-t')
        self.assertEqual(latex(shift_n).replace(' ', ''), 'S_{n}-t')

    def test_mixed_shift_derivative_aliases_keep_coefficient_axes(self):
        ring = PolynomialRing(QQ, ('n', 'y'))
        algebra = OreAlgebra(ring, 'Sn', 'Dy')
        data = [[binomial(j, n) for j in range(6)] for n in range(7)]
        result = guess(data, algebra, path=[((1, 1), 1)], holdout=4)
        operator = next(op for op in result.operators if (1, 1) in map(tuple, op.dict()))

        self.assertEqual(latex(operator), r'\left(y - 1\right) S D + S + y D + 1')
        self.assertEqual(
            operator_latex(operator, Sn=r'\Sigma', Dy=r'\partial_y'),
            r'\left(y - 1\right) {\Sigma} {\partial_y} + {\Sigma} + y {\partial_y} + 1',
        )
        self.assertEqual(str(operator), '(y - 1)*Sn*Dy + Sn + y*Dy + 1')

    def test_q_recurrences_keep_forward_E_and_multiplication_Q(self):
        q = LaurentPolynomialRing(ZZ, 'q').gen()
        values = [q**(n*n) for n in range(24)]
        operator = guess_qsequence(values, max_order=1, max_degree=2).to_ore()
        self.assertEqual(latex(operator), '-E + q Q^{2}')
        self.assertEqual(str(operator), '-E + q*Q^2')
        set_latex_symbols(E=r'\mathcal{L}')
        self.assertEqual(latex(operator), r'-{\mathcal{L}} + q Q^{2}')
        self.assertEqual(str(operator), '-E + q*Q^2')

    def test_guessed_units_and_generators_leave_cached_native_elements_alone(self):
        algebra = OreAlgebra(QQ['n'], 'Sn')
        unit, generator = algebra.one(), algebra.gen()
        native_latex = (latex(unit), latex(generator))
        guess_unit, guess_generator = guess([0]*12, algebra, path=[(1, 0)], holdout=3).operators
        self.assertEqual((guess_unit, guess_generator), (unit, generator))
        self.assertIsNot(guess_unit, unit)
        self.assertIsNot(guess_generator, generator)

        set_latex_symbols(S='T')
        self.assertEqual(latex(guess_generator), 'T')
        self.assertEqual((latex(unit), latex(generator)), native_latex)


if __name__ == '__main__':
    unittest.main()
