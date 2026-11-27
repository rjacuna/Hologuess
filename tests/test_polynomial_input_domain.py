"""Rational observations do not require a fraction-field output parent."""
import tempfile
from pathlib import Path
import unittest

from sage.all import QQ, ZZ, PolynomialRing
from ore_algebra import OreAlgebra
from hologuess import guess


class PolynomialInputDomainTests(unittest.TestCase):
    def test_rational_parameter_data_preserves_polynomial_operator_parent(self):
        K = PolynomialRing(QQ, 't'); t = K.gen()
        R = PolynomialRing(K, 'n'); n = R.gen()
        A = OreAlgebra(R, 'Sn')
        data = [1/(t+k) for k in range(20)]
        found = guess(data, A, path=[(1,1)], parameter_degree=1, holdout=4)
        expected = (n+t+1)*A.gen()-(n+t)
        self.assertEqual(found.operator.parent(), A)
        self.assertEqual(found.operator.leading_coefficient()*expected,
                         expected.leading_coefficient()*found.operator)
        self.assertTrue(found.metadata['exact_holdout'])
        self.assertEqual(found.metadata['parameter_names'], ('t',))

    def test_integer_parent_accepts_exact_rational_values_and_file(self):
        R = PolynomialRing(ZZ, 'n'); n = R.gen(); A = OreAlgebra(R, 'Sn')
        data = [QQ(1)/(k+1) for k in range(16)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'input.txt'
            path.write_text('\n'.join(map(str,data)))
            for source in (data, str(path)):
                found = guess(source, A, path=[(1,1)], holdout=3)
                self.assertEqual(found.operator.parent(), A)
                expected = (n+2)*A.gen()-(n+1)
                self.assertEqual(found.operator.leading_coefficient()*expected,
                                 expected.leading_coefficient()*found.operator)


if __name__=='__main__': unittest.main()
