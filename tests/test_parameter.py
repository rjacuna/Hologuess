import unittest
try:
    from sage.all import QQ, PolynomialRing
    from hologuess.module import ModuleProblem
    from hologuess.parameter import interpolate_problem
    HAVE_SAGE=True
except ImportError:
    HAVE_SAGE=False


@unittest.skipUnless(HAVE_SAGE, 'SageMath required')
class ParameterTests(unittest.TestCase):
    def test_rational_observations_skip_pole_and_validate_holdout(self):
        R=PolynomialRing(QQ,'t'); K=R.fraction_field(); t=K.gen()
        values=[(t+1)**n/(t-2) for n in range(16)]
        problem=ModuleProblem([values[:-1],values[1:]], labels=('P_n','P_n+1'),training_rows=12)
        result=interpolate_problem(problem,t,band=(-3,3),workers=1)
        coefficients=[K.zero(),K.zero()]
        for (j,e),c in result.coefficients.items(): coefficients[j]+=c*t**e
        self.assertEqual(coefficients[0]+(t+1)*coefficients[1],0)
        self.assertNotEqual(coefficients[1],0)
        self.assertTrue(result.metadata['exact_all_observations'])
        self.assertEqual(result.metadata['holdout_observations'],3)

    def test_corrupted_holdout_rejected(self):
        R=PolynomialRing(QQ,'t'); K=R.fraction_field(); t=K.gen()
        values=[(t+1)**n/(t-2) for n in range(16)]
        right=values[1:]; right[-1]+=1
        problem=ModuleProblem([values[:-1],right],training_rows=12)
        with self.assertRaises(ArithmeticError):
            interpolate_problem(problem,t,band=(-3,3),workers=1)

if __name__=='__main__':unittest.main()
