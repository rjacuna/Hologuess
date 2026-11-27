"""Public Ore/SymPy guessing examples, translated to exact Hologuess obligations.

The inventory pins each upstream call and records differences in output scope.
These tests use Hologuess's public guess(), never an upstream guessing routine.
Upstream ore_algebra doctest examples are GPL-2.0-or-later; SymPy examples are
BSD-3-Clause. Mathematical reference operators/matrices below are independently
constructed. E denotes forward shift; upstream generator names are preserved.
"""
from itertools import product
from math import factorial, prod
from pathlib import Path
import json
import signal
import unittest

try:
    from sage.all import (QQ, ZZ, GF, SR, PolynomialRing, PowerSeriesRing,
                          binomial, fibonacci, matrix)
    from ore_algebra import OreAlgebra
    from hologuess import guess
    HAVE_SAGE = True
except ImportError:
    HAVE_SAGE = False


def _coefficient_powers(dimension, degree):
    return [p for p in product(range(degree+1), repeat=dimension) if sum(p)<=degree]


def _operator_vector(op, ring, orders, degree):
    """Coefficient coordinates, independent of Hologuess's action builder."""
    actions = list(product(*(range(r+1) for r in orders)))
    powers = _coefficient_powers(len(orders), degree)
    polynomials = op.polynomial().dict()
    if len(orders)>1:
        polynomials = {tuple(k):v for k,v in polynomials.items()}
    result = []
    for action in actions:
        key = action[0] if len(orders)==1 else action
        polynomial = ring(polynomials.get(key, 0))
        coefficients = polynomial.dict()
        if len(orders)>1:
            coefficients = {tuple(k):v for k,v in coefficients.items()}
        for power in powers:
            exponent = power[0] if len(orders)==1 else power
            result.append(coefficients.get(exponent, 0))
    return result


def _independent_action_matrix(data, constants, orders, degree, kinds):
    """Literal coefficient identities for shifts/Taylor derivatives on a grid."""
    dimension = len(orders)
    shape = (len(data),) if dimension==1 else (len(data), len(data[0]))
    points = list(product(*(range(n-r) for n,r in zip(shape,orders))))
    columns = [(v,u) for v in product(*(range(r+1) for r in orders))
               for u in _coefficient_powers(dimension,degree)]
    rows = []
    for at in points:
        row = []
        for action,power in columns:
            target = list(at); weight = constants.one()
            for axis,kind in enumerate(kinds):
                a,b,n = power[axis],action[axis],at[axis]
                if kind=='S':
                    target[axis] += b
                    weight *= constants(n)**a
                elif kind=='D':
                    if n<a:
                        weight = constants.zero(); break
                    target[axis] = n-a+b
                    weight *= prod(range(n-a+1,n-a+b+1))
                else:
                    raise ValueError('reference checker implements only shift/derivative')
            value = data[target[0]] if dimension==1 else data[target[0]][target[1]]
            row.append(weight*constants(value))
        rows.append(row)
    return matrix(constants,rows)


@unittest.skipUnless(HAVE_SAGE, 'SageMath and ore_algebra required')
class _ExactOreChecks(unittest.TestCase):
    """Exact operator and independent-space assertions shared by both sources."""

    def setUp(self):
        self._alarm_handler = signal.getsignal(signal.SIGALRM)
        def timeout(signum, frame):
            raise TimeoutError('Upstream compatibility case exceeded 180 seconds')
        signal.signal(signal.SIGALRM,timeout)
        signal.alarm(180)

    def tearDown(self):
        signal.alarm(0)
        signal.signal(signal.SIGALRM,self._alarm_handler)

    def equivalent(self, actual, expected):
        self.assertTrue(actual)
        expected = actual.parent()(expected)
        self.assertEqual(expected.leading_coefficient()*actual,
                         actual.leading_coefficient()*expected)

    def checked(self, found):
        self.assertTrue(found.operators)
        self.assertTrue(found.metadata['exact_training'])
        self.assertTrue(found.metadata['exact_holdout'])
        self.assertGreater(found.metadata['holdout_observations'],0)
        self.assertFalse(found.metadata['holdout_used_to_select'])

    def compare_space(self, found, data, ring, orders, degree, kinds):
        self.checked(found)
        constants = ring.base_ring().fraction_field()
        observed = _independent_action_matrix(data,constants,orders,degree,kinds)
        actual = matrix(constants,[_operator_vector(op,ring,orders,degree)
                                   for op in found.operators]).row_space()
        self.assertEqual(actual,observed.right_kernel())


@unittest.skipUnless(HAVE_SAGE, 'SageMath and ore_algebra required')
class UpstreamOreExamples(_ExactOreChecks):
    """All eleven guessing.py calls and its extra recurrence-source call."""

    def test_ore_symbolic_rational_coercion(self):
        ring = PolynomialRing(QQ,'n'); n = ring.gen(); A = OreAlgebra(ring,'Sn')
        data = [SR(QQ(1)/(i+1)) for i in range(10)]
        result = guess(data,A,path=[(1,1)],holdout=2)
        self.checked(result)
        self.equivalent(result.operator,(n+2)*A.gen()-(n+1))

    def test_ore_large_integer_order_six_degree_ninety(self):
        ring = PolynomialRing(ZZ,'n'); n = ring.gen(); A = OreAlgebra(ring,'Sn')
        data = [(2*i+1)**15*(1+2**i+3**i)**2 for i in range(1000)]
        result = guess(data,A,path=[(6,90)],holdout=20)
        self.checked(result)
        # Six exponentials have a constant-coefficient recurrence. Conjugating
        # it by p(n)=(2n+1)^15 and clearing denominators gives this polynomial
        # operator, independently of any reconstructed coefficient.
        zring = PolynomialRing(ZZ,'z'); z = zring.gen()
        constant = prod(z-a for a in (1,2,3,4,6,9))
        expected = sum((constant[j]*prod((2*(n+k)+1)**15 for k in range(7) if k!=j)
                        *A.gen()**j for j in range(7)),A.zero())
        self.assertEqual((result.operator.order(),result.operator.degree()),(6,90))
        self.equivalent(result.operator,expected)

    def test_ore_rational_parameter_with_polynomial_parent(self):
        parameters = PolynomialRing(QQ,'t'); t = parameters.gen()
        ring = PolynomialRing(parameters,'n'); n = ring.gen(); A = OreAlgebra(ring,'Sn')
        data = [1/(i+t)+t**i for i in range(100)]
        result = guess(data,A,path=[(2,2)],parameter_degree=4,holdout=8)
        self.checked(result)
        expected = ((-t+1)*n**2+(-2*t**2-t+2)*n-t**3-2*t**2)*A.gen()**2
        expected += ((t**2-1)*n**2+(2*t**3+3*t**2-2*t-1)*n+t**4+3*t**3+t**2-t)*A.gen()
        expected += (-t**2+t)*n**2+(-2*t**3+t)*n-t**4-t**3+t**2
        self.equivalent(result.operator,expected)

    def test_ore_catalan_algebraic(self):
        ring = PolynomialRing(ZZ,'x'); x = ring.gen(); A = OreAlgebra(ring,'C')
        data = [binomial(2*n,n)//(n+1) for n in range(10)]
        result = guess(data,A,path=[(2,1)],holdout=2)
        self.checked(result)
        self.equivalent(result.operator,-x*A.gen()**2+A.gen()-1)

    def test_ore_raw_finite_field_full_kernel(self):
        field = GF(1091); ring = PolynomialRing(field,'n'); A = OreAlgebra(ring,'Sn')
        data = [field(QQ(5*n+3)/(3*n+4)*fibonacci(n)**3) for n in range(200)]
        result = guess(data,A,path=[(5,3)],holdout=8)
        self.compare_space(result,data,ring,(5,),3,('S',))
        self.assertEqual(len(result.operators),2)

    def test_ore_hermite_pade_differential(self):
        field = GF(1091); ring = PolynomialRing(field,'x'); x = ring.gen(); A = OreAlgebra(ring,'Dx')
        data = [field(binomial(2*n,n)*fibonacci(n)**3) for n in range(2000)]
        result = guess(data,A,path=[(4,4)],holdout=20)
        self.checked(result)
        D = A.gen()
        expected = ((x**4+819*x**3+136*x**2+17*x+635)*D**4
                    +(14*x**3+417*x**2+952*x+605)*D**3
                    +(598*x**2+497*x+99)*D**2+(598*x+794)*D+893)
        self.equivalent(result.operator,expected)

    def test_ore_hermite_pade_algebraic_full_bounded_space(self):
        field = GF(1091); ring = PolynomialRing(field,'x'); x = ring.gen(); A = OreAlgebra(ring,'C')
        data = [field(binomial(2*n,n)*fibonacci(n)**3) for n in range(2000)]
        result = guess(data,A,path=[(16,64)],holdout=20)
        self.checked(result)
        # Binet and sum binom(2n,n)r^n x^n=(1-4rx)^(-1/2):
        # independently eliminate four square roots. No upstream guesser is run.
        K = ring.fraction_field(); Y = PolynomialRing(K,'y'); y = Y.gen()
        Z = PolynomialRing(Y,'z'); z = Z.gen()
        sqrt5 = field(5).sqrt(); alpha=(1+sqrt5)/2; beta=(1-sqrt5)/2
        roots = (alpha**3,beta**3,-alpha,-beta)
        weights = (1,-1,-3,3)
        squares = [K(w*w)/(125*(1-4*r*x)) for r,w in zip(roots,weights)]
        polynomial = y*y-squares[0]
        for square in squares[1:]:
            shifted = sum((Z(c)*(Z(y)-z)**j for j,c in enumerate(polynomial.list())),Z.zero())
            polynomial = Y(shifted.resultant(z*z-Y(square)))
        denominator = ring.one()
        for c in polynomial.list(): denominator=denominator.lcm(c.denominator())
        expected = sum((ring(c*denominator)*A.gen()**j for j,c in enumerate(polynomial.list())),A.zero())
        content = ring.zero()
        for c in expected.list():content=content.gcd(c)
        expected = A([c//content for c in expected.list()])
        expected_degree = max(c.degree() for c in expected.list())
        multiples = [x**j*expected for j in range(65-expected_degree)]
        actual = matrix(field,[_operator_vector(op,ring,(16,),64) for op in result.operators]).row_space()
        reference = matrix(field,[_operator_vector(op,ring,(16,),64) for op in multiples]).row_space()
        self.assertEqual(actual,reference)
        self.assertEqual(expected.order(),16)

    def _binomial_grid(self,kinds,names):
        ring = PolynomialRing(ZZ,tuple(names)); A = OreAlgebra(ring,*[k+v for k,v in zip(kinds,names)])
        data = [[binomial(n,k) for n in range(10)] for k in range(10)]
        degree = 0 if kinds==('S','S') else 1
        result = guess(data,A,path=[((1,1),degree)],holdout=5)
        self.compare_space(result,data,ring,(1,1),degree,kinds)

    def test_ore_multivariate_binomial_shifts(self):self._binomial_grid(('S','S'),('n','k'))
    def test_ore_multivariate_binomial_derivatives(self):self._binomial_grid(('D','D'),('x','y'))
    def test_ore_multivariate_binomial_shift_derivative(self):self._binomial_grid(('S','D'),('n','y'))
    def test_ore_multivariate_binomial_derivative_shift(self):self._binomial_grid(('D','S'),('x','k'))

    def test_ore_extra_exponential_rational_recurrence(self):
        ring = PolynomialRing(QQ,'n'); n = ring.gen(); A = OreAlgebra(ring,'Sn')
        data = [QQ((-3)**k)*(k+1)/(2*k+4)-QQ(2)**k*k**3/(k+3) for k in range(500)]
        result = guess(data,A,path=[(2,8)],holdout=20)
        self.compare_space(result,data,ring,(2,),8,('S',))
        self.assertEqual(result.operator.order(),2)


@unittest.skipUnless(HAVE_SAGE, 'SageMath and ore_algebra required')
class UpstreamSymPyExamples(_ExactOreChecks):
    """Independent mathematical cases from all six concrete/tests/test_guess functions.

    Closed-form/OGF/EGF presentation APIs differ. We test their exact coefficient
    equations, including nonlinear-product examples after the same ratio step,
    and explicitly reject the approximate-rationalization category.
    """
    def test_sympy_simple_recurrence_vector(self):
        ring = PolynomialRing(QQ,'n'); A = OreAlgebra(ring,'Sn'); E=A.gen()
        data = [fibonacci(k) for k in range(12)]
        result = guess(data,A,path=[(2,0)],holdout=3)
        self.checked(result);self.equivalent(result.operator,E**2-E-1)

    def test_sympy_named_simple_recurrence(self):
        ring = PolynomialRing(QQ,'n'); A=OreAlgebra(ring,'Sn'); E=A.gen()
        data = [1,1,1]
        for _ in range(15):data.append(5*data[-1]-3*data[-2]+8*data[-3])
        result = guess(data,A,path=[(3,0)],holdout=3)
        self.checked(result);self.equivalent(result.operator,E**3-5*E**2+3*E-8)

    def test_sympy_no_simple_recurrence(self):
        ring=PolynomialRing(QQ,'n');A=OreAlgebra(ring,'Sn')
        data=[0,2,15,74,12,3,0,1,2,85,4,5,63]
        # The upstream zero means no constant-coefficient recurrence within its
        # supported data budget, not that no unrestricted Ore relation exists.
        result=guess(data,A,path=[(r,0) for r in range(1,6)],holdout=2)
        self.assertEqual(result.operators,())

    def test_sympy_shifted_fibonacci_rational_generating_function(self):
        ring=PolynomialRing(QQ,'x');x=ring.gen();A=OreAlgebra(ring,'C')
        data=[fibonacci(k) for k in range(5,15)]
        result=guess(data,A,path=[(1,2)],holdout=2)
        self.checked(result);self.equivalent(result.operator,(1-x-x*x)*A.gen()-(5+3*x))

    def test_sympy_algebraic_generating_function(self):
        ring=PolynomialRing(QQ,'x');x=ring.gen();A=OreAlgebra(ring,'C')
        source=[1,2,5,14,41,124,383,1200,3799,12122,38919]
        # The source's specialized square-root search needs fewer observations
        # than the unrestricted algebraic box. Extend from the pinned formula
        # before fitting, and retain a separate suffix for exact validation.
        series_ring=PowerSeriesRing(QQ,'x',default_prec=30);s=series_ring.gen()
        series=(1-4*s+2*s*s+s**4)**(-QQ(1)/2)
        data=[series[k] for k in range(28)]
        self.assertEqual(data[:len(source)],source)
        result=guess(data,A,path=[(2,4)],holdout=4)
        self.checked(result)
        self.equivalent(result.operator,(1-4*x+2*x*x+x**4)*A.gen()**2-1)

    def test_sympy_rational_generating_function_with_zero(self):
        ring=PolynomialRing(QQ,'x');x=ring.gen();A=OreAlgebra(ring,'C')
        data=list(map(QQ,['3/2','11/2','0','-121/2','-363/2','121','4719/2','11495/2','-8712','-178717/2']))
        result=guess(data,A,path=[(1,2)],holdout=2)
        self.checked(result);self.equivalent(result.operator,(1-3*x+11*x*x)*A.gen()-(x+QQ(3)/2))

    def test_sympy_factorial_exponential_generating_function(self):
        ring=PolynomialRing(QQ,'x');x=ring.gen();A=OreAlgebra(ring,'C')
        data=[QQ(factorial(k))/factorial(k) for k in range(12)]
        result=guess(data,A,path=[(1,1)],holdout=3)
        self.checked(result);self.equivalent(result.operator,(1-x)*A.gen()-1)

    def test_sympy_linear_exponential_generating_function(self):
        ring=PolynomialRing(QQ,'x');x=ring.gen();A=OreAlgebra(ring,'Dx')
        data=[QQ(k+1)/factorial(k) for k in range(12)]
        result=guess(data,A,path=[(1,1)],holdout=3)
        self.checked(result);self.equivalent(result.operator,(x+1)*A.gen()-(x+2))

    def test_sympy_factorial_product_presentations(self):
        ring=PolynomialRing(QQ,'n');n=ring.gen();A=OreAlgebra(ring,'Sn')
        # evaluate=False, default evaluation, and custom variable names all
        # describe this same sequence. Presentation flags are not HG options.
        result=guess([1,2,6,24,120],A,path=[(1,1)],holdout=1)
        self.checked(result);self.equivalent(result.operator,A.gen()-(n+2))

    def test_sympy_iterated_product_ratio(self):
        ring=PolynomialRing(QQ,'n');n=ring.gen();A=OreAlgebra(ring,'Sn')
        source=[1,2,7,42,429,7436,218348,10850216]
        # This closed-form guess is nested hypergeometric, outside linear
        # P-recursive guessing for the original ASM sequence. Its ratio of
        # successive ratios is rational, exactly the iterated-guess step.
        second_ratios=[QQ(source[k+2])*source[k]/source[k+1]**2 for k in range(len(source)-2)]
        # Exact product formula supplies extra independent suffix observations.
        extended=[prod(QQ(factorial(3*j+1))/factorial(m+j) for j in range(m)) for m in range(1,17)]
        self.assertEqual(extended[:8],source)
        ratios=[extended[k+2]*extended[k]/extended[k+1]**2 for k in range(len(extended)-2)]
        self.assertEqual(ratios[:len(second_ratios)],second_ratios)
        result=guess(ratios,A,path=[(1,4)],holdout=3)
        self.checked(result)
        self.compare_space(result,ratios,ring,(1,),4,('S',))

    def test_sympy_short_no_closed_form(self):
        ring=PolynomialRing(QQ,'n');A=OreAlgebra(ring,'Sn')
        result=guess([1,0,2],A,path=[(1,0)],holdout=1)
        self.assertEqual(result.operators,())

    def test_sympy_rationalize_is_outside_exact_input(self):
        ring=PolynomialRing(QQ,'n');A=OreAlgebra(ring,'Sn')
        with self.assertRaises(TypeError):guess([0.5,0.25,0.125],A,path=[(1,0)])


class UpstreamInventory(unittest.TestCase):
    def test_every_upstream_call_has_an_explicit_coverage_account(self):
        inventory=json.loads(Path(__file__).with_name('upstream_ore_inventory.json').read_text())
        counts={}
        for source in inventory['sources']:
            counts[source['id']]=len(source['cases'])
            for case in source['cases']:
                self.assertIn(case['status'],('covered','covered_with_semantic_translation','outside_exact_input_scope'))
                owner=globals()[case['test'].split('.')[0]]
                self.assertTrue(callable(getattr(owner,case['test'].split('.')[1])))
                self.assertTrue(case['source_expression'])
                self.assertTrue(case['verification'])
        self.assertEqual(counts,{'ore_algebra_guessing':11,'ore_algebra_extra':1,'sympy_guess':19})


if __name__=='__main__':
    unittest.main()
