"""FriCAS mantepse mathematical examples through public Hologuess interfaces.

Every original assertion, including xfails and private evaluator/options tests,
is accounted for in upstream_fricas_inventory.json. Exact nonlinear equations
are finite feature relations in ModuleProblem; no FriCAS runtime is required.
Upstream example provenance/license: tests/upstream_data/fricas/LICENSE.txt.
"""
from itertools import product
from math import factorial, prod
from pathlib import Path
import json
import signal
import unittest

try:
    from sage.all import (QQ, ZZ, GF, PolynomialRing, PowerSeriesRing,
                          NumberField, binomial, matrix, vector)
    from ore_algebra import OreAlgebra
    from hologuess import guess, ModuleProblem, solve_module
    HAVE_SAGE = True
except ImportError:
    HAVE_SAGE = False
    QQ = None


def _scalar_matrix(columns, constants, parameter_ring=None):
    """Independent exact row expansion, without Hologuess's internal helpers."""
    if parameter_ring is None:
        return matrix(constants,[list(row) for row in zip(*columns)])
    field=parameter_ring.fraction_field(); rows=[]
    for observations in zip(*columns):
        values=[field(v) for v in observations]
        denominator=parameter_ring.one()
        for value in values:denominator=denominator.lcm(value.denominator())
        polynomials=[parameter_ring(value*denominator) for value in values]
        dictionaries=[p.dict() for p in polynomials]
        keys=set().union(*(d.keys() for d in dictionaries))
        for key in sorted(keys):rows.append([d.get(key,0) for d in dictionaries])
    return matrix(constants,len(rows),len(columns),sum(rows,[]))


def _series_columns(values, support, base=QQ, parameter=None):
    """Support is (powers of f,f',f'', x-degree, parameter-degree)."""
    ring=PowerSeriesRing(base,'x',default_prec=len(values));x=ring.gen()
    # A finite coefficient list is a truncation, not an exact polynomial with
    # all later coefficients known to vanish. Keep that distinction through
    # derivative losses and subsequent multiplication by x.
    f=ring(values,prec=len(values))
    functions=(f,f.derivative(),f.derivative().derivative())
    features=[]
    count=len(values)
    for a,b,c,d,h in support:
        value=x**d*functions[0]**a*functions[1]**b*functions[2]**c
        if h:value*=parameter**h
        features.append(value)
        count=min(count,value.precision_absolute())
    return [[value[n] for n in range(int(count))] for value in features]


def _nonlinear_shift_columns(values,support):
    """Support entries are (shift-power tuple, polynomial n-degree)."""
    maximum=len(support[0][0])-1
    return [[n**degree*prod(values[n+j]**power for j,power in enumerate(powers))
             for n in range(len(values)-maximum)] for powers,degree in support]


@unittest.skipUnless(HAVE_SAGE,'SageMath and ore_algebra required')
class FriCASExamples(unittest.TestCase):
    def setUp(self):
        self._old_alarm=signal.getsignal(signal.SIGALRM)
        def timeout(signum,frame):raise TimeoutError('FriCAS compatibility case exceeded 180 seconds')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(180)

    def tearDown(self):
        signal.alarm(0);signal.signal(signal.SIGALRM,self._old_alarm)

    def features(self,columns,expected=(),*,constants=QQ,parameter_ring=None,
                 holdout=3,untouched=True,empty=False):
        count=len(columns[0]);training=count-holdout if holdout else None
        result=solve_module(ModuleProblem(columns,training_rows=training,name=self.id()))
        independent=_scalar_matrix(columns,constants,parameter_ring)
        actual=matrix(constants,len(result.valid_basis),len(columns),
                      [c for row in result.valid_basis for c in row]).row_space()
        self.assertEqual(actual,independent.right_kernel())
        self.assertTrue(result.metadata['exact_training'])
        self.assertFalse(result.metadata['all_indices_proved'])
        if holdout and untouched:
            self.assertTrue(result.metadata['exact_holdout'])
            self.assertFalse(result.metadata['holdout_used_to_select'])
        for coefficients in expected:self.assertIn(vector(constants,coefficients),actual)
        if empty:self.assertEqual(actual.dimension(),0)
        return result

    def equivalent(self,found,expected):
        self.assertTrue(found.operators)
        actual=found.operator;expected=actual.parent()(expected)
        self.assertEqual(expected.leading_coefficient()*actual,
                         actual.leading_coefficient()*expected)
        self.assertTrue(found.metadata['exact_training'])
        self.assertTrue(found.metadata['exact_holdout'])
        self.assertFalse(found.metadata['holdout_used_to_select'])

    def test_rational_integer_and_algebraic_number_examples(self):
        # Literal source prefixes, followed by the pinned rational expression's
        # exact values, make the unrestricted feature box independently testable.
        cases=[([1,1],lambda n:QQ(1),0,0,[1,-1]),
               ([0,1,2],lambda n:QQ(n),0,1,[1,0,-1]),
               ([1,2,3],lambda n:QQ(n+1),0,1,[1,-1,-1]),
               ([1,QQ(1)/2,QQ(1)/3],lambda n:QQ(1)/(n+1),1,0,[1,1,-1]),
               ([0,1,4,9],lambda n:QQ(n*n),0,2,[1,0,0,-1]),
               ([0,QQ(1)/2,QQ(2)/3,QQ(3)/4],lambda n:QQ(n)/(n+1),1,1,[1,1,0,-1]),
               ([QQ(1)/2,QQ(2)/3,QQ(3)/4,QQ(4)/5],lambda n:QQ(n+1)/(n+2),1,1,[2,1,-1,-1]),
               ([0,1,4,9,16],lambda n:QQ(n*n),0,2,[1,0,0,-1])]
        for prefix,function,d,p,expected in cases:
            with self.subTest(prefix=prefix):
                values=[function(n) for n in range(16)]
                self.assertEqual(values[:len(prefix)],prefix)
                columns=[[n**a*values[n] for n in range(16)] for a in range(d+1)]
                columns += [[QQ(n)**b for n in range(16)] for b in range(p+1)]
                self.features(columns,[expected])

    def test_rational_data_budget_and_zero_examples(self):
        self.features([[0,1,4],[1,1,1],[0,1,2]],holdout=0,empty=True)
        ring=PolynomialRing(QQ,'n');A=OreAlgebra(ring,'Sn')
        one=guess([0],A,path=[(0,0)])
        self.assertFalse(one.metadata['all_indices_proved'])
        self.assertIsNone(one.metadata['exact_holdout'])
        zeros=guess([0,0],A,path=[(0,0)],holdout=1)
        self.equivalent(zeros,A.one())

    def test_rational_equation_with_singular_index(self):
        prefix=list(map(QQ,[3,4,QQ(7)/2,QQ(18)/5,QQ(11)/3,QQ(26)/7]))
        values=[QQ(2*(2*n+3))/(n+2) if n!=1 else QQ(4) for n in range(16)]
        self.assertEqual(values[:6],prefix)
        columns=[[n**a*values[n] for n in range(16)] for a in range(3)]
        columns += [[QQ(n)**b for n in range(16)] for b in range(3)]
        self.features(columns,[[2,-1,-1,-6,2,4]])
        # Cancelling n-1 would incorrectly change the exceptional source value.
        self.assertNotEqual(values[1],QQ(2*(2*1+3))/(1+2))

    def test_q_exponential_and_mixed_rational_examples(self):
        ring=PolynomialRing(QQ,'q');q=ring.gen()
        for prefix in (3,6,7):
            with self.subTest(prefix=prefix):
                values=[q**n if prefix==3 else n**3*q**n+q**(2*n) for n in range(14)]
                powers=[(a,b) for b in range(3) for a in range(4)] if prefix!=3 else [(0,0),(0,1)]
                columns=[values]+[[n**a*q**(b*n) for n in range(14)] for a,b in powers]
                coefficients=[1]+[-int((a,b) in ((3,1),(0,2))) for a,b in powers]
                if prefix==3:coefficients=[1,0,-1]
                self.features(columns,[coefficients],parameter_ring=ring)

    def test_finite_field_polynomial_parameter(self):
        field=GF(5);ring=PolynomialRing(field,'z');z=ring.gen()
        values=[(n+1)*z for n in range(12)]
        self.features([values,[z]*12,[n*z for n in range(12)]],[[1,-1,-1]],
                      constants=field,parameter_ring=ring)

    def test_finite_field_rational_examples_and_xfails(self):
        for length in (3,4):
            with self.subTest(characteristic=2,source_length=length):
                field=GF(2);values=[field(n+1) for n in range(14)]
                self.assertEqual(values[:length],[field(v) for v in [1,0,1,0][:length]])
                self.features([values,[field(1)]*14,[field(n) for n in range(14)]],
                              [[1,1,1]],constants=field)
        for field,values in ((GF(2),[1,0,1,1]),
                             (GF(3),[binomial(2*n,n) for n in range(7)]),
                             (GF(7),[1,0,1,0,1,0,1])):
            values=list(map(field,values));count=len(values)
            columns=[values,[field(n)*values[n] for n in range(count)],
                     [field(1)]*count,[field(n) for n in range(count)]]
            self.features(columns,constants=field,holdout=1,untouched=False,
                          empty=field.characteristic()!=2)

    def test_pade_rational_series(self):
        ring=PolynomialRing(QQ,'x');x=ring.gen();A=OreAlgebra(ring,'C')
        result=guess([1]*12,A,path=[(1,1)],holdout=3)
        self.equivalent(result,(1-x)*A.gen()-1)
        negative=guess([1,1,2,3],A,path=[(1,1)],holdout=1)
        self.assertEqual(negative.operators,())
        values=[1,1]
        for _ in range(22):values.append(values[-1]+values[-2])
        self.assertEqual(values[:4],[1,1,2,3])
        result=guess(values,A,path=[(1,2)],holdout=4)
        self.equivalent(result,(1-x-x*x)*A.gen()-1)

    def test_pade_algebraic_constants_and_parameter(self):
        u=PolynomialRing(QQ,'u').gen();K=NumberField(u*u-2,'a');a=K.gen()
        for symbolic in (False,True):
            with self.subTest(symbolic_initial=symbolic):
                coefficients=PolynomialRing(K,'z') if symbolic else K
                z=coefficients.gen() if symbolic else K.zero()
                values=[z,coefficients(1),coefficients(1)]
                for n in range(3,24):values.append(a*values[-1]+3*values[-2]+values[-3])
                ring=PolynomialRing(coefficients,'x');x=ring.gen();A=OreAlgebra(ring,'C')
                result=guess(values,A,path=[(1,3)],parameter_degree=1,holdout=5)
                numerator=z+(1-a*z)*x+(1-a-3*z)*x*x
                self.equivalent(result,(1-a*x-3*x*x-x**3)*A.gen()-numerator)

    def test_algebraic_catalan_short_source_prefixes(self):
        ring=PolynomialRing(QQ,'x');x=ring.gen();A=OreAlgebra(ring,'C')
        for prefix in (4,6):
            with self.subTest(source_length=prefix):
                values=[binomial(2*n+1,n)/QQ(2*n+1) for n in range(20)]
                self.assertEqual(values[:prefix],[1,1,2,5,14,42][:prefix])
                result=guess(values,A,path=[(2,1)],holdout=4)
                self.equivalent(result,x*A.gen()**2-A.gen()+1)

    def test_p_recursive_central_binomial_and_budget(self):
        ring=PolynomialRing(QQ,'n');n=ring.gen();A=OreAlgebra(ring,'Sn')
        values=[binomial(2*n,n) for n in range(20)]
        result=guess(values,A,path=[(1,1)],holdout=4)
        self.equivalent(result,(n+1)*A.gen()-2*(2*n+1))
        # Native #0 assertions depend on its inhomogeneity/safety search budget.
        # A deliberately too-small constant-coefficient box is independently empty.
        for length in (4,5):
            empty=guess(values[:length],A,path=[(1,0)],holdout=1)
            self.assertEqual(empty.operators,())

    def test_p_recursive_leading_zeros_are_not_a_closed_form(self):
        ring=PolynomialRing(QQ,'n');n=ring.gen();A=OreAlgebra(ring,'Sn')
        values=[0,0,0,1,2,3,4,5]
        result=guess(values,A,path=[(1,1)],holdout=2)
        self.equivalent(result,(n-2)*A.gen()-(n-1))
        self.assertNotEqual(values,[k-2 for k in range(8)])

    def test_mixed_index_q_recurrence(self):
        ring=PolynomialRing(QQ,'q');q=ring.gen();values=[ring(1),ring(1)]
        for n in range(18):values.append(q**(n+1)*values[n+1]+n*values[n])
        support=[(j,a,b,c) for j in range(3) for a,b in ((0,0),(1,0),(0,1)) for c in range(2)]
        columns=[[n**a*q**(b*n+c)*values[n+j] for n in range(len(values)-2)] for j,a,b,c in support]
        expected=[int((j,a,b,c)==(0,1,0,0))+int((j,a,b,c)==(1,0,1,1))-int((j,a,b,c)==(2,0,0,0)) for j,a,b,c in support]
        self.features(columns,[expected],parameter_ring=ring,holdout=4)

    def test_many_parameters_and_exceptional_initial_index(self):
        parameters=PolynomialRing(QQ,('a','z'));a,z=parameters.gens()
        ring=PolynomialRing(parameters,'n');A=OreAlgebra(ring,'Sn')
        values=[z*a**i for i in range(16)]
        self.equivalent(guess(values,A,path=[(1,0)],parameter_degree=1,holdout=3),A.gen()-a)
        coefficients=PolynomialRing(QQ,('a00','a01','a02','a03','a04'))
        values=list(coefficients.gens())
        for n in range(5,16):values.append(sum(values[-4:]))
        ring=PolynomialRing(coefficients,'n');n=ring.gen();A=OreAlgebra(ring,'Sn');E=A.gen()
        result=guess(values,A,path=[(4,1)],parameter_degree=0,holdout=3)
        self.equivalent(result,n*(E**4-E**3-E**2-E-1))

    def test_finite_support_homogeneous_xfails(self):
        ring=PolynomialRing(QQ,'n');A=OreAlgebra(ring,'Sn')
        for place in range(3):
            values=[int(n==place) for n in range(8)]
            result=guess(values,A,path=[(1,1)],holdout=2)
            self.assertTrue(result.operators)
            columns=[[QQ(n)**a*values[n+j] for n in range(7)] for j in range(2) for a in range(2)]
            # Entire finite bounded relation space, including singular coefficients.
            reference=_scalar_matrix(columns,QQ).right_kernel()
            actual=matrix(QQ,result.module_result.valid_basis).row_space()
            self.assertEqual(actual,reference)

    def test_lambert_w_algebraic_differential_equation(self):
        values=[QQ(n)**n/factorial(n) for n in range(36)]
        self.assertEqual(values[:4],[1,1,2,QQ(9)/2])
        support=[(a,b,0,d,0) for a in range(4) for b in range(4-a) for d in range(2)]
        expected=[int((a,b,d)==(0,1,1))-int((a,b,d)==(3,0,0))+int((a,b,d)==(2,0,0)) for a,b,c,d,h in support]
        self.features(_series_columns(values,support),[expected],holdout=8)

    def test_jacobi_sn_algebraic_differential_equations(self):
        parameters=PolynomialRing(QQ,'m');m=parameters.gen();values=[parameters.zero(),parameters.one()]
        for n in range(48):
            cube=sum((values[i]*values[j]*values[n-i-j] for i in range(n+1) for j in range(n-i+1)),parameters.zero())
            values.append((-(1+m)*values[n]+2*m*cube)/((n+2)*(n+1)))
        for degree in (0,1):
            with self.subTest(coefficient_degree=degree):
                support=[(a,b,c,d,h) for a,b,c in product(range(5),repeat=3)
                         if a+b+c<=4 and b+2*c<=2 for d in range(degree+1) for h in range(2)]
                expected=[int((a,b,c,d,h)==(1,0,1,0,0))-int((a,b,c,d,h)==(0,2,0,0,0))
                          -int((a,b,c,d,h)==(4,0,0,0,1))+int((a,b,c,d,h)==(0,0,0,0,0))
                          for a,b,c,d,h in support]
                self.features(_series_columns(values,support,parameters,m),[expected],parameter_ring=parameters,holdout=10)

    def test_q_carlitz_functional_equation(self):
        parameters=PolynomialRing(QQ,'q');q=parameters.gen();values=[parameters.one()]
        for n in range(1,18):values.append(sum((values[i]*values[n-1-i]*q**(n-1-i) for i in range(n)),parameters.zero()))
        prefix=[1,1,q+1,q**3+q*q+2*q+1,
                q**6+q**5+2*q**4+3*q**3+3*q*q+3*q+1,
                q**10+q**9+2*q**8+3*q**7+5*q**6+5*q**5+7*q**4+7*q**3+6*q*q+4*q+1]
        self.assertEqual(values[:6],prefix)
        T=PowerSeriesRing(parameters,'x',default_prec=18);x=T.gen();f=T(values);fq=T([v*q**n for n,v in enumerate(values)])
        support=[(a,b,d) for a in range(3) for b in range(3-a) for d in range(2)]
        columns=[[(x**d*f**a*fq**b)[n] for n in range(18)] for a,b,d in support]
        expected=[int((a,b,d)==(1,0,0))-int((a,b,d)==(0,0,0))-int((a,b,d)==(1,1,1)) for a,b,d in support]
        self.features(columns,[expected],parameter_ring=parameters,holdout=4)

    def test_somos_four_nonlinear_recurrence(self):
        values=[QQ(1)]*4
        for n in range(15):values.append((values[n+1]*values[n+3]+values[n+2]**2)/values[n])
        support=[(powers,0) for powers in product(range(3),repeat=5)
                 if sum(powers)==2 and sum(j*e for j,e in enumerate(powers))==4]
        expected=[-int(p==(1,0,0,0,1))+int(p==(0,1,0,1,0))+int(p==(0,0,2,0,0)) for p,d in support]
        for count in (19,13,7):
            with self.subTest(source_length=count):
                self.features(_nonlinear_shift_columns(values[:count],support),[expected],holdout=1 if count==7 else 3)

    def test_nested_products_and_somos_six_equation(self):
        values=[prod((prod(factorial(i) for i in range(j)) for j in range(n)),start=1) for n in range(18)]
        support=[(powers,d) for powers in product(range(5),repeat=4)
                 if sum(powers)==4 and sum(j*e for j,e in enumerate(powers))==6 for d in range(2)]
        expected=[-int(p==(0,3,0,1) and d==0)+int(p==(1,0,3,0) and d in (0,1)) for p,d in support]
        self.features(_nonlinear_shift_columns(values,support),[expected],holdout=4)
        third=[QQ(values[n+3])*values[n+1]**3/(QQ(values[n+2])**3*values[n]) for n in range(15)]
        ring=PolynomialRing(QQ,'n');n=ring.gen();A=OreAlgebra(ring,'Sn')
        result=guess(third,A,path=[(1,1)],holdout=3)
        self.equivalent(result,(n+1)*A.gen()-(n+2))
        rebuilt=list(map(QQ,values[:3]))
        for k,c in enumerate(third):rebuilt.append(c*rebuilt[k]*rebuilt[k+2]**3/rebuilt[k+1]**3)
        self.assertEqual(rebuilt,values)

    def test_nonhomogeneous_nonlinear_recurrence(self):
        values=[QQ(1),QQ(2)]
        for _ in range(6):values.append(values[-1]**2*values[-2])
        support=[(p,0) for p in product(range(4),repeat=3)
                 if sum(p)<=3 and sum(j*e for j,e in enumerate(p))==2]
        expected=[int(p==(0,0,1))-int(p==(1,2,0)) for p,d in support]
        self.features(_nonlinear_shift_columns(values,support),[expected],holdout=1)

    def test_sparse_zero_termination_regressions(self):
        data_sets=[[0,1,0,0,0,2,0,7,0,0,0,15,0,0,0],
                   [1,1,2,0,0,3,0,0,2,0,0]]
        # The source's long sparse sequence exercises checking termination;
        # no particular guessed equation is promised by that assertion.
        data_sets.append([1,0,1,2,0,1,0,1,0,1,-2,0,0,0,1,0,-3,0,-1,0,1,2,0,-3,0,-2,0,1,0,5,0,-2,0,-3,0,1,-2,0,8,0,0,0,-4,0,1,0,-7,0,10,0,3,0,-5,0,1,2,0,-15,0,10,0,7,0,-6,0,1,0,9,0,-25,0,7,0,12,0,-7,0,1,-2,0,24,0,-35,0,0,0,18,0,-8,0,1,0,-11,0,49,0,-42,0,-12,0])
        support=[(p,0) for p in product(range(3),repeat=3) if sum(p)<=2]
        for values in data_sets:
            with self.subTest(length=len(values)):
                self.features(_nonlinear_shift_columns(values,support),holdout=2,untouched=False)

    def test_leading_zero_functional_equation(self):
        values=[0,0,0,1,0,0,0]
        # f(x)=x^3 satisfies f(2x)-8f(x)=0, including initial zero coefficients.
        columns=[values,[2**n*v for n,v in enumerate(values)]]
        self.features(columns,[[-8,1]],holdout=2)

    def test_exponential_rational_candidate_family(self):
        values=[n*(n+2)**n for n in range(22)]
        self.assertEqual(values[:5],[0,3,32,375,5184])
        support=[(a,b) for b in (1,2,3) for a in range(3)]
        columns=[values]+[[n**a*(n+b)**n for n in range(22)] for a,b in support]
        expected=[1]+[-int((a,b)==(1,2)) for a,b in support]
        self.features(columns,[expected],holdout=5)

    def test_binomial_rational_parameter_candidate_family(self):
        parameters=PolynomialRing(QQ,'k');k=parameters.gen()
        def term(a,b,n):
            return parameters(prod(((a*k+b)*n-j for j in range(n)),start=parameters.one())/factorial(n)/((a*k+b-1)*n+1))
        values=[term(1,1,n) for n in range(13)]
        self.assertEqual(values[:5],[1,1,k+1,(3*k*k+5*k+2)/2,(8*k**3+18*k*k+13*k+3)/3])
        support=[(a,b) for a in (0,1) for b in (1,2,3)]
        columns=[values]+[[term(a,b,n) for n in range(13)] for a,b in support]
        expected=[1]+[-int((a,b)==(1,1)) for a,b in support]
        self.features(columns,[expected],parameter_ring=parameters,holdout=3)

    def test_binomial_rational_finite_field_candidate_family(self):
        field=GF(3001)
        support=list(product((6,7,8),(2,3,4),(2,3,4)))
        def term(a,b,d,n):return field(binomial(a*n+b,n))/field(n+d)
        values=[term(7,3,3,n) for n in range(45)]
        columns=[values]+[[term(a,b,d,n) for n in range(45)] for a,b,d in support]
        expected=[1]+[-int((a,b,d)==(7,3,3)) for a,b,d in support]
        self.features(columns,[expected],constants=field,holdout=10)

    def test_robbins_nested_product(self):
        values=[prod((QQ(factorial(3*j+1))/factorial(n+j) for j in range(n)),start=QQ(1)) for n in range(20)]
        self.assertEqual(values[:8],[1,1,2,7,42,429,7436,218348])
        second=[values[n+2]*values[n]/values[n+1]**2 for n in range(18)]
        ring=PolynomialRing(QQ,'n');n=ring.gen();A=OreAlgebra(ring,'Sn')
        result=guess(second,A,path=[(1,4)],holdout=4)
        p=3*(3*n+2)*(3*n+4);q=4*(2*n+1)*(2*n+3)
        self.equivalent(result,q(n+1)*p*A.gen()-p(n+1)*q)
        rebuilt=values[:2]
        for r in second:rebuilt.append(r*rebuilt[-1]**2/rebuilt[-2])
        self.assertEqual(rebuilt,values)

    def test_nested_sum_and_nonholonomic_product(self):
        ring=PolynomialRing(QQ,'n');n=ring.gen();A=OreAlgebra(ring,'Sn');E=A.gen()
        sums=[sum(factorial(k+1) for k in range(n)) for n in range(20)]
        self.assertEqual(sums[:5],[0,1,3,9,33])
        differences=[sums[i+1]-sums[i] for i in range(19)]
        self.equivalent(guess(differences,A,path=[(1,1)],holdout=3),E-(n+2))
        rebuilt=[0]
        for d in differences:rebuilt.append(rebuilt[-1]+d)
        self.assertEqual(rebuilt,sums)
        products=[prod((1+factorial(k) for k in range(n)),start=1) for n in range(20)]
        ratios=[QQ(products[i+1])/products[i] for i in range(19)]
        expected=n*E**2-(n*n+3*n+1)*E+(n+1)**2
        self.equivalent(guess(ratios,A,path=[(2,2)],holdout=3),expected)
        rebuilt=[1]
        for r in ratios:rebuilt.append(rebuilt[-1]*r)
        self.assertEqual(rebuilt,products)

    def test_native_option_validation_is_explicitly_not_api_parity(self):
        ring=PolynomialRing(QQ,'n');A=OreAlgebra(ring,'Sn')
        for options in ({'Somos':3},{'Somos':True},{'Somos':True,'maxShift':4,'homogeneous':1},
                        {'Somos':5,'maxShift':4,'homogeneous':1}):
            with self.subTest(options=options):
                with self.assertRaises(TypeError):guess([1,2,3,4,5],A,path=[(1,1)],**options)

    def test_corrupted_extra_coefficient_is_rejected(self):
        values=list(map(QQ,[0,1,2,3,4,5,6,1]))
        support=[(0,1,0,1,0),(0,1,0,2,0),(1,0,0,0,0),(1,0,0,1,0)]
        columns=_series_columns(values,support)
        result=self.features(columns,holdout=1,untouched=False,empty=True)
        self.assertEqual(result.status,'holdout_failed')
        # checkExtraValues=false is a private unchecked-fit option. Its
        # mathematical counterpart remains explicitly finite and unchecked.
        unchecked=self.features([c[:-1] for c in columns],[[1,-1,-1,-1]],holdout=0)
        self.assertIsNone(unchecked.metadata['exact_holdout'])

    def test_recop_evaluator_xfail_is_finite_feature_validation(self):
        values=[0,1,0,1,1,0,0,0,0,0,0,0,0,0,0]
        support=[(a,b,0,d,0) for a in range(3) for b in range(3-a) for d in range(2)]
        columns=_series_columns(list(map(QQ,values)),support)
        result=self.features(columns,holdout=5,untouched=False)
        self.assertFalse(result.metadata['all_indices_proved'])


class FriCASInventory(unittest.TestCase):
    def test_every_original_assertion_is_accounted_for(self):
        inventory=json.loads(Path(__file__).with_name('upstream_fricas_inventory.json').read_text())
        cases=inventory['cases']
        self.assertEqual(len(cases),73)
        self.assertEqual(sum(not c['upstream_xfail'] for c in cases),59)
        self.assertEqual(sum(c['upstream_xfail'] for c in cases),14)
        self.assertEqual(len({c['source_line'] for c in cases}),73)
        for case in cases:
            self.assertIn(case['status'],('covered_ore','covered_module_features','covered_ratio_translation','classified_native_search_policy','classified_native_options','classified_recop_evaluator'))
            self.assertTrue(callable(getattr(FriCASExamples,case['test'])))
            self.assertTrue(case['coverage_note'])
            self.assertTrue(case['source_expression'])


if __name__=='__main__':unittest.main()
