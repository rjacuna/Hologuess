"""Independent mathematical ports of all Guess handbook and QREGuess call sites.

Source inventories distinguish mathematical checks from upstream-only syntax or
output-selection policies. No RISC package or restricted implementation is used.
E is the forward shift; Q multiplies by q**n. Multivariate constrained supports
are passed through Hologuess's public observable-module interface.
"""
from functools import lru_cache
from itertools import product, permutations
import json
from pathlib import Path
import random
import unittest

from sage.all import (QQ, ZZ, GF, PolynomialRing, LaurentPolynomialRing, NumberField,
                      matrix, vector, factorial, binomial, fibonacci, nth_prime,
                      previous_prime)
from ore_algebra import OreAlgebra
from hologuess import (guess, prepare, ModuleProblem, solve_module,
                       interpolate_problem, guess_qsequence)

HERE = Path(__file__).resolve().parent
DATA = HERE/'upstream_data/risc'
EXPECTED = json.loads((DATA/'expected_relations.json').read_text())
QEXPECTED = json.loads((DATA/'expected_q_relations.json').read_text())
INVENTORY = json.loads((HERE/'upstream_risc_inventory.json').read_text())


def powers(dim, degree):
    return tuple(a for a in product(range(degree+1), repeat=dim) if sum(a)<=degree)


def combine_terms(operators):
    result = []
    for terms in operators:
        op = {}
        for term in terms:
            key=(tuple(term['shift']),tuple(term['powers']))
            op[key]=op.get(key,QQ.zero())+QQ(term['coefficient'])
        result.append({key:c for key,c in op.items() if c})
    return result


@lru_cache(None)
def harmonic(n):
    return sum((QQ(1)/j for j in range(1,n+1)),QQ.zero())


def jacobi(n,a,b):
    # Exact binomial definition of P_n^(a,b)(3), independent of its recurrences.
    return sum(binomial(n+a,j)*binomial(n+b,n-j)*2**j for j in range(n+1))


@lru_cache(None)
def signed_stirling(n,k):
    if n==0:return ZZ(k==0)
    if k<0 or k>n:return ZZ.zero()
    return signed_stirling(n-1,k-1)-(n-1)*signed_stirling(n-1,k)


@lru_cache(None)
def rational_gf(kind, last):
    """Exact coefficients of 1/D; target guessed relations never generate data."""
    values={}
    zero=QQ.zero()
    for i,j,k in product(range(last+1),repeat=3):
        if i==j==k==0: values[i,j,k]=QQ.one();continue
        def get(a,b,c):return values.get((a,b,c),zero)
        value=get(i-1,j,k)+get(i,j-1,k)+get(i,j,k-1)
        if kind=='xyz4': value-=4*get(i-1,j-1,k-1)
        else: value-=QQ(3)/4*(get(i-1,j-1,k)+get(i-1,j,k-1)+get(i,j-1,k-1))
        values[i,j,k]=value
    return values


def gf_value(kind, point):
    return rational_gf(kind, max(32,max(point)))[tuple(point)]


CONES=[((0,0,0),(0,1,1)),((0,1,0),(1,1,0)),
       ((1,0,0),(1,0,0)),((0,0,1),(1,0,2)),
       ((0,2,0),(0,2,0)),((1,1,0),(0,1,0)),((0,0,2),(0,0,4))]
GROEBNER={42:(1,0,0),43:(1,1,1),44:(1,2,3),45:(1,3,3),46:(1,4,3),
          47:(2,1,3),48:(2,2,4),49:(2,3,5),50:(2,4,5),51:(3,1,5),
          52:(3,2,5),53:(3,3,6),54:(3,4,6),55:(4,1,6),56:(4,2,6),
          57:(4,3,7),58:(4,4,7)}


def specification(case):
    """Finite ansatz and independently defined input for one source call."""
    unary=lambda r:tuple((j,) for j in range(r+1))
    if case in (3,7):
        shifts=((0,0),(1,0),(0,1)) if case==3 else ((0,0),(1,0))
        return 2,1,shifts,lambda n,k:factorial(n)*2**k,6,None
    if case in (4,5):return 1,2 if case==4 else 3,unary(2),lambda n:nth_prime(n+1),11,None
    if case==6:return 1,1,unary(1),lambda n:n+1,10,None
    if case in range(8,14):
        shifts={8:tuple((n,a,b) for n in range(3) for a,b in ((0,0),(1,0),(0,1))),
                9:((0,0,0),(1,0,0),(2,0,0)),10:((0,0,0),(0,1,0),(0,2,0)),
                11:((0,0,0),(0,0,1),(0,0,2)),12:tuple(product(range(2),repeat=3)),
                13:((0,0,0),(0,0,2),(0,1,1),(0,2,0))}[case]
        return 3,3 if case==9 else 0 if case==8 else 1,shifts,jacobi,6,None
    if case in (14,15):
        return 3,4,((0,0,0),(1,1,1),(2,2,2)),lambda *n:gf_value('xyz4',n),16,'symmetric' if case==15 else None
    if case in (17,18,19):return 1,0 if case in (18,19) else 1,unary(1),lambda n:QQ(1)/factorial(n),6,None
    if case in (20,34,35):return 2,1,tuple(product(range(2),repeat=2)),binomial,6 if case==20 else 11,'triangle' if case==20 else None
    if case in (21,22,23):return 1,2,unary(3),harmonic,16,None
    if case in (25,26,27):return 1,3,unary(2),lambda n:harmonic(2*n),101,None
    if case in (28,29):return 2,3,tuple(product(range(3),repeat=2)),lambda n,k:signed_stirling(n,k)*binomial(n,k),21,None
    if case in (30,31,32,33):return 1,0,unary(1),lambda n:2**n,11,None
    if case in (36,37,38):return 1,1,unary(2),(lambda n:harmonic(n+10)) if case==36 else harmonic,141 if case in(36,37) else 151,'start10' if case==37 else None
    if case==39:return 3,1,((0,0,0),(1,0,0),(0,1,0),(0,0,1)),lambda *n:gf_value('pairs34',n),11,None
    if case==40:return 2,1,((0,0),(1,0),(0,1)),lambda m,k:gf_value('pairs34',(0,m,k)),11,None
    if case==41:return 1,2,unary(2),lambda n:gf_value('pairs34',(n,n,n)),32,None
    if case in GROEBNER:
        bound,degree,count=GROEBNER[case]
        return 3,degree,tuple(product(range(bound+1),repeat=3)),lambda *n:gf_value('pairs34',n),51,('cones',count)
    raise AssertionError(case)


def finite_problem(case):
    dim,degree,shifts,value,size,option=specification(case)
    ps=powers(dim,degree)
    if option=='symmetric':
        orbits={tuple(sorted(a)):tuple(sorted(set(permutations(a)))) for a in ps}
        labels=[(s,orbit) for s in shifts for orbit in orbits.values()]
    else:labels=[(s,(a,)) for s in shifts for a in ps]
    if case==19:labels.append(((1,),((1,),)))
    if case in (22,23):
        labels=[(s,aa) for s,aa in labels if not(s==(3,) and (aa[0]==(1,) if case==22 else aa[0][0]>=1))]
    if isinstance(option,tuple) and option[0]=='cones':
        labels=[(s,aa) for s,aa in labels if not any(
            all(x>=y for x,y in zip(aa[0],a)) and all(x>=y for x,y in zip(s,b))
            for a,b in CONES[:option[1]])]
    maxshift=tuple(max(s[j] for s in shifts) for j in range(dim))
    starts=(10,) if option=='start10' else (0,)*dim
    # For the 3D GF stages, source already defaults to a bounded equation subset.
    # Use a deterministic, spread subset from the original available grid. This
    # avoids a huge redundant51^3 matrix while preserving every candidate column.
    cap=14 if case in(14,15) else 13 if case in GROEBNER else None
    ranges=[range(starts[j], starts[j]+(min(size-maxshift[j],cap) if cap else size-maxshift[j])) for j in range(dim)]
    points=list(product(*ranges))
    if option=='triangle':points=[n for n in points if n[1]<=n[0]]
    if cap:
        random.Random(706+case).shuffle(points)
        points=points[:max(80,2*len(labels)+12)]
    # Original univariate demonstration arrays are retained; extra values are
    # independently generated only for heldout checks. The diagonal case uses32
    # independently generated values rather than the source's101.
    if case==41:points=[(n,) for n in range(24)]
    held=[]
    for i in range(12):
        n=tuple(size+3+(i*(j+1))%7 for j in range(dim))
        if case in GROEBNER:n=tuple(19+(i*(j+1))%7 for j in range(dim))
        if case in(14,15):n=tuple(17+(i*(j+1))%7 for j in range(dim))
        if case==41:n=(24+i%6,)
        if option=='start10':n=(155+i,)
        if case in(20,34,35):n=(size+3+i,(i*2)%(size+4))
        if case in(28,29):n=(size+3+i,(3*i)%12)
        if n not in held:held.append(n)
    if case in (4,5):held=[]  # exact finite prime-prefix rank/underdetermination tests
    observations=points+held
    def monomial(n,a):
        result=ZZ.one()
        for v,e in zip(n,a):result*=v**e
        return result
    columns=[[sum((monomial(n,a) for a in orbit),ZZ.zero())*
              value(*(n[j]+shift[j] for j in range(dim))) for n in observations]
             for shift,orbit in labels]
    return ModuleProblem(columns,labels=labels,training_rows=len(points),name='RISC handbook call%d'%case),points,held


class UpstreamRiscTests(unittest.TestCase):
    maxDiff=1200
    def equal_operator(self,actual,expected):
        self.assertTrue(actual)
        self.assertEqual(expected.leading_coefficient()*actual,actual.leading_coefficient()*expected)

    def check_module(self,case):
        problem,points,held=finite_problem(case)
        result=solve_module(problem)
        self.assertTrue(result.metadata['full_training_kernel'])
        self.assertTrue(result.metadata['exact_training'])
        self.assertFalse(result.metadata['all_indices_proved'])
        if held:self.assertTrue(result.metadata['exact_holdout'],(case,result.metadata))
        cols=len(problem.columns)
        expected_ops=combine_terms(EXPECTED[str(case)]['operators']) if str(case) in EXPECTED else []
        if case in(29,30,31,32):
            expected_ops=combine_terms(EXPECTED['28' if case==29 else '33']['operators'])
        if case==5:
            # Source raises an insufficient-data diagnostic. HG deliberately
            # exposes the full finite kernel instead of claiming an annihilator.
            rows=list(zip(*problem.columns));exact=matrix(QQ,rows).right_kernel().basis_matrix()
            actual=matrix(QQ,result.basis)
            self.assertEqual(actual.row_space(),exact.row_space())
            self.assertGreater(len(result.basis),0)
            self.assertGreater(cols,len(points))
            return
        expected=[]
        for op in expected_ops:
            row=[]
            for shift,orbit in problem.labels:
                coefficients=[op.get((shift,a),QQ.zero()) for a in orbit]
                self.assertEqual(len(set(coefficients)),1,(case,shift,orbit,coefficients))
                row.append(coefficients[0])
            expected.append(row)
        exp=matrix(QQ,len(expected),cols,sum(expected,[]))
        actual=matrix(QQ,len(result.basis),cols,sum((list(v) for v in result.basis),[]))
        self.assertEqual(actual.row_space(),exp.row_space(),case)
        # Independent finite-field rank plus exact expected basis/residual is a
        # lower+upper rank certificate for the whole bounded observed space.
        rows=list(zip(*(col[:problem.training_rows] for col in problem.columns)))
        mod=matrix(GF(1000003),rows)
        self.assertEqual(mod.rank()+exp.rank(),cols,case)
        for row in expected:
            for values in zip(*problem.columns):self.assertEqual(sum(a*b for a,b in zip(row,values)),0)
        if case==29:
            target=[j for j,(s,aa) in enumerate(problem.labels) if s==(2,2) and aa[0] in((0,0),(1,0),(0,1))]
            self.assertTrue(any(any(v[j] for j in target) for v in result.basis))
        if case==30:self.assertEqual(len(result.basis),1)
        if case==31:
            self.assertEqual(list(zip(*problem.columns))[:3],[(1,2),(2,4),(4,8)])
        if case==32:self.assertEqual(problem.labels,(((0,),((0,),)),((1,),((0,),))))

    def run_case(self,case):
        if case==1:
            R=PolynomialRing(QQ,'n');A=OreAlgebra(R,'Sn')
            found=guess([fibonacci(n+1) for n in range(29)],A,path=[(2,0)],holdout=10)
            self.equal_operator(found.operator,A.gen()**2-A.gen()-1)
        elif case==2:
            R=PolynomialRing(QQ,'n');n=R.gen();A=OreAlgebra(R,'Sn');E=A.gen()
            found=guess([harmonic(j) for j in range(19)],A,path=[(2,1)],holdout=8)
            self.equal_operator(found.operator,(n+1)-(2*n+3)*E+(n+2)*E**2)
        elif case==16:
            R=PolynomialRing(QQ,'q');q=R.gen();K=R.fraction_field()
            values=[K.one()];acc=K.one()
            for n in range(1,27):
                acc*=(1+q**n)/(1+2*q**(n+1));values.append(acc)
            values=values[1:]
            cols=[[q**(j*k)*values[k+b] for k in range(len(values)-1)] for b,j in product(range(2),repeat=2)]
            found=interpolate_problem(ModuleProblem(cols,labels=((0,0),(0,1),(1,0),(1,1)),training_rows=19),q,band=(-8,16),max_points=128)
            coefficients={}
            for (col,e),c in found.coefficients.items():coefficients[col]=coefficients.get(col,R.zero())+c*q**e
            expected=[-1,-q**2,1,2*q**3]
            # Compare the full line over QQ(q), then independent rational residuals.
            pivot=next(i for i,c in enumerate(expected) if c)
            for i,c in enumerate(expected):self.assertEqual(coefficients.get(i,0)*expected[pivot],c*coefficients[pivot])
            for k in range(len(values)-1):self.assertEqual(sum(c*cols[i][k] for i,c in enumerate(expected)),0)
            self.assertTrue(found.metadata['exact_all_observations'])
            self.assertGreater(found.metadata['holdout_observations'],0)
        elif case==24:
            P=PolynomialRing(QQ,'z');K=NumberField(P.gen()**2-2,'a');a=K.gen()
            R=PolynomialRing(K,'n');n=R.gen();A=OreAlgebra(R,'Sn');E=A.gen()
            # Explicit Legendre coefficient formula, not the target recurrence.
            def leg(j):return sum((-1)**m*QQ(factorial(2*j-2*m))/(2**j*factorial(m)*factorial(j-m)*factorial(j-2*m))*a**(j-2*m) for m in range(j//2+1))
            found=guess([leg(j) for j in range(24)],A,path=[(2,1)],holdout=8)
            self.equal_operator(found.operator,(n+1)-a*(2*n+3)*E+(n+2)*E**2)
        elif case==59:
            R=PolynomialRing(QQ,'n');A=OreAlgebra(R,'Sn')
            data=[1,1,2,3,5,8,13,21,35]
            found=guess(data,A,path=[(2,0)])
            self.assertEqual(found.operators,())
            self.assertEqual(data[-1]-data[-2]-data[-3],1)
        elif case in(60,61):
            self.check_mixed_harmonics(case)
        elif case==62:
            R=PolynomialRing(QQ,'x');x=R.gen();A=OreAlgebra(R,'Dx');D=A.gen()
            found=guess([harmonic(n) for n in range(38)],A,path=[(2,2)],holdout=12)
            self.equal_operator(found.operator,1+3*(x-1)*D+(x-1)**2*D**2)
        elif case==63:
            R=PolynomialRing(QQ,'x');x=R.gen();A=OreAlgebra(R,'C')
            found=guess([binomial(QQ(1)/2,n) for n in range(20)],A,path=[(2,1)],holdout=9)
            self.equal_operator(found.operator,1+x-A.gen()**2)
        elif case==64:
            R=PolynomialRing(QQ,'x');x=R.gen();A=OreAlgebra(R,'C');C=A.gen()
            data=[QQ.one()]
            for n in range(116):data.append(data[-1]*16*(n+QQ(5)/6)*(n+QQ(1)/2)/((n+QQ(5)/3)*(n+2)))
            found=guess(data,A,path=[(8,7)],holdout=16)
            expected=sum((QQ(t['coefficient'])*x**t['x_degree']*C**t['power'] for t in json.loads((DATA/'expected_ae64.json').read_text())),A.zero())
            self.equal_operator(found.operator,expected)
        else:self.check_module(case)

    def check_mixed_harmonics(self,case):
        total=112 if case==60 else 61
        values=[harmonic(n)**2+harmonic(2*n)+QQ(1)/factorial(n) for n in range(total)]
        labels=[((b,),((j,),)) for b in range(5) for j in range(12)]
        cols=[[n**j*values[n+b] for n in range(total-4)] for (b,),((j,),) in labels]
        problem=ModuleProblem(cols,labels,training_rows=92 if case==60 else 57)
        result=solve_module(problem)
        if case==61:
            # An underdetermined prefix can have enormous artificial finite
            # kernel vectors: the default finite prime budget must report this
            # honestly, not silently declare the ansatz empty. Then exercise
            # the supported caller-supplied larger reconstruction budget.
            self.assertEqual(result.status,'reconstruction_exhausted')
            self.assertFalse(result.metadata['full_training_kernel'])
            self.assertEqual(result.metadata['rank_lower_bound'],57)
            primes=[];p=ZZ(2)**31
            for _ in range(320):
                p=previous_prime(p);primes.append(int(p))
            result=solve_module(problem,primes=primes,max_primes=320)
            self.assertTrue(result.metadata['full_training_kernel'])
            self.assertEqual(result.metadata['training_observations'],57)
            self.assertGreaterEqual(len(result.basis),3)
            exact=matrix(QQ,list(zip(*cols))).right_kernel().basis_matrix()
            actual=matrix(QQ,result.basis)
            self.assertEqual(actual.row_space(),exact.row_space())
            self.assertFalse(result.metadata['all_indices_proved'])
        else:
            self.assertTrue(result.metadata['exact_holdout'])
            op=combine_terms(EXPECTED['60']['operators'])[0]
            expected=vector(QQ,[op.get(((b,),(j,)),0) for (b,),((j,),) in labels])
            space=matrix(QQ,result.basis).row_space()
            self.assertEqual(space.dimension(),1)
            self.assertIn(expected,space)

    def check_qcase(self,index):
        L=LaurentPolynomialRing(ZZ,'q');q=L.gen()
        if index==0:
            values=[L.one()]
            for n in range(1,28):values.append(values[-1]*(1+q**n))
            fit=20;order=1;degree=1
        else:
            values=[];p=L.one()
            for n in range(54):
                if n:p*=(1-q**(3*n-2))*(1-q**(3*n-1))
                values.append(L({int(e)//3:c for e,c in p.dict().items() if int(e)%3==0}))
            fit=46;order=4;degree=6
        result=guess_qsequence(values[:fit],max_order=order,max_degree=degree,workers=1,max_points=256)
        self.assertIsNotNone(result)
        expected={}
        for b,j,e,c in QEXPECTED[index]['coefficients']:
            key=(b,j,e);expected[key]=expected.get(key,QQ.zero())+QQ(c)
        expected={key:c for key,c in expected.items() if c}
        # Compare grouped E coefficients up to a common QQ(q,Q) unit, including
        # polynomial content; no numerical output comparison is used.
        R=PolynomialRing(QQ,('q','Q'));qq,Q=R.gens()
        def grouped(op):
            out={}
            for (b,j,e),c in op.items():out[b]=out.get(b,R.zero())+c*qq**e*Q**j
            return out
        actual=grouped(result.coefficients);ex=grouped(expected)
        pivot=max(ex)
        self.assertEqual(set(actual),set(ex))
        for b in ex:self.assertEqual(actual[b]*ex[pivot],ex[b]*actual[pivot])
        self.assertTrue(result.metadata['finite_window_valid'])
        grouped_laurent={}
        for (b,j,e),c in result.coefficients.items():grouped_laurent[b,j]=grouped_laurent.get((b,j),L.zero())+c*q**e
        for n in range(fit-max(actual),len(values)-max(actual)):
            self.assertEqual(sum((coefficient*q**(j*n)*values[n+b] for (b,j),coefficient in grouped_laurent.items()),L.zero()),0)

    def test_inventory_is_exhaustive_and_explicit(self):
        cases=INVENTORY['cases']
        self.assertEqual(len(cases),66)
        self.assertEqual({c['id'] for c in cases},{'guess-%02d'%i for i in range(1,65)}|{'qguess-01','qguess-02'})
        for case in cases:
            self.assertTrue(case['source_line'])
            self.assertTrue(case['source_input_cell'])
            self.assertTrue(case['mathematical_check'])
            self.assertTrue(case['test'])
            self.assertIn(case['coverage'],('mathematical_equivalent','intentional_policy_difference'))
            self.assertTrue(hasattr(self,case['test']))


def _source_case(case):
    def run(self):self.run_case(case)
    run.__name__='test_guess_%02d'%case
    run.__doc__='Mathematical port of RISC Guess handbook call %d.'%case
    return run

for _case in range(1,65):setattr(UpstreamRiscTests,'test_guess_%02d'%_case,_source_case(_case))
for _index in range(2):
    def _qrun(self,index=_index):self.check_qcase(index)
    setattr(UpstreamRiscTests,'test_qguess_%02d'%(_index+1),_qrun)

if __name__=='__main__':unittest.main()
