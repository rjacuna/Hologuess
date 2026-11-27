"""Independent mathematical ports of the pinned gfun guessing regressions.

No Maple implementation is copied or imported.  The companion inventory accounts
for every upstream Test call, including API/scheduling assertions which cannot be
passed by a different program.  Named transcendental constants are represented
by exact formal parameters; this does not test Maple's expression-protection API.
"""
from fractions import Fraction
import json
from math import factorial, prod
from pathlib import Path
import unittest

try:
    from sage.all import (QQ, GF, PolynomialRing, PowerSeriesRing, CyclotomicField,
                          QuadraticField, binomial, matrix, vector)
    from ore_algebra import OreAlgebra
    from hologuess import guess, ModuleProblem, solve_module
    HAVE_SAGE = True
except ImportError:
    HAVE_SAGE = False

INVENTORY = Path(__file__).with_name('upstream_gfun_inventory.json')


def fibonacci(count):
    values = [1, 1]
    while len(values) < count:
        values.append(values[-1] + values[-2])
    return values[:count]


def rational_recurrence(coefficients, seeds, count):
    """Exact fixture generation, with explicitly supplied singular initial data."""
    order = len(coefficients) - 1
    values = dict(seeds)
    for n in range(count - order):
        row = [p(n) for p in coefficients]
        residual = sum(row[j]*values[n+j] for j in range(order))
        if n+order in values:
            if residual+row[-1]*values[n+order]:
                raise AssertionError('Inconsistent independent recurrence fixture')
        elif row[-1]:
            values[n+order] = -residual/row[-1]
        else:
            raise AssertionError('A singular index needs explicit seed data')
    return [values[n] for n in range(count)]


def ore_case(name):
    """Return exact data and independently expressed expected mathematical relation."""
    K = QQ
    action = 'shift'
    holdout = 4
    parameter_degree = 0
    N = 32
    if (name.startswith(('symbolic_', 'geometric_', 'protected_', 'parameter_'))
            and name not in ('geometric_algebraic','geometric_twice_algebraic','parameter_binomial_function')):
        if name in ('geometric_sqrt2', 'symbolic_sqrt2'):
            K = QuadraticField(2, 'a'); a = K.gen()
        elif name in ('geometric_two_roots', 'symbolic_algebraic_product',
                      'symbolic_algebraic_sum', 'geometric_algebraic_parameter'):
            C = CyclotomicField(24); zeta = C.gen()
            a = zeta**3 + zeta**-3
            if name == 'geometric_two_roots':
                K = C; a += zeta**2 + zeta**-2
            else:
                K = PolynomialRing(C, 'tau').fraction_field(); tau = K.gen()
        else:
            K = PolynomialRing(QQ, 'tau').fraction_field(); tau = K.gen()
        if name.startswith('symbolic_'):
            if name == 'symbolic_sqrt2': value = a
            elif name == 'symbolic_algebraic_product': value = a*tau
            elif name == 'symbolic_algebraic_sum': value = (a-1)*tau
            else: value = tau
            data = [value]*N
            build = lambda n,E: E-1
        elif name == 'protected_initial_exception':
            data = [tau]+[K.one()]*(N-1)
            build = lambda n,E: n*(E-1)
        elif name == 'protected_scaled_geometric':
            data = [tau*2**n for n in range(N)]
            build = lambda n,E: E-2
        elif name == 'geometric_algebraic_parameter':
            data = [(a*tau)**n for n in range(N)]; parameter_degree = 1
            build = lambda n,E: E-a*tau
        elif name in ('geometric_sqrt2', 'geometric_two_roots'):
            data = [a**n for n in range(N)]
            build = lambda n,E: E-a
        elif name == 'geometric_parameter_reciprocal':
            data = [(1/(tau+1))**n for n in range(N)]; parameter_degree = 1
            build = lambda n,E: (tau+1)*E-1
        elif name == 'parameter_quadratic':
            data = [(tau**2+1)**n for n in range(N)]; parameter_degree = 2
            build = lambda n,E: E-tau**2-1
        elif name == 'parameter_linear':
            data = [(tau+1)**n for n in range(N)]; parameter_degree = 1
            build = lambda n,E: E-tau-1
        elif name == 'parameter_vanishing_image':
            data = [(tau-12)*2**n for n in range(N)]
            build = lambda n,E: E-2
        elif name == 'parameter_pole':
            data = [K(2)**n/(tau-1010) for n in range(N)]
            build = lambda n,E: E-2
        else:
            data = [tau**n for n in range(N)]; parameter_degree = 1
            build = lambda n,E: E-tau
        order, degree = 1, int(name == 'protected_initial_exception')
    elif name == 'constant_sequence':
        data=[1]*N;order,degree=1,0;build=lambda n,E:E-1
    elif name == 'finite_field_parameter':
        K = PolynomialRing(GF(1009), 'tau').fraction_field(); tau = K.gen()
        data = [tau**n for n in range(18)]; parameter_degree = 1
        order, degree = 1,0; build = lambda n,E: E-tau
    elif name == 'fib':
        data = fibonacci(N); order,degree = 2,0; build = lambda n,E:E**2-E-1
    elif name in ('catalan', 'catalan_modular'):
        if name.endswith('modular'): K = GF(1009)
        data = [K(QQ(binomial(2*n,n))/(n+1)) for n in range(N)]
        order,degree = 1,1; build = lambda n,E:(n+2)*E-4*n-2
    elif name == 'shifted_catalan':
        data=[QQ(binomial(2*(n+2),n+2))/(n+3) for n in range(N)]
        order,degree=1,1;build=lambda n,E:(n+4)*E-4*n-10
    elif name == 'central_binomial_squared':
        data = [binomial(2*n,n)**2 for n in range(N)]
        order,degree = 1,2; build = lambda n,E:(n+1)**2*E-4*(2*n+1)**2
    elif name == 'zero_sequence':
        data = [0]*N; order,degree = 0,0; build=lambda n,E:E.parent().one()
    elif name in ('finite_support_zero', 'finite_support_one'):
        k = int(name.endswith('one')); data = [int(n==k) for n in range(N)]
        order,degree = 0,1; build=lambda n,E:E.parent()(n-k)
    elif name == 'quadratic_sequence':
        data = [1+n*(n+1)//2 for n in range(N)]
        order,degree=1,2; build=lambda n,E:(n*n+n+2)*E-(n*n+3*n+4)
    elif name == 'piecewise_linear':
        data = [abs(n-8)+1 for n in range(N)]
        order,degree=2,1; build=lambda n,E:(n-7)*(E**2-2*E+1)
    elif name == 'cosh_coefficients':
        data = [QQ(1)/factorial(n) if n%2==0 else QQ.zero() for n in range(N)]
        order,degree=2,2; build=lambda n,E:(n+1)*(n+2)*E**2-1
    elif name == 'singular_first_order':
        data=[0]+[QQ(2*n)*binomial(2*n+1,n)/(n+2) for n in range(1,N)]
        order,degree=1,2; build=lambda n,E:n*(n+3)*E-2*(2*n+3)*(n+1)
    elif name == 'parity_rational':
        data=[QQ.zero()]*N; data[2]=QQ.one()
        for n in range(2,N-2,2):data[n+2]=data[n]*(n*n+2*n+2)/((n+1)*(n+2))
        order,degree=2,3; build=lambda n,E:n*((n+1)*(n+2)*E**2-(n*n+2*n+2))
    elif name == 'rational_crt':
        a=QQ(100039)/100007; data=[a**n for n in range(N)]
        order,degree=1,0; build=lambda n,E:E-a
    elif name == 'singular_second_order':
        data=rational_recurrence([lambda n:QQ(2),lambda n:QQ(-5-2*n),lambda n:QQ(4+2*n)],
                                 {0:QQ.one(),1:QQ(3)/2},N)
        order,degree=2,1; build=lambda n,E:2-(5+2*n)*E+(4+2*n)*E**2
    elif name in ('leading_zeros_recurrence','leading_zeros_differential',
                  'third_order_historical','ternary_hypergeometric','ternary_differential'):
        if name.startswith('leading_zeros'):
            coefficients=[lambda n:-8*n*n-40*n-48,
                lambda n:-15*n**3-174*n*n+59*n+1338,
                lambda n:-6*n**4-117*n**3-485*n*n+1104*n+6588,
                lambda n:n**5+26*n**4-95*n**3-4440*n*n+5724*n+147744]
            seeds={n:QQ.zero() for n in range(13)};seeds[12]=QQ(1)/factorial(12)
            data=rational_recurrence(coefficients,seeds,80)
            order,degree=3,5
            build=lambda n,E:sum(c(n)*E**j for j,c in enumerate(coefficients))
            if name.endswith('differential'):
                action='differential';order,degree=5,5
                build=lambda x,D:(95040+3280*x+1120*x*x-48*x**3)+(24640*x+1830*x*x+218*x**3-48*x**4)*D+(-3310*x*x-176*x**3-174*x**4-8*x**5)*D**2+(-226*x**3-105*x**4-15*x**5)*D**3+(21*x**4-6*x**5)*D**4+x**5*D**5
        elif name=='third_order_historical':
            coefficients=[lambda n:1728+4896*n+2880*n*n,lambda n:144+48*n+240*n*n,
                          lambda n:-60-230*n-100*n*n,lambda n:3+16*n+5*n*n]
            data=rational_recurrence(coefficients,{0:QQ.zero(),1:QQ.zero(),2:QQ.one()},40)
            order,degree=3,2;build=lambda n,E:sum(c(n)*E**j for j,c in enumerate(coefficients))
        else:
            data=rational_recurrence([lambda n:QQ(-27*n*n-81*n-60),lambda n:QQ(4*n*n+22*n+30)],{0:QQ.one()},40)
            order,degree=1,2;build=lambda n,E:(4*n*n+22*n+30)*E-(27*n*n+81*n+60)
            if name.endswith('differential'):
                action='differential';order,degree=3,3
                build=lambda x,D:60+(-30+276*x)*D+(189*x*x-26*x)*D**2+(27*x**3-4*x*x)*D**3
    elif name == 'hypergeometric_three_parameters':
        K=PolynomialRing(QQ,('a','b','c')).fraction_field();a,b,c=K.gens()
        data=[K.one()]
        for n in range(11):data.append(data[-1]*(n+a)*(n+b)/((n+1)*(n+c)))
        parameter_degree=2;holdout=2
        order,degree=1,2;build=lambda n,E:(n+1)*(n+c)*E-(n+a)*(n+b)
    elif name == 'degree_eight_rational_ratio':
        def odd_product(m):return prod(range(1,m+1,2))
        data=[QQ(2)**(8*k+6)*(4*k+1)*(4*k+3)*odd_product(4*k-1)**2*
              odd_product(4*k+1)**2/((2*k+1)**4*odd_product(2*k-1)**8) for k in range(32)]
        order,degree=1,8
        build=lambda n,E:(2*n+3)**4*(2*n+1)**4*E-256*(4*n+7)*(4*n+1)*(4*n+5)**3*(4*n+3)**3
    elif name in ('algebraic_initial_conditions','two_algebraic_initial_conditions'):
        C=CyclotomicField(8);I=C.gen()**2;a=C.gen()+C.gen()**-1;K=C
        base=rational_recurrence([lambda n:QQ(4*n*n+8*n+3),lambda n:QQ(-12*n*n-40*n-33),
                                 lambda n:QQ(8*n*n+36*n+40)],{0:QQ.one(),1:QQ(5)/12},N)
        if name=='algebraic_initial_conditions':
            data=[(1-I)*(-I)**n*v for n,v in enumerate(base)]
            build=lambda n,E:-(4*n*n+8*n+3)+I*(12*n*n+40*n+33)*E+(8*n*n+36*n+40)*E**2
        else:
            data=[I*a*v for v in base]
            build=lambda n,E:4*n*n+8*n+3-(12*n*n+40*n+33)*E+(8*n*n+36*n+40)*E**2
        order,degree=2,2
    elif name in ('complex_factorial','complex_hypergeometric'):
        K=CyclotomicField(4);I=K.gen()
        if name=='complex_factorial':
            data=[-I**n*factorial(n)/2 for n in range(N)]
            order,degree=1,1;build=lambda n,E:E-I*(n+1)
        else:
            data=[(-I)**n*QQ(factorial(2*n))**2/(32**n*factorial(n)**3) for n in range(N)]
            order,degree=1,2;build=lambda n,E:8*(n+1)*E+I*(2*n+1)**2
    elif name=='scaled_parameter_hypergeometric':
        K=PolynomialRing(QQ,'tau').fraction_field();t=K.gen()
        data=[t**2]
        for n in range(N-1):data.append(data[-1]*(-t)*(n+2)**2/(2*n+3)**2)
        parameter_degree=1;order,degree=1,2
        build=lambda n,E:(2*n+3)**2*E+t*(n+2)**2
    elif name=='triple_exponential_coefficients':
        K=PolynomialRing(QQ,'tau').fraction_field();t=K.gen()
        data=[(t**(2*n)+t**n+1)/factorial(n) for n in range(30)]
        parameter_degree=3;order,degree=3,3
        build=lambda n,E:(n+1)*(n+2)*(n+3)*E**3-(1+t+t*t)*(n+1)*(n+2)*E**2+(t+t*t+t**3)*(n+1)*E-t**3
    elif name in ('constant_function','monomial_function','exp_function','exp_twice_function',
                  'catalan_differential','binomial_differential','exp_sqrt_function',
                  'parametric_rational_function','parameter_binomial_function',
                  'parameterized_residual_regression','chi_at_one'):
        action='differential'
        if name=='constant_function':data=[1]+[0]*(N-1);order,degree=1,0;build=lambda x,D:D
        elif name=='monomial_function':data=[0,1]+[0]*(N-2);order,degree=1,1;build=lambda x,D:x*D-1
        elif name in ('exp_function','exp_twice_function'):
            a=2 if name=='exp_twice_function' else 1
            data=[QQ(a)**n/factorial(n) for n in range(N)];order,degree=1,0;build=lambda x,D:D-a
        elif name=='catalan_differential':
            data=[QQ(binomial(2*n,n))/(n+1) for n in range(N)]
            order,degree=2,2;build=lambda x,D:x*(1-4*x)*D**2+(2-10*x)*D-2
        elif name=='binomial_differential':
            data=[binomial(2*n,n)**2 for n in range(N)]
            order,degree=2,2;build=lambda x,D:x*(1-16*x)*D**2+(1-32*x)*D-4
        elif name=='exp_sqrt_function':
            data=[sum(QQ(binomial(2*k,k))/(4**k*factorial(n-k)) for k in range(n+1)) for n in range(N)]
            order,degree=1,1;build=lambda x,D:2*(1-x)*D-(3-2*x)
        elif name=='parametric_rational_function':
            K=PolynomialRing(QQ,('p','q')).fraction_field();p,q=K.gens()
            data=[K.one(),2*p]
            for n in range(2,20):data.append(2*p*data[-1]+q*p*p*data[-2])
            parameter_degree=3;order,degree=1,2
            build=lambda x,D:(1-2*p*x-q*p*p*x*x)*D-(2*p+2*q*p*p*x)
        elif name=='parameter_binomial_function':
            K=PolynomialRing(QQ,'tau').fraction_field();t=K.gen()
            data=[K.one()]
            for n in range(19):data.append(data[-1]*(t+n)/(n+1))
            parameter_degree=1;order,degree=1,1;build=lambda x,D:(1-x)*D-t
        elif name=='parameterized_residual_regression':
            K=PolynomialRing(QQ,'tau').fraction_field();t=K.gen()
            wire=json.loads((INVENTORY.parent/'upstream_data/gfun/mathematical_inputs.json').read_text())
            data=[QQ(a)+QQ(b)*t for a,b in wire['parameterized_residual_coefficients']]
            holdout=2;order,degree=3,3
            build=lambda x,D:(3*x**3+2*x*x-x)*D**3+(21*x*x+11*x-4)*D**2+(30*x+10)*D+6
        else:
            K=PolynomialRing(QQ,('epsilon','tau')).fraction_field();e,tau=K.gens()
            T=PowerSeriesRing(K,'s',default_prec=N+2);s=T.gen()
            derivative=((e+1/e)/2*s.cosh()+(e-1/e)/2*s.sinh())/(1+s)
            series=tau+derivative.integral();data=[series[n] for n in range(N)]
            order,degree=3,1;build=lambda x,D:(1+x)*D**3+2*D**2-(1+x)*D
    elif name in ('catalan_algebraic','cube_root_algebraic','fibonacci_algebraic',
                  'geometric_algebraic','geometric_twice_algebraic','zero_function_algebraic',
                  'piecewise_algebraic','quadratic_algebraic','sqrt_algebraic','krew80','krew90'):
        action='algebraic'
        if name=='catalan_algebraic':
            data=[QQ(binomial(2*n,n))/(n+1) for n in range(N)]
            order,degree=2,1;build=lambda x,C:x*C**2-C+1
        elif name=='cube_root_algebraic':
            data=[QQ(binomial(QQ(1)/3,n))*(-1)**n for n in range(N)]
            order,degree=3,1;build=lambda x,C:C**3+x-1
        elif name=='fibonacci_algebraic':
            data=fibonacci(N);order,degree=1,2;build=lambda x,C:(1-x-x*x)*C-1
        elif name=='geometric_algebraic':
            data=[1]*N;order,degree=1,1;build=lambda x,C:(1-x)*C-1
        elif name=='geometric_twice_algebraic':
            data=[2**n for n in range(N)];order,degree=1,1;build=lambda x,C:(1-2*x)*C-1
        elif name=='zero_function_algebraic':
            data=[0]*N;order,degree=1,0;build=lambda x,C:C
        elif name=='piecewise_algebraic':
            data=[abs(n-8)+1 for n in range(40)]
            order,degree=1,9;build=lambda x,C:(1-x)**2*C-(9-10*x+2*x**9)
        elif name=='quadratic_algebraic':
            data=[1+n*(n+1)//2 for n in range(N)]
            order,degree=1,3;build=lambda x,C:(1-x)**3*C-(1-x+x*x)
        elif name=='sqrt_algebraic':
            data=[QQ(1)-QQ(1)]+[-binomial(QQ(1)/2,n)*(-4)**n for n in range(1,N)]
            order,degree=2,1;build=lambda x,C:C*C-2*C+4*x
        else:
            N=80 if name=='krew80' else 90
            data=[QQ.zero()]*N
            for k in range((N-1)//3+1):data[3*k]=QQ(4)**k*binomial(3*k,k)/((k+1)*(2*k+1))
            order,degree=3,6;build=lambda x,C:-1+54*x**3+(1-72*x**3)*C+16*x**3*C**2+64*x**6*C**3
    else:
        raise KeyError(name)
    variable='n' if action=='shift' else 'x'
    R=PolynomialRing(K,variable);n=R.gen()
    A=OreAlgebra(R,{'shift':'Sn','differential':'Dx','algebraic':'C'}[action])
    return data,A,build(n,A.gen()),order,degree,parameter_degree,holdout


def module_case(name):
    """Exact feature observations for approximation and nonlinear identities."""
    T=PowerSeriesRing(QQ,'x',default_prec=40);x=T.gen()
    if name in ('hermite_sin_cos_exp','hermite_sin_cos_at_pi'):
        if name.endswith('at_pi'):
            series=[-x.sin(),-x.cos()];degrees=[3,3]
        else:
            series=[x.sin(),x.cos(),x.exp()];degrees=[3,2,5]
        rows=sum(degrees)+len(degrees)-1
        columns=[[(x**a*f)[n] for n in range(rows)] for f,d in zip(series,degrees) for a in range(d+1)]
        return ModuleProblem(columns,training_rows=rows),None
    if name=='zero_approximant_space':
        return ModuleProblem([[0]*8,[0]*8],training_rows=5),[(1,0),(0,1)]
    if name=='exp_approximant_basis':
        f=x.exp();degrees=[3,3];rows=5
        columns=[[(x**a*g)[n] for n in range(rows)] for g,d in zip([T.one(),f],degrees) for a in range(d+1)]
        return ModuleProblem(columns),None
    if name=='catalan_reversion':
        f=T([QQ(binomial(2*n,n))/(n+1) for n in range(36)])*x
        columns=[[g[n] for n in range(32)] for g in (f,f*f,x)]
        return ModuleProblem(columns,training_rows=25),[(1,-1,-1)]
    if name=='logarithmic_egf':
        f=(x*x.exp()).exp()
        assert [factorial(n)*f[n] for n in range(8)]==[1,1,3,10,41,196,1057,6322]
        features=[f*f.derivative(2),f.derivative()**2,f*f.derivative()]
        columns=[[(x**a*g)[n] for n in range(30)] for g in features for a in range(2)]
        return ModuleProblem(columns,training_rows=22),[(1,1,-1,-1,-2,-1)]
    if name in ('total_degree_algebraic','total_degree_differential'):
        algebraic=name.endswith('algebraic');f=1/(1-x)
        features=[];labels=[]
        for b in range(2):
            for a in range(3-b):
                g=(f**b if algebraic else (f.derivative(b)))
                features.append(x**a*g);labels.append((a,b))
        columns=[[g[n] for n in range(24)] for g in features]
        if algebraic: expected=[-1,0,0,1,-1]
        else:expected=[-1,0,0,1,-1]
        return ModuleProblem(columns,training_rows=18),[tuple(expected)]
    if name=='pi_series_approximant':
        K=PolynomialRing(QQ,'tau').fraction_field();t=K.gen()
        data1=[K.zero()]*21; data2=[K.zero()]*20
        for j,c in {1:1,5:-t*t/40,9:t**4/3456,13:-t**6/599040,17:t**8/175472640}.items():data1[j]=c
        for j,c in {0:1,4:-t*t/8,8:t**4/384,12:-t**6/46080,16:t**8/10321920}.items():data2[j]=c
        columns=[[K.zero() if n<a else v[n-a] for n in range(7)] for v in (data1,data2) for a in range(4)]
        return ModuleProblem(columns),[(0,0,-1,0,0,0,0,1)]
    if name=='constant_series_infinite_order':
        rational=lambda s:QQ(Fraction(s))
        v1=list(map(rational,['1','-1','1/4','-1/24','-1/192','-11/1920','-103/23040','-1177/322560','-15713/5160960','-240403/92897280','-4149703/1857945600']))
        v2=list(map(rational,['1','-1/2','0','-1/48','-5/384','-3/320','-329/46080','-731/129024','-1329/286720','-20609/5308416','-12310199/3715891200']))
        columns=[[QQ.zero() if n<a else v[n-a] for n in range(8)] for v in (v1,v2,[QQ.one()]+[QQ.zero()]*10) for a in range(3)]
        expected=(2973864,123960,2195,-3377328,1145784,-49510,403464,15456,701)
        return ModuleProblem(columns),[expected]
    raise KeyError(name)


ORE_CASES = (
    'fib','constant_sequence','catalan','catalan_modular','shifted_catalan','central_binomial_squared','zero_sequence',
    'finite_support_zero','finite_support_one','quadratic_sequence','piecewise_linear',
    'cosh_coefficients','singular_first_order','parity_rational','rational_crt',
    'singular_second_order','hypergeometric_three_parameters','degree_eight_rational_ratio',
    'leading_zeros_recurrence','leading_zeros_differential','third_order_historical',
    'ternary_hypergeometric','ternary_differential',
    'algebraic_initial_conditions','two_algebraic_initial_conditions','complex_factorial',
    'complex_hypergeometric','scaled_parameter_hypergeometric','triple_exponential_coefficients',
    'constant_function','monomial_function','exp_function','exp_twice_function',
    'catalan_differential','binomial_differential','exp_sqrt_function','parametric_rational_function',
    'chi_at_one','parameter_binomial_function','parameterized_residual_regression',
    'catalan_algebraic','cube_root_algebraic','fibonacci_algebraic',
    'geometric_algebraic','geometric_twice_algebraic','zero_function_algebraic',
    'piecewise_algebraic','quadratic_algebraic','sqrt_algebraic','krew80','krew90',
    'symbolic_name','symbolic_pi','symbolic_gamma','symbolic_catalan','symbolic_exp2',
    'symbolic_sqrt2','symbolic_algebraic_product','symbolic_algebraic_sum',
    'geometric_sqrt2','geometric_two_roots','geometric_algebraic_parameter',
    'protected_initial_exception','protected_scaled_geometric','geometric_parameter',
    'geometric_log_parameter','geometric_parameter_reciprocal','parameter_quadratic',
    'parameter_linear','parameter_vanishing_image','parameter_pole','finite_field_parameter',
)
MODULE_CASES = ('hermite_sin_cos_exp','hermite_sin_cos_at_pi','zero_approximant_space',
                'exp_approximant_basis','catalan_reversion','logarithmic_egf',
                'total_degree_algebraic','total_degree_differential','pi_series_approximant',
                'constant_series_infinite_order')


@unittest.skipUnless(HAVE_SAGE,'SageMath and ore_algebra are required')
class GfunMathematicsTests(unittest.TestCase):
    def check_ore(self,name):
        data,A,expected,r,d,pd,holdout=ore_case(name)
        result=guess(data,A,path=[(r,d)],parameter_degree=pd,holdout=holdout,backend='module')
        self.assertTrue(result.operators,name)
        for actual in result.operators:
            self.assertEqual(expected.leading_coefficient()*actual,actual.leading_coefficient()*expected,name)
        self.assertTrue(result.metadata['exact_training'])
        self.assertTrue(result.metadata['exact_holdout'])
        self.assertFalse(result.metadata['holdout_used_to_select'],name)
        self.assertGreater(result.metadata['holdout_observations'],0)
        # Independently evaluate the supplied expected operator using its public
        # coefficient actions, including every originally unseen observation.
        from hologuess import prepare
        problem=prepare(data,A,order=r,degree=d,parameter_degree=pd)
        # Every recovered vector is checked against the full exact observation
        # matrix; comparison above also fixes its complete operator up to a unit.
        for coefficients in result.module_result.valid_basis:
            for row in zip(*problem.columns):
                self.assertEqual(sum(a*c for a,c in zip(row,coefficients)),0,name)

    def check_module(self,name):
        problem,expected=module_case(name)
        result=solve_module(problem)
        self.assertTrue(result.valid_basis,name)
        # Independent exact elimination checks the *whole* finite relation space,
        # including approximants for which no all-index relation is asserted.
        K=QQ
        if name=='pi_series_approximant':K=problem.columns[0][0].parent()
        exact=matrix(K,list(zip(*problem.columns))).right_kernel().basis_matrix()
        actual=matrix(K,result.valid_basis).echelon_form()
        self.assertEqual(actual,exact.echelon_form(),name)
        if expected is not None:
            for row in expected:self.assertIn(vector(K,row),exact.row_space(),name)
        for coefficients in result.valid_basis:
            for row in zip(*problem.columns):self.assertEqual(sum(a*c for a,c in zip(row,coefficients)),0)
        if problem.training_rows<len(problem.columns[0]):
            self.assertTrue(result.metadata['exact_holdout'])
            self.assertFalse(result.metadata['holdout_used_to_select'])

    def test_empty_and_one_term_data_are_rejected(self):
        A=OreAlgebra(PolynomialRing(QQ,'x'),'Dx')
        for values in ([],[1]):
            with self.subTest(values=values),self.assertRaises(ValueError):
                guess(values,A,order=1,degree=0)

    def test_parameter_degree_overflow_is_a_bounded_empty_space(self):
        K=PolynomialRing(QQ,'tau').fraction_field();t=K.gen()
        A=OreAlgebra(PolynomialRing(K,'n'),'Sn')
        result=guess([t**(2*n) for n in range(18)],A,path=[(1,0)],parameter_degree=1,holdout=3)
        self.assertEqual(result.operators,())
        self.assertEqual(result.metadata['valid_nullity'],0)

    def test_total_degree_one_has_no_geometric_equation(self):
        values=[QQ.one()]*24
        # Algebraic and differential equations both need x times the generator.
        for generator in ('C','Dx'):
            A=OreAlgebra(PolynomialRing(QQ,'x'),generator)
            from hologuess import prepare
            full=prepare(values,A,order=1,degree=1,holdout=4)
            keep=[j for j,label in enumerate(full.labels)
                  if label.polynomial().degree()+label.leading_coefficient().degree()<=1]
            result=solve_module(ModuleProblem([full.columns[j] for j in keep],training_rows=full.training_rows))
            self.assertEqual(result.valid_basis,())

    def test_chi_has_no_first_order_degree_three_relation(self):
        data,A,expected,r,d,pd,holdout=ore_case('chi_at_one')
        result=guess(data,A,path=[(1,3)],parameter_degree=0,holdout=4,backend='module')
        self.assertEqual(result.operators,())

    def test_short_rational_input_empty_in_explicit_small_ansatz(self):
        A=OreAlgebra(PolynomialRing(QQ,'n'),'Sn')
        result=guess([0,1,4,5,8,9],A,path=[(1,0)],holdout=1)
        self.assertEqual(result.operators,())
        self.assertEqual(result.metadata['rank'],2)

    def test_short_algebraic_input_empty_in_explicit_small_ansatz(self):
        wire=json.loads((INVENTORY.parent/'upstream_data/gfun/mathematical_inputs.json').read_text())
        data=[QQ(Fraction(s)) for s in wire['short_algebraic_negative_coefficients']]
        A=OreAlgebra(PolynomialRing(QQ,'x'),'C')
        result=guess(data,A,path=[(2,2)],holdout=3)
        self.assertEqual(result.operators,())
        self.assertEqual(result.metadata['rank'],9)

    def test_fractional_holdout_is_rejected(self):
        A=OreAlgebra(PolynomialRing(QQ,'n'),'Sn')
        with self.assertRaises(ValueError):
            guess(fibonacci(16),A,path=[(2,0)],holdout=QQ(1)/5)

    def test_high_valuation_fixture_matches_all_original_48_coefficients(self):
        wire=json.loads((INVENTORY.parent/'upstream_data/gfun/mathematical_inputs.json').read_text())
        values,*_=ore_case('leading_zeros_differential')
        self.assertEqual(values[:48],[QQ(Fraction(s)) for s in wire['high_valuation_48']])

    def test_explicit_source_prefixes_match_fixture_generators(self):
        integer_prefixes={
            'singular_first_order':[0,2,10,42,168,660,2574,10010],
            'central_binomial_squared':[1,4,36,400,4900,63504,853776],
            'third_order_historical':[0,0,1,20,307,4280,56914,736568],
            'ternary_hypergeometric':[1,2,6,22,91,408,1938,9614],
        }
        for case,prefix in integer_prefixes.items():
            with self.subTest(case=case):self.assertEqual(ore_case(case)[0][:len(prefix)],prefix)
        data,A,*_=ore_case('scaled_parameter_hypergeometric');t=A.base_ring().base_ring().gen()
        self.assertEqual(data[:4],[t*t,-QQ(4)/9*t**3,QQ(4)/25*t**4,-QQ(64)/1225*t**5])
        values,A,*_=ore_case('complex_hypergeometric');I=A.base_ring().base_ring().gen()
        self.assertEqual(values[:3],[1,-I/8,-QQ(9)/128])

    def test_function_fixtures_retain_their_action_and_input_conventions(self):
        for case in ('geometric_algebraic','geometric_twice_algebraic'):
            values,A,*_=ore_case(case)
            self.assertTrue(A.is_C())
            ratio=2 if case=='geometric_twice_algebraic' else 1
            self.assertEqual(values[:8],[ratio**n for n in range(8)])
        values,A,*_=ore_case('parameter_binomial_function')
        self.assertIsNot(A.is_D(),False)
        t=A.base_ring().base_ring().gen()
        self.assertEqual(values[:3],[1,t,t*(t+1)/2])

    def test_inventory_accounts_for_every_source_call(self):
        inventory=json.loads(INVENTORY.read_text())
        self.assertEqual(inventory['total_calls'],185)
        entries=inventory['calls'];self.assertEqual(len(entries),185)
        self.assertEqual(len({(r['file'],r['line']) for r in entries}),185)
        allowed={'mathematical_port','upstream_api','upstream_internal','upstream_search_policy'}
        executable=set(ORE_CASES)|set(MODULE_CASES)|{'input_rejection','parameter_degree_overflow','total_degree_one_empty',
            'chi_first_order_empty','short_rational_empty','short_algebraic_empty','fractional_holdout_rejected'}
        for entry in entries:
            self.assertIn(entry['disposition'],allowed)
            self.assertTrue(entry['reason'])
            if entry['disposition']=='mathematical_port':
                self.assertIn(entry['case'],executable)
            elif entry.get('mathematical_analogue'):
                self.assertIn(entry['mathematical_analogue'],executable)


for _name in ORE_CASES:
    setattr(GfunMathematicsTests,'test_ore_'+_name,lambda self,name=_name:self.check_ore(name))
for _name in MODULE_CASES:
    setattr(GfunMathematicsTests,'test_module_'+_name,lambda self,name=_name:self.check_module(name))


if __name__=='__main__':
    unittest.main()
