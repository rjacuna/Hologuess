"""Independent exact actions versus input-first modular adapter images."""
from itertools import product
from math import comb, factorial
import unittest

import numpy as np
from sage.all import QQ, ZZ, GF, PolynomialRing, PowerSeriesRing, matrix
from ore_algebra import OreAlgebra
from hologuess.adapters import prepare
from hologuess.module import _expand, solve_module
from hologuess import _qhsolve as core


def literal_matrix(data, kinds, orders, degree, *, starts=None,
                   point_filter=None, term_filter=None, training_rows=None):
    """Factorial/difference identities, not recursive modular action arrays."""
    dimension = len(kinds)
    shape = (len(data),) if dimension == 1 else (len(data), len(data[0]))
    starts = starts or (0,)*dimension
    points = [at for at in product(*(range(n-r) for n,r in zip(shape,orders)))
              if point_filter is None or point_filter(*(n+s for n,s in zip(at,starts)))]
    if training_rows is not None:
        points = points[:training_rows]
    coefficients = [u for u in product(range(degree+1), repeat=dimension) if sum(u)<=degree]
    actions = [v for v in product(*(range(r+1) for r in orders))
               if term_filter is None or term_filter(v)]
    rows = []
    for at in points:
        row = []
        for v in actions:
            for u in coefficients:
                total = QQ.zero()
                difference_axes = [axis for axis,(kind,_) in enumerate(kinds) if kind=='F']
                for shifts in product(*(range(v[axis]+1) for axis in difference_axes)):
                    target = list(at); weight = QQ.one()
                    for axis,((kind,q),b,a,n) in enumerate(zip(kinds,v,u,at)):
                        if kind=='D':
                            if n<a:
                                weight = QQ.zero(); break
                            k = n-a
                            target[axis] = k+b
                            weight *= factorial(k+b)//factorial(k)
                        elif kind=='F':
                            k = shifts[difference_axes.index(axis)]
                            target[axis] += k
                            weight *= (-1)**(b-k)*comb(b,k)*(n+starts[axis])**a
                        else:
                            target[axis] += b
                            weight *= q**((n+starts[axis])*a) if kind=='Q' else (n+starts[axis])**a
                    value = data[target[0]] if dimension==1 else data[target[0]][target[1]]
                    total += weight*QQ(value)
                row.append(total)
        rows.append(row)
    return matrix(QQ,rows)


class ModularInputActionsTests(unittest.TestCase):
    def compare(self, problem, expected, primes=(101, 1009, 2130706433)):
        self.assertIs(problem._modular_columns, problem.columns)
        self.assertEqual(problem._modular_training_rows, problem.training_rows)
        _, actual, _, _ = _expand(problem)
        self.assertEqual(actual, expected)
        for p in primes:
            image = problem._modular_matrix(p)
            self.assertEqual(image.dtype, np.dtype(np.int64))
            self.assertEqual(image.shape, (expected.nrows(), expected.ncols()))
            self.assertEqual(matrix(GF(p),image.tolist()), expected.change_ring(GF(p)))
            self.assertTrue(np.all(image>=0))
            self.assertTrue(np.all(image<p))

    def test_derivative_rational_data_and_holdout(self):
        R=PolynomialRing(QQ,'x'); A=OreAlgebra(R,'Dx')
        data=[QQ((-1)**k*(k+2))/(k+1) for k in range(24)]
        p=prepare(data,A,order=5,degree=4,holdout=3)
        self.compare(p,literal_matrix(data,[('D',None)],(5,),4,training_rows=16))

    def test_high_order_derivative_factors(self):
        R=PolynomialRing(QQ,'x'); A=OreAlgebra(R,'Dx')
        data=[QQ(2*k+1)/(k+1) for k in range(48)]
        p=prepare(data,A,order=15,degree=15)
        self.compare(p,literal_matrix(data,[('D',None)],(15,),15))

    def test_small_characteristic_derivatives_vanish_at_prime_order(self):
        R=PolynomialRing(QQ,'x'); A=OreAlgebra(R,'Dx')
        data=[k*k+2 for k in range(18)]
        p=prepare(data,A,order=7,degree=4)
        self.compare(p,literal_matrix(data,[('D',None)],(7,),4),primes=(3,5,7,2130706433))

    def test_integer_parent_shift_with_huge_start(self):
        R=PolynomialRing(ZZ,'n'); A=OreAlgebra(R,'Sn')
        data=[QQ(3*k-2)/(k+1) for k in range(18)]
        start=10**50+17
        p=prepare(data,A,order=3,degree=4,start=start)
        self.compare(p,literal_matrix(data,[('S',None)],(3,),4,starts=(start,)))

    def test_forward_difference_reuses_recursive_images(self):
        R=PolynomialRing(QQ,'n'); A=OreAlgebra(R,'Fn')
        data=[QQ(k*k-3*k+2)/(k+1) for k in range(24)]
        p=prepare(data,A,order=7,degree=3,start=5)
        self.compare(p,literal_matrix(data,[('F',None)],(7,),3,starts=(5,)))

    def test_rational_q_shift(self):
        R=PolynomialRing(QQ,'Q'); Q=R.gen(); q=QQ(2)/3
        A=OreAlgebra(R,('E',{Q:q*Q},{}))
        data=[QQ(2*k+1)/(k+2) for k in range(18)]
        p=prepare(data,A,order=3,degree=4,start=7)
        self.compare(p,literal_matrix(data,[('Q',q)],(3,),4,starts=(7,)))

    def test_algebraic_convolution_cannot_overflow_int64(self):
        R=PolynomialRing(QQ,'x'); A=OreAlgebra(R,'C')
        data=[QQ(2130706432-k)/(k+1) for k in range(20)]
        p=prepare(data,A,order=5,degree=3,holdout=2)
        T=PowerSeriesRing(QQ,'t',default_prec=len(data)); f=T(data)
        rows=[[QQ((f**b)[n-a]) if n>=a else QQ.zero()
               for b in range(6) for a in range(4)] for n in range(18)]
        self.compare(p,matrix(QQ,rows))

    def test_mixed_shift_derivative_filters_and_holdout(self):
        R=PolynomialRing(QQ,('n','x')); A=OreAlgebra(R,'Sn','Dx')
        data=[[QQ(2*n+3*k+1)/(n+k+1) for k in range(11)] for n in range(10)]
        point_filter=lambda n,k:(n+k)%3!=1
        term_filter=lambda v:sum(v)<=2
        p=prepare(data,A,order=(2,3),degree=3,start=(4,0),
                  point_filter=point_filter,term_filter=term_filter,training_rows=31)
        self.compare(p,literal_matrix(data,[('S',None),('D',None)],(2,3),3,
                     starts=(4,0),point_filter=point_filter,term_filter=term_filter,training_rows=31))

    def test_mixed_forward_difference_shift(self):
        R=PolynomialRing(QQ,('n','k')); A=OreAlgebra(R,'Fn','Sk')
        data=[[QQ(n*n+3*k+1)/(n+k+1) for k in range(10)] for n in range(9)]
        p=prepare(data,A,order=(3,2),degree=3,start=(5,7))
        self.compare(p,literal_matrix(data,[('F',None),('S',None)],(3,2),3,starts=(5,7)))

    def test_generator_order_can_differ_from_axis_order(self):
        R=PolynomialRing(QQ,('n','x')); A=OreAlgebra(R,'Dx','Sn')
        data=[[QQ(3*n-k+2)/(n+k+1) for k in range(10)] for n in range(9)]
        p=prepare(data,A,order=(2,3),degree=2,start=(4,0))
        axis_order=literal_matrix(data,[('S',None),('D',None)],(3,2),2,starts=(4,0))
        # Reference helper orders its operators by axes S,D; requested generators
        # are D,S, so permute only the separately constructed reference columns.
        permutation=[(s*3+d)*6+u for d in range(3) for s in range(4) for u in range(6)]
        exact=axis_order.matrix_from_columns(permutation)
        self.compare(p,exact)
        first=p.labels.index(R.gens()[1]*p.labels[0].parent().gen(0))
        self.assertEqual(list(p.columns[first])[:3],[0,data[0][1],2*data[0][2]])

    def test_raw_pole_requests_fallback_after_action_cancellation(self):
        R=PolynomialRing(QQ,'x'); A=OreAlgebra(R,'Dx')
        p=prepare([0,0,0,QQ(1)/3],A,order=3,degree=0,term_filter=lambda v:v==(3,))
        with self.assertRaises(NotImplementedError):p._modular_matrix(3)
        _, exact, _, _=_expand(p)
        self.assertEqual(exact,matrix(QQ,[[2]]))
        self.assertEqual(exact.change_ring(GF(3)),matrix(GF(3),[[2]]))

    def test_unused_holdout_pole_requests_fallback(self):
        R=PolynomialRing(QQ,'n'); A=OreAlgebra(R,'Sn')
        p=prepare([1,2,QQ(1)/3],A,order=0,degree=0,training_rows=2)
        with self.assertRaises(NotImplementedError):p._modular_matrix(3)
        self.assertEqual(_expand(p)[1].change_ring(GF(3)),matrix(GF(3),[[1],[2]]))

    def test_unused_q_with_pole_is_not_reduced(self):
        R=PolynomialRing(QQ,'Q'); Q=R.gen(); A=OreAlgebra(R,('E',{Q:Q/3},{}))
        p=prepare([1,2,3,4,5],A,order=1,degree=0)
        self.compare(p,matrix(QQ,[[1,2],[2,3],[3,4],[4,5]]),primes=(3,5))

    def test_q_pole_can_cancel_on_selected_zero_index(self):
        R=PolynomialRing(QQ,'Q'); Q=R.gen(); A=OreAlgebra(R,('E',{Q:Q/3},{}))
        p=prepare([1,2,3],A,order=0,degree=1,point_filter=lambda n:n==0)
        with self.assertRaises(NotImplementedError):p._modular_matrix(3)
        self.assertEqual(_expand(p)[1],matrix(QQ,[[1,1]]))
        result=solve_module(p,primes=[3,5,7])
        self.assertEqual(result.basis,((QQ(-1),QQ(1)),))
        self.assertEqual(result.metadata['skipped_primes'],[])

    def test_actual_training_pole_still_rejects_prime(self):
        R=PolynomialRing(QQ,'n'); A=OreAlgebra(R,'Sn')
        p=prepare([QQ(1)/3,2,3],A,order=0,degree=0)
        result=solve_module(p,primes=[3,5,7])
        self.assertEqual(result.status,'no_relations')
        self.assertEqual(result.metadata['skipped_primes'],[(3,'denominator')])

    def test_input_oracle_persistent_cache_matches_fallback(self):
        from hologuess.module import ModuleProblem
        R=PolynomialRing(QQ,'n'); A=OreAlgebra(R,'Sn')
        p=prepare([QQ(3)**k/QQ(2)**(k+1) for k in range(12)],A,order=1,degree=0,holdout=2)
        store=core.MemoryKernelStore()
        first=solve_module(p,store=store)
        again=solve_module(p,store=store)
        fallback=solve_module(ModuleProblem(p.columns,labels=p.labels,training_rows=p.training_rows),store=store)
        self.assertEqual(first.basis,again.basis)
        self.assertEqual(first.basis,fallback.basis)
        self.assertGreater(again.metadata['cache_hits'],0)
        self.assertGreater(fallback.metadata['cache_hits'],0)
        self.assertTrue(again.metadata['exact_holdout'])

    def test_input_oracle_rejects_corrupt_cached_kernel(self):
        from hologuess.module import _cache_key
        R=PolynomialRing(QQ,'n'); A=OreAlgebra(R,'Sn')
        p=prepare([1,2,4,8,16,32],A,order=1,degree=0)
        exact=_expand(p)[1]
        store=core.MemoryKernelStore()
        store.put_many([(_cache_key(exact,101),101,1,np.array([1,0,0,1]))])
        with self.assertRaisesRegex(ValueError,'modular residual'):
            solve_module(p,primes=[101,103,107],store=store)

    def test_unsupported_constant_domains_keep_fallback(self):
        for K in (GF(101),PolynomialRing(QQ,'t').fraction_field()):
            R=PolynomialRing(K,'n'); A=OreAlgebra(R,'Sn')
            p=prepare([K(k+1) for k in range(8)],A,order=1,degree=1)
            self.assertFalse(hasattr(p,'_modular_matrix'))

    def test_changed_columns_disable_stale_callback(self):
        R=PolynomialRing(QQ,'n'); A=OreAlgebra(R,'Sn')
        p=prepare([1,2,4,8,16,32],A,order=1,degree=0)
        p.columns=tuple(tuple(c if j==0 else c+1 for c in col) for j,col in enumerate(p.columns))
        p._modular_matrix=lambda prime: (_ for _ in ()).throw(AssertionError('stale callback'))
        result=solve_module(p)
        self.assertEqual(result.status,'no_relations')

    def test_changed_training_count_disables_stale_callback(self):
        R=PolynomialRing(QQ,'n'); A=OreAlgebra(R,'Sn')
        p=prepare([1,2,4,8,16,32],A,order=1,degree=0)
        p.training_rows=3
        p._modular_matrix=lambda prime: (_ for _ in ()).throw(AssertionError('stale callback'))
        result=solve_module(p)
        self.assertEqual(result.valid_basis,((QQ(-2),QQ(1)),))
        self.assertTrue(result.metadata['exact_holdout'])

    def test_filtered_high_order_state_budget_uses_exact_fallback(self):
        R=PolynomialRing(QQ,'x'); A=OreAlgebra(R,'Dx')
        data=[QQ(1)/factorial(n) for n in range(64)]
        p=prepare(data,A,order=20,degree=0,point_filter=lambda n:n==0)
        with self.assertRaisesRegex(NotImplementedError,'state bound'):
            p._modular_matrix(3)
        expected=matrix(QQ,[[1]*21])
        self.assertEqual(_expand(p)[1],expected)
        result=solve_module(p,primes=[3,5,7])
        self.assertEqual(matrix(QQ,result.basis).row_space(),expected.right_kernel())
        self.assertEqual(result.metadata['skipped_primes'],[])
        self.assertEqual(result.metadata['modular_input_fallbacks'],[3,5])

    def test_sparse_high_order_term_budget_counts_intermediate_states(self):
        R=PolynomialRing(QQ,'x'); A=OreAlgebra(R,'Dx')
        data=[QQ(1)/factorial(n) for n in range(64)]
        p=prepare(data,A,order=63,degree=0,term_filter=lambda v:v==(63,))
        with self.assertRaisesRegex(NotImplementedError,'state bound'):
            p._modular_matrix(3)
        self.assertEqual(_expand(p)[1],matrix(QQ,[[1]]))
        result=solve_module(p,primes=[3,5,7])
        self.assertEqual(result.basis,())
        self.assertEqual(result.status,'no_relations')
        self.assertEqual(result.metadata['skipped_primes'],[])
        self.assertEqual(result.metadata['modular_input_fallbacks'],[3])


if __name__=='__main__':unittest.main()
