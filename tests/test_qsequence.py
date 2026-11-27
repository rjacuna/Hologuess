import unittest
import tempfile
from pathlib import Path
from fractions import Fraction

import numpy as np
from sage.all import GF, matrix

from hologuess import _qhsolve as core
from hologuess.qsequence import QSequence, guess_qsequence
from hologuess.interpolation import ActionOracle, interpolate_relation


class QSequenceTests(unittest.TestCase):
    def test_known_q_recurrence_and_nonzero_start(self):
        values = [{n*(n-1)//2: 1} for n in range(5, 25)]
        result = guess_qsequence(values, start=5, max_order=1, max_degree=1,
                                max_points=256)
        self.assertEqual(result.to_ore(), -result.to_ore().parent().gen() +
                         result.to_ore().parent().base_ring().gen(1))
        self.assertTrue(result.metadata['finite_window_valid'])
        self.assertFalse(result.metadata['all_indices_proved'])

    def test_rational_laurent_values_are_not_truncated(self):
        values = [{n: Fraction(1, 2)} for n in range(20)]
        result = guess_qsequence(values, max_order=1, max_degree=0, max_points=256)
        self.assertEqual(result.coefficients, {(0, 0, 1): 1, (1, 0, 0): -1})

    def test_zero_initial_polynomial_uses_safe_degree_fallback(self):
        values = [{n:1,0:-1} if n else {} for n in range(24)]
        result = guess_qsequence(values,max_order=1,max_degree=1,max_points=256)
        self.assertTrue(result.metadata['finite_window_valid'])
        self.assertFalse(result.metadata['band_predicted'])

    def test_blocked_elimination_matches_independent_sage(self):
        rng = np.random.default_rng(372)
        for p in (101, core.PRIMES[0]):
            a = rng.integers(0, p, size=(81, 73), dtype=np.int64)
            a[-3:] = a[:3]
            actual, pivots = core.rref_mod(a, p, panel=7)
            expected = matrix(GF(p), a.tolist()).rref()
            self.assertEqual(actual.tolist(), [[int(x) for x in row] for row in expected.rows()])
            self.assertEqual(len(pivots), expected.rank())

    def test_wang_noninvertible_denominator_rejected(self):
        self.assertIsNone(core.ratrec(36, 105))

    def test_batched_pivot_profiles_and_exact_kernels(self):
        p = 101
        rng = np.random.default_rng(56)
        data = rng.integers(0, p, size=(7, 80, 73), dtype=np.int64)
        data[:, :, -1] = data[:, :, 0]
        data[-1] = 0  # exceptional point may not vote against live profiles
        basis, alive = core.batched_nullspace(data, p)
        self.assertFalse(alive[-1])
        self.assertGreater(int(alive.sum()), 0)
        for i in np.flatnonzero(alive):
            expected = matrix(GF(p), data[i].tolist())
            self.assertEqual(basis.shape[1], expected.right_nullity())
            for v in basis[i]:
                self.assertEqual(expected * matrix(GF(p), 1, len(v), v.tolist()).transpose(), 0)

    def test_reject_approximate_coefficients_and_invalid_primes(self):
        with self.assertRaises(TypeError): QSequence([{0: 0.5}, {1: 0.5}])
        with self.assertRaises(ValueError):
            guess_qsequence([{n: 1} for n in range(20)], primes=[101, 101])

    def test_exact_sage_laurent_coefficients(self):
        from sage.all import QQ, LaurentPolynomialRing
        R = LaurentPolynomialRing(QQ, 'q'); q = R.gen()
        result = guess_qsequence([q**n/2 for n in range(20)], max_order=1,
                                 max_degree=0, max_points=256)
        self.assertTrue(result.metadata['finite_window_valid'])
        from sage.all import PolynomialRing
        from ore_algebra import OreAlgebra
        K = R.fraction_field()
        base = PolynomialRing(K,'Q'); Q = base.gen()
        A = OreAlgebra(base, ('E', {Q: K.gen()*Q}, {}))
        E = A.gen(); Q = A.base_ring().gen()
        self.assertEqual(result.to_ore(A), A(K.gen()) - E)

    def test_modular_oracle_does_not_truncate_approximate_data(self):
        for bad_points in (True, False):
            def oracle(p, count):
                points = np.arange(2,count+2,dtype=float if bad_points else np.int64)
                data = np.ones((20,count),dtype=np.int64 if bad_points else float)
                return points,data
            with self.assertRaises(TypeError): QSequence(at=oracle,terms=20).at(101,4)

    def test_modular_oracle_accepts_compatible_prefetched_points(self):
        def oracle(p,count):
            points = np.arange(2,count+5,dtype=np.int64)
            return points,np.ones((20,len(points)),dtype=np.int64)
        seq = QSequence(at=oracle,terms=20)
        points,values = seq.at(101,4)
        self.assertEqual(len(points),7)
        self.assertEqual(values.shape,(20,7))
        result = guess_qsequence(seq,max_order=1,max_degree=0,max_points=256)
        self.assertEqual(result.coefficients,{(0,0,0):1,(1,0,0):-1})

    def test_module_interpolation_clears_nonmonomial_denominator(self):
        def evaluate(p, points):
            v = np.arange(1, 13, dtype=np.int64)[:, None]
            return np.stack([v * (points+1) % p, v * (points*points+1) % p], axis=1)
        def exact_validate(c):
            from sage.all import PolynomialRing, QQ
            R = PolynomialRing(QQ, 't'); t = R.gen()
            a = sum(v*t**e for (i,e),v in c.items() if i == 0)
            b = sum(v*t**e for (i,e),v in c.items() if i == 1)
            return a*(t+1)+b*(t*t+1) == 0 and a != 0
        oracle = ActionOracle(evaluate, ('A', 'B'), 12, exact_validate)
        with tempfile.TemporaryDirectory() as d:
            store = core.SQLiteKernelStore(str(Path(d)/'kernels.sqlite'))
            result = interpolate_relation(oracle, band=(-4, 4), max_points=256, store=store)
            solved = core.KERNEL_STATS['solved']
            again = interpolate_relation(oracle, band=(-4, 4), max_points=256, store=store)
            self.assertEqual(core.KERNEL_STATS['solved'], solved)
        self.assertEqual(result.coefficients, again.coefficients)
        self.assertTrue(result.metadata['exact_validation'])
        self.assertTrue(result.metadata['denominator_cleared'])

    def test_tensor_cache_keys_bind_action_order_and_observations(self):
        q = np.array([2,3], dtype=np.int64)
        a = np.arange(24, dtype=np.int64).reshape(6,2,2)
        old = core.point_keys(101,q,a,[(0,0),(1,0)],6)
        reordered = core.point_keys(101,q,a,[(1,0),(0,0)],6)
        self.assertNotEqual(old,reordered)
        a[-1,-1,-1] += 1
        self.assertNotEqual(old,core.point_keys(101,q,a,[(0,0),(1,0)],6))
        sequence = np.arange(16,dtype=np.int64).reshape(8,2)
        self.assertNotEqual(core.point_keys(101,q,sequence,[(0,0),(1,0)],6),
                            core.point_keys(101,q,sequence,[(1,0),(0,0)],6))

    def test_generic_probe_skips_exceptional_first_image(self):
        def evaluate(p, points):
            v = np.arange(1, 13, dtype=np.int64)[:, None]
            return np.stack([v*(points-2)%p, v*((points-2)**2)%p], axis=1)
        result = interpolate_relation(ActionOracle(evaluate, ('a','b'), 12),
                                      band=(-4,4), max_points=256)
        self.assertTrue(result.coefficients)


if __name__ == '__main__': unittest.main()
