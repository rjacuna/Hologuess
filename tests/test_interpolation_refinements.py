"""Independent controls for the support-restriction stage of parameter guessing."""
import unittest
import numpy as np
from sage.all import QQ, PolynomialRing
from hologuess import ActionOracle, interpolate_relation
from hologuess import _qhsolve as core


class InterpolationRefinements(unittest.TestCase):
    def test_pruned_noncontiguous_support_keeps_original_coordinates(self):
        R = PolynomialRing(QQ, 't'); K = R.fraction_field(); t = K.gen()
        checked = []

        def evaluate(p, points):
            v = np.arange(1, 13, dtype=np.int64)[:, None]
            return np.stack([v*np.ones_like(points), v*v*np.ones_like(points),
                             v*(points+1)], axis=1) % p

        def validate(coefficients):
            weights = [K.zero() for _ in range(3)]
            for (column, exponent), coefficient in coefficients.items():
                weights[column] += coefficient*t**exponent
            checked.append(weights)
            return (weights[1] == 0 and weights[0] + (t+1)*weights[2] == 0
                    and weights[2] != 0)

        labels = ('v', 'independent action', '(t+1)v')
        result = interpolate_relation(ActionOracle(evaluate, labels, 12, validate),
                                      band=(-4,4), max_points=256)
        self.assertEqual(result.labels, labels)
        self.assertEqual(result.metadata['original_actions'], 3)
        self.assertEqual(result.metadata['retained_actions'], 2)
        self.assertEqual(result.metadata['support_indices'], (0,2))
        self.assertEqual({column for column, exponent in result.coefficients}, {0,2})
        self.assertTrue(result.metadata['exact_validation'])
        self.assertEqual(len(checked), 1)

    def test_unlucky_zero_probe_support_fails_closed(self):
        def evaluate(p, points):
            v = np.arange(1, 13, dtype=np.int64)[:, None]
            f = np.ones_like(points)
            for root in (2,3,4,5): f = f*(points-root) % p
            # The true relation is (t+1)A - B - (t+1)f(t)C = 0.
            # Its C coefficient vanishes at all four detection points.
            return np.stack([v+f*v*v, (points+1)*v, v*v*np.ones_like(points)], axis=1) % p
        with self.assertRaises(core.KernelNotALine):
            interpolate_relation(ActionOracle(evaluate, ('A','B','C'),12),
                                 band=(-4,4),max_points=256)

    def test_exact_validator_rejects_modular_observation_mismatch(self):
        def evaluate(p, points):
            v = np.arange(1, 13, dtype=np.int64)[:,None]
            return np.stack([v*np.ones_like(points),v*v*np.ones_like(points),
                             v*(points+1)],axis=1) % p
        with self.assertRaises(ArithmeticError):
            interpolate_relation(ActionOracle(evaluate, ('A','B','C'),12,
                                               lambda coefficients:False),
                                 band=(-4,4),max_points=256)


if __name__ == '__main__': unittest.main()
