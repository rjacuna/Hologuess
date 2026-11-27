"""No-store reconstruction must not serialize a persistent-cache identity."""
import unittest
from unittest.mock import patch
from hologuess import ModuleProblem, solve_module


class UncachedModuleTests(unittest.TestCase):
    def test_no_cache_identity_without_a_store(self):
        with patch('hologuess.module._cache_key', side_effect=AssertionError('unexpected persistent-cache work')):
            result = solve_module(ModuleProblem([[1,2,3],[2,4,6]],training_rows=2))
        self.assertEqual(result.valid_basis, ((-2,1),))
        self.assertTrue(result.metadata['exact_training'])
        self.assertTrue(result.metadata['exact_holdout'])
        self.assertEqual(result.metadata['cache_hits'],0)


if __name__=='__main__':unittest.main()
