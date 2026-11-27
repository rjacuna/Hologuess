"""The precomputed pool is the original algorithm's pool, not a new prime choice."""
import unittest
from unittest.mock import patch
from sage.all import is_prime
from hologuess import _qhsolve as core
from hologuess._primes import reconstruction_primes


class PrimePool(unittest.TestCase):
    def test_exact_original_pool_values_and_order(self):
        prefix = [c*(1<<24)+1 for c in range(127,0,-1)
                  if is_prime(c*(1<<24)+1)]
        tail = []; candidate = (1<<31)-1
        while len(tail)<50:
            if candidate not in prefix and is_prime(candidate): tail.append(candidate)
            candidate -= 2
        self.assertEqual(core.PRIMES,prefix+tail)
        self.assertTrue(all(is_prime(p) for p in core.PRIMES))

    def test_trusted_pool_does_not_repeat_trial_division(self):
        with patch.object(core,'_is_prime',side_effect=AssertionError('trusted primes rechecked')):
            self.assertEqual(reconstruction_primes(),core.PRIMES)
            self.assertEqual(reconstruction_primes(core.PRIMES[:2]),core.PRIMES[:2])

    def test_unfamiliar_caller_primes_are_still_checked(self):
        with patch.object(core,'_is_prime',wraps=core._is_prime) as primality:
            self.assertEqual(reconstruction_primes([101,103]),[101,103])
            self.assertEqual(primality.call_count,2)
        for values in ([101,101],[101,49],[101,103.0],[True,103],[101],[] ):
            with self.subTest(values=values),self.assertRaises(ValueError):
                reconstruction_primes(values)


if __name__=='__main__':unittest.main()
