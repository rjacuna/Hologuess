import unittest
import numpy as np
from threadpoolctl import threadpool_limits, ThreadpoolController
from hologuess import _qhsolve as core


class OptionalInterpolation(unittest.TestCase):
    def test_fast_interpolation_remains_exact_and_restores_native_limits(self):
        p = core.PRIMES[0]
        points = np.arange(2,15,dtype=np.int64)
        exponents = list(range(-3,4))
        coefficients = np.array([[1,7],[0,4],[3,0],[8,9],[2,6],[5,0],[4,3]],dtype=np.int64)
        values = np.zeros((len(points),2),dtype=np.int64)
        for i,e in enumerate(exponents):
            power = core.pow_mod(points,e,p) if e>=0 else core.pow_mod(core.inv_mod(points,p),-e,p)
            values = (values+power[:,None]*coefficients[i]%p)%p
        observer = ThreadpoolController()
        with threadpool_limits(limits=3):
            before = observer.info()
            reconstructed,valid = core.vandermonde_solve(points,values,exponents,p,method='fast')
            self.assertTrue(valid)
            np.testing.assert_array_equal(reconstructed,coefficients)
            self.assertEqual([item['num_threads'] for item in observer.info()],
                             [item['num_threads'] for item in before])


if __name__=='__main__':unittest.main()
