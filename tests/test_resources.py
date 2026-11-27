import unittest
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from hologuess._resources import limit_worker_threads


def count_threads():
    np.ones((2,2)) @ np.ones((2,2))
    return [info['num_threads'] for info in threadpool_info()]


class ResourceTests(unittest.TestCase):
    def test_preloaded_native_pools_are_limited_in_workers(self):
        # A notebook may have loaded BLAS before importing Hologuess.
        with threadpool_limits(limits=3):
            before = count_threads()
            with ProcessPoolExecutor(2,mp_context=get_context('fork'),
                                     initializer=limit_worker_threads) as pool:
                counts = list(pool.map(count_threads_for_task,range(2)))
            self.assertTrue(all(n == 1 for worker in counts for n in worker))
            self.assertEqual(count_threads(),before)


def count_threads_for_task(_): return count_threads()


if __name__ == '__main__': unittest.main()
