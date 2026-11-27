"""Serial execution preserves kernels, persistence and the caller's native limits."""
from unittest.mock import patch
import unittest
from threading import Event, Thread

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from hologuess import _qhsolve as core
from hologuess import _resources as resources


def native_counts():
    np.ones((2, 2)) @ np.ones((2, 2))
    return tuple((item['filepath'], item['num_threads']) for item in threadpool_info())


class RecordingStore(core.MemoryKernelStore):
    def __init__(self):
        super().__init__()
        self.completions = []

    def put_many(self, rows):
        self.completions.append(tuple(row[0] for row in rows))
        super().put_many(rows)


class SerialExecution(unittest.TestCase):
    def fixture(self, tensor):
        p = 101
        points = np.arange(2, 10, dtype=np.int64)
        if tensor:
            row = np.arange(1, 9, dtype=np.int64)[:, None]
            seq = np.stack([row*np.ones_like(points), row*row*np.ones_like(points),
                            row*(points+1)], axis=1) % p
            cells = [(0, 0), (1, 0), (2, 0)]
        else:
            seq = np.repeat(np.array([pow(2, i, p) for i in range(9)],
                                     dtype=np.int64)[:, None], len(points), axis=1)
            cells = [(0, 0), (1, 0)]
        return p, points, seq, cells, 8

    def test_serial_and_pool_have_identical_2d_and_tensor_kernels(self):
        for tensor in (False, True):
            with self.subTest(tensor=tensor):
                args = self.fixture(tensor)
                serial = core.solve_points(*args, procs=1, K=3,
                                           store=core.MemoryKernelStore())
                pooled = core.solve_points(*args, procs=2, K=3,
                                           store=core.MemoryKernelStore())
                np.testing.assert_array_equal(serial[0], pooled[0])
                np.testing.assert_array_equal(serial[1], pooled[1])

    def test_serial_path_does_not_construct_a_process_pool(self):
        with patch('concurrent.futures.ProcessPoolExecutor',
                   side_effect=AssertionError('serial execution must stay in process')):
            values, alive = core.solve_points(*self.fixture(True), procs=1, K=3,
                                              store=core.MemoryKernelStore())
        self.assertTrue(alive.all())
        self.assertEqual(values.shape, (8, 3))

    def test_serial_persists_each_chunk_and_cache_hits_do_not_resolve(self):
        args = self.fixture(True)
        store = RecordingStore()
        before = dict(core.KERNEL_STATS)
        first = core.solve_points(*args, procs=1, K=3, store=store)
        self.assertEqual([len(batch) for batch in store.completions], [3, 3, 2])
        self.assertEqual(core.KERNEL_STATS['solved']-before['solved'], 8)
        self.assertEqual(core.KERNEL_STATS['reused']-before['reused'], 0)
        before = dict(core.KERNEL_STATS)
        with patch.object(core, 'solve_points_task', side_effect=AssertionError('cache miss')):
            second = core.solve_points(*args, procs=1, K=3, store=store)
        self.assertEqual(core.KERNEL_STATS['solved']-before['solved'], 0)
        self.assertEqual(core.KERNEL_STATS['reused']-before['reused'], 8)
        self.assertEqual(len(store.completions), 3)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_serial_failure_keeps_completed_chunk_and_restores_threads(self):
        args = self.fixture(True)
        store = RecordingStore()
        original = core.solve_points_task
        calls = []
        def fail_second(task):
            calls.append(int(task[1][0]))
            # Tiny elementwise kernels need no library discovery. Trigger a real
            # BLAS path before checking that this operation retains its limits.
            core.gemm_mod(101, np.ones((2,2), dtype=np.int64),
                          np.ones((2,2), dtype=np.int64))
            self.assertTrue(all(count == 1 for path, count in native_counts()))
            if len(calls) == 2:
                raise RuntimeError('injected next-task failure')
            return original(task)
        with threadpool_limits(limits=3):
            before = native_counts()
            with patch.object(core, 'solve_points_task', side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError, 'next-task failure'):
                    core.solve_points(*args, procs=1, K=2, store=store)
            self.assertEqual(native_counts(), before)
        keys = core.point_keys(*args)
        self.assertEqual(set(store.get_many(keys)), set(keys[:2]))
        self.assertEqual(len(store.completions), 1)

    def test_non_line_kernel_is_rejected_without_persisting_invalid_batch(self):
        p, points, seq, cells, rows = self.fixture(True)
        # The first two action columns are independent, hence no line relation.
        cells = cells[:2]
        store = RecordingStore()
        with self.assertRaises(core.KernelNotALine) as caught:
            core.solve_points(p, points, seq, cells, rows, procs=1, K=3, store=store)
        self.assertEqual(caught.exception.dim, 0)
        self.assertEqual(store.completions, [])

    def test_outer_native_context_is_restoring_and_reentrant(self):
        with threadpool_limits(limits=3):
            before = native_counts()
            with resources.native_thread_context():
                self.assertTrue(all(n == 1 for path, n in native_counts()))
                with resources.native_thread_context():
                    self.assertTrue(all(n == 1 for path, n in native_counts()))
                self.assertTrue(all(n == 1 for path, n in native_counts()))
            self.assertEqual(native_counts(), before)
            with self.assertRaisesRegex(RuntimeError, 'context failure'):
                with resources.native_thread_context():
                    raise RuntimeError('context failure')
            self.assertEqual(native_counts(), before)

    def test_extraction_controller_is_refreshed_between_outer_contexts(self):
        import threadpoolctl
        real_factory = threadpoolctl.ThreadpoolController
        # The class may be imported locally by the implementation or exposed in
        # its resource module. Patch its import boundary rather than cache state.
        if hasattr(resources, 'ThreadpoolController'):
            target, name = resources, 'ThreadpoolController'
        else:
            target, name = threadpoolctl, 'ThreadpoolController'
        with patch.object(target, name, wraps=real_factory) as factory:
            with resources.native_thread_context():
                with resources.native_thread_context():
                    pass
            self.assertEqual(factory.call_count, 1)
            # A fresh boundary sees libraries a notebook may have loaded since
            # the previous extraction; an eternal singleton would miss them.
            with resources.native_thread_context():
                pass
            self.assertEqual(factory.call_count, 2)

    def test_independent_python_thread_contexts_cannot_overlap_restores(self):
        first_entered, second_attempted, second_entered = Event(), Event(), Event()
        release_first, release_second = Event(), Event()
        errors = []
        def first():
            try:
                with resources.native_thread_context():
                    first_entered.set()
                    if not release_first.wait(5): raise TimeoutError('first context release')
            except BaseException as error:
                errors.append(error)
        def second():
            try:
                second_attempted.set()
                with resources.native_thread_context():
                    second_entered.set()
                    if not release_second.wait(5): raise TimeoutError('second context release')
            except BaseException as error:
                errors.append(error)
        with threadpool_limits(limits=3):
            before = native_counts()
            a, b = Thread(target=first), Thread(target=second)
            try:
                a.start()
                self.assertTrue(first_entered.wait(5))
                b.start()
                self.assertTrue(second_attempted.wait(5))
                self.assertFalse(second_entered.wait(0.2),
                                 'process-global native contexts must serialize')
                release_first.set()
                self.assertTrue(second_entered.wait(5))
                self.assertTrue(all(n == 1 for path, n in native_counts()))
            finally:
                release_first.set(); release_second.set()
                if a.ident is not None: a.join(5)
                if b.ident is not None: b.join(5)
            self.assertFalse(a.is_alive() or b.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(native_counts(), before)

    def test_unblocked_kernels_do_not_discover_native_libraries(self):
        import threadpoolctl
        target = resources if hasattr(resources, 'ThreadpoolController') else threadpoolctl
        with patch.object(target, 'ThreadpoolController',
                          side_effect=AssertionError('elementwise path needs no controller')):
            a = np.arange(17*13, dtype=np.int64).reshape(17,13) % 101
            core.rref_mod(a,101)
            core.batched_nullspace(np.stack([a,a]),101)
            empty = core.gemm_mod(101,np.empty((3,0),dtype=np.int64),
                                  np.empty((0,4),dtype=np.int64))
            np.testing.assert_array_equal(empty,np.zeros((3,4),dtype=np.int64))

    def test_one_lazy_controller_across_multiple_blas_calls_and_error(self):
        import threadpoolctl
        factory_class = threadpoolctl.ThreadpoolController
        target = resources if hasattr(resources, 'ThreadpoolController') else threadpoolctl
        with threadpool_limits(limits=3):
            before = native_counts()
            observer = factory_class()
            with patch.object(target, 'ThreadpoolController', wraps=factory_class) as factory:
                with self.assertRaisesRegex(RuntimeError, 'after BLAS'):
                    with resources.native_operation():
                        self.assertEqual(factory.call_count,0)
                        for _ in range(2):
                            result = core.gemm_mod(101,np.ones((3,2),dtype=np.int64),
                                                   np.ones((2,4),dtype=np.int64))
                            np.testing.assert_array_equal(result,np.full((3,4),2))
                            self.assertEqual(factory.call_count,1)
                            self.assertTrue(all(info['num_threads'] == 1
                                                for info in observer.info()))
                        raise RuntimeError('after BLAS')
                self.assertEqual(factory.call_count,1)
            self.assertEqual(native_counts(),before)


if __name__ == '__main__':
    unittest.main()
