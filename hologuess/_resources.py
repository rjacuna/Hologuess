"""Restoring, operation-scoped native limits for notebook and worker execution."""
from contextlib import contextmanager, ExitStack
from functools import wraps
from threading import local, RLock

_state = local()
_operation_lock = RLock()
_limit = None


@contextmanager
def native_thread_context():
    """Discover loaded libraries once per operation and restore caller limits.

    Nested point batches share that controller. A new outer operation creates a
    fresh controller, including libraries imported since the preceding operation.
    NumPy (the batch worker's backend) is loaded before this context is entered.
    """
    if getattr(_state, 'depth', 0):
        _state.depth += 1
        try:
            yield
        finally:
            _state.depth -= 1
        return
    # Native limits are process-global. Serialize independent Python-thread
    # contexts so overlapping restores cannot leak a changed notebook setting.
    with _operation_lock:
        from threadpoolctl import ThreadpoolController
        controller = ThreadpoolController()
        _state.controller, _state.depth = controller, 1
        try:
            with controller.limit(limits=1):
                yield
        finally:
            del _state.controller
            del _state.depth


def limit_extraction_threads(function):
    @wraps(function)
    def limited(*args, **kwargs):
        with native_operation():
            return function(*args, **kwargs)
    return limited


@contextmanager
def native_operation():
    """Keep a lazy native-limit lifetime across an entire reconstruction.

    Elementwise/unblocked NumPy work does not use BLAS pools. Discover and limit
    libraries only when a real BLAS product or a worker pool is first needed.
    """
    if getattr(_state, 'operation_depth', 0):
        _state.operation_depth += 1
        try:
            yield
        finally:
            _state.operation_depth -= 1
        return
    _state.operation_depth = 1
    try:
        with ExitStack() as stack:
            _state.stack, _state.limited = stack, False
            yield
    finally:
        del _state.operation_depth
        del _state.stack
        del _state.limited


@contextmanager
def operation_native_limits():
    if getattr(_state, 'operation_depth', 0):
        if not _state.limited:
            _state.stack.enter_context(native_thread_context())
            _state.limited = True
        yield
    else:
        with native_thread_context():
            yield


def limit_worker_threads():
    """Persist limits in a forked worker, reusing the prepared parent controller."""
    global _limit
    controller = getattr(_state, 'controller', None)
    if controller is None:
        from threadpoolctl import ThreadpoolController
        controller = ThreadpoolController()
    _limit = controller.limit(limits=1)
