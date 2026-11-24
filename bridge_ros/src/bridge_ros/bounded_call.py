"""Bound blocking ROS service calls without accumulating abandoned workers."""
import threading


class CallPending(TimeoutError):
    """The same service already has an outstanding, bounded caller."""


class BoundedCall:
    def __init__(self, name):
        self.name = name
        self.lock = threading.Lock()
        self.pending = None

    def __call__(self, function, *args, timeout=1.):
        with self.lock:
            if self.pending is not None and not self.pending['done'].is_set():
                raise CallPending(self.name+' previous call is still pending')
            call = {'done': threading.Event()}
            self.pending = call

            def invoke():
                try:
                    call['value'] = function(*args)
                except Exception as exc:
                    call['error'] = exc
                finally:
                    call['done'].set()

            threading.Thread(target=invoke, daemon=True).start()
        if not call['done'].wait(timeout):
            raise TimeoutError(self.name+' response timeout')
        if 'error' in call:
            raise call['error']
        return call['value']
