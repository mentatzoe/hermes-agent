"""Thread ownership for synchronization tests: join before fixture patches unwind."""

import threading
import time


class CheckedThread(threading.Thread):
    """Keep worker failures visible to the test instead of a thread-warning only."""

    def __init__(self, target, *, name):
        super().__init__(name=name, daemon=True)
        self.callback = target
        self.value = None
        self.error = None

    def run(self):
        try:
            self.value = self.callback()
        except BaseException as exc:
            self.error = exc


def join_checked(*threads, timeout=5):
    """One overall cleanup deadline; callers release their fixture blockers first."""
    deadline = time.monotonic() + timeout
    for thread in threads:
        if thread.ident is not None:
            thread.join(max(0, deadline - time.monotonic()))
    alive = [thread.name for thread in threads if thread.is_alive()]
    assert not alive, f"test workers did not finish: {alive}"
    for thread in threads:
        if thread.error is not None:
            raise thread.error
