"""agent/periodic_scheduler: one timer thread dispatches isolated callbacks.

A shorter interval is due more often when the timer wakes in time to tell the
intervals apart. A wake later than the slower interval fires each already-due
handle once — missed intervals are not caught up — so those counts match.
That is the lease and liveness contract, not a missed ratio. Regression for
mentatzoe/aleph-vault#266.
"""

import heapq
import threading
import time

from agent import periodic_scheduler
from agent.periodic_scheduler import PeriodicScheduler, schedule


def _wait_until(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


def _queue(sched, fn, interval):
    """Queue one handle the way ``schedule`` does, without starting the timer thread."""
    handle = periodic_scheduler.ScheduledHandle(sched, fn, float(interval))
    with sched._cond:
        sched._requeue(handle)
    return handle


def _fire_due_once(sched):
    """Dispatch each currently due handle once, through the real requeue path.

    ``_run`` uses the same comparison: ``due - now > 0`` waits, otherwise one
    pop and one dispatch. A handle whose next due time is still ahead does not
    fire again in that wake.
    """
    fired = 0
    now = periodic_scheduler.time.monotonic
    while True:
        with sched._cond:
            while sched._heap and sched._heap[0][2]._cancelled:
                heapq.heappop(sched._heap)
            if not sched._heap or sched._heap[0][0] > now():
                return fired
            handle = heapq.heappop(sched._heap)[2]
        sched._run_callback(handle)
        fired += 1


class _LateWakeScheduler(PeriodicScheduler):
    """Positive condition waits return at least ``overshoot`` seconds late."""

    def __init__(self, overshoot):
        super().__init__()
        real_wait = self._cond.wait

        def late(timeout=None):
            waited = real_wait(timeout)
            if timeout and timeout > 0 and overshoot > 0:
                deadline = time.monotonic() + overshoot
                while time.monotonic() < deadline:
                    time.sleep(0.005)
            return waited

        self._cond.wait = late


def test_shorter_interval_is_due_more_often_when_wakes_are_on_time(monkeypatch):
    clock = {"t": 0}
    monkeypatch.setattr(periodic_scheduler.time, "monotonic", lambda: clock["t"])
    sched = PeriodicScheduler()
    fast, slow = [], []
    _queue(sched, lambda: fast.append(clock["t"]), 1)
    _queue(sched, lambda: slow.append(clock["t"]), 5)

    for t in range(1, 16):
        clock["t"] = t
        _fire_due_once(sched)

    assert slow == [5, 10, 15]
    assert fast == list(range(1, 16))
    assert len(fast) > len(slow)

    # A jump past both intervals is one fire each. Catch-up would burst.
    clock["t"] = 100
    assert _fire_due_once(sched) == 2
    assert fast.count(100) == 1 and slow.count(100) == 1


def test_late_wake_fires_each_due_handle_once():
    sched = _LateWakeScheduler(0.07)
    fast, slow = [], []
    h_fast = sched.schedule(lambda: fast.append(1), 0.01)
    h_slow = sched.schedule(lambda: slow.append(1), 0.05)
    try:
        assert _wait_until(lambda: len(slow) >= 3)
        assert len(fast) == len(slow), (len(fast), len(slow))
    finally:
        h_fast.cancel(wait=2.0)
        h_slow.cancel(wait=2.0)


def test_cancel_stops_one_callback_without_stopping_its_sibling():
    sched = PeriodicScheduler()
    fast, slow = [], []
    h_fast = sched.schedule(lambda: fast.append(1), 0.01)
    h_slow = sched.schedule(lambda: slow.append(1), 0.05)
    assert _wait_until(lambda: fast and slow)
    assert sched._thread is not None and sched._thread.is_alive()

    h_fast.cancel(wait=2.0)
    n_fast, n_slow = len(fast), len(slow)
    assert _wait_until(lambda: len(slow) > n_slow), (
        "sibling callback stopped when another was cancelled"
    )
    assert len(fast) == n_fast, "cancelled callback kept firing"
    h_slow.cancel(wait=2.0)
    # With every handle quiesced, scheduling + cancelling adds no persistent thread.
    before = threading.active_count()
    sched.schedule(lambda: None, 0.01).cancel(wait=2.0)
    assert threading.active_count() == before


def test_raising_callback_is_rescheduled_and_does_not_kill_sibling():
    sched = PeriodicScheduler()
    boom, ok = [], []

    def raises():
        boom.append(1)
        raise RuntimeError("bad callback")

    h1 = sched.schedule(raises, 0.01)
    h2 = sched.schedule(lambda: ok.append(1), 0.01)
    assert _wait_until(lambda: len(boom) >= 3 and len(ok) >= 3)
    h1.cancel()
    h2.cancel()


def test_returning_false_stops_callback_and_cancel_wait_joins_inflight():
    sched = PeriodicScheduler()
    calls = []
    sched.schedule(lambda: (calls.append(1), False)[1], 0.01)
    assert _wait_until(lambda: len(calls) == 1)
    time.sleep(0.05)
    assert calls == [1]

    entered = threading.Event()
    release = threading.Event()

    def blocking():
        entered.set()
        release.wait(2.0)

    h = sched.schedule(blocking, 0.01)
    assert entered.wait(2.0)
    threading.Timer(0.05, release.set).start()
    t0 = time.monotonic()
    h.cancel(wait=2.0)  # returns once the in-flight run finished
    assert release.is_set()
    assert time.monotonic() - t0 < 1.5


def test_module_level_schedule_uses_shared_default():
    hits = []
    h = schedule(lambda: hits.append(1), 0.01)
    assert _wait_until(lambda: hits)
    h.cancel(wait=1.0)
    thread = periodic_scheduler._DEFAULT._thread
    assert thread is not None and thread.name == "hermes-periodic-scheduler"
    # Scheduling more timers on the shared default adds no persistent OS threads.
    before = threading.active_count()
    handles = [schedule(lambda: None, 0.01) for _ in range(20)]
    for handle in handles:
        handle.cancel(wait=1.0)
    assert threading.active_count() == before


def test_blocked_callback_does_not_stall_due_sibling(monkeypatch):
    scheduler = PeriodicScheduler()
    monkeypatch.setattr(periodic_scheduler, "_DEFAULT", scheduler)
    blocker_entered = threading.Event()
    release_blocker = threading.Event()
    sibling_ran = threading.Event()

    def blocker():
        blocker_entered.set()
        release_blocker.wait(5.0)
        return False

    def sibling():
        sibling_ran.set()
        return False

    blocker_handle = schedule(blocker, 0.01)
    assert blocker_entered.wait(2.0)
    sibling_handle = schedule(sibling, 0.01)
    try:
        # Ordering, not a wall-clock bound: the sibling must fire WHILE the blocker still holds
        # its worker. On main the sibling only runs after the blocker's 5 s wait expires.
        assert sibling_ran.wait(2.0) and not release_blocker.is_set(), (
            "a blocked periodic callback stalled an unrelated due callback"
        )
    finally:
        release_blocker.set()
        blocker_handle.cancel(wait=1.0)
        sibling_handle.cancel(wait=1.0)


def test_worker_start_failure_keeps_timer(monkeypatch):
    sched = PeriodicScheduler()
    fired: list = []
    real_thread = threading.Thread
    attempts = {"n": 0}

    def flaky(*args, **kwargs):
        # Only this scheduler's own callback worker fails, once; a leaked handle on the shared
        # _DEFAULT scheduler must not be the one that consumes the single Boom.
        # Bound methods are fresh objects per access: compare with ==, never `is`.
        if kwargs.get("target") == sched._run_callback and attempts["n"] == 0:
            attempts["n"] += 1

            class Boom:
                def start(self):
                    raise RuntimeError("no threads")

            return Boom()
        return real_thread(*args, **kwargs)

    monkeypatch.setattr(periodic_scheduler.threading, "Thread", flaky)
    handle = sched.schedule(lambda: fired.append(1), 0.01)
    try:
        assert _wait_until(lambda: bool(fired), timeout=3.0), (
            "worker-start failure silently retired the timer"
        )
        assert attempts["n"] == 1, "the fake never intercepted the callback worker"
        assert not handle.cancelled
    finally:
        handle.cancel(wait=1.0)
