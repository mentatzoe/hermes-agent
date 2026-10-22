"""agent/periodic_scheduler: one timer thread dispatches isolated callbacks.

A shorter interval is due more often when the timer wakes in time to tell the
intervals apart. A wake later than the slower interval fires each already-due
handle once — missed intervals are not caught up — so those counts match.
That is the lease and liveness contract, not a missed ratio. Regression for
mentatzoe/aleph-vault#266.
"""

import threading
import time
from types import SimpleNamespace

import pytest

from agent import periodic_scheduler
from agent.periodic_scheduler import PeriodicScheduler, schedule


def _wait_until(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


class _ControlledWake:
    """Control only the clock and condition wake, not scheduling or dispatch.

    The real timer and callback threads run until the timer parks with every
    worker finished. Observations at that point cannot race a delayed worker.
    """

    def __init__(self, monkeypatch):
        self.now = 0.0
        self.scheduler = PeriodicScheduler()
        self.handles = []
        self.settled = threading.Event()
        monkeypatch.setattr(
            periodic_scheduler, "time", SimpleNamespace(monotonic=lambda: self.now)
        )
        self.real_wait = self.scheduler._cond.wait
        self.scheduler._cond.wait = self._wait

    def _wait(self, timeout=None):
        if all(h._runner is None for h in self.handles):
            self.settled.set()
        # Only the driver advances virtual time. Callback completion still
        # notifies this real condition, so dispatch and requeue run normally.
        return self.real_wait()

    def schedule_pair(self, fast, slow):
        with self.scheduler._cond:
            self.handles = [
                self.scheduler.schedule(fast, 0.01),
                self.scheduler.schedule(slow, 0.05),
            ]
        assert self.settled.wait(5.0), "timer did not park after scheduling"

    def advance(self, now):
        with self.scheduler._cond:
            self.settled.clear()
            self.now = now
            self.scheduler._cond.notify()
        assert self.settled.wait(5.0), "timer or callback did not finish the wake"

    def close(self):
        for handle in self.handles:
            handle.cancel(wait=5.0)
        with self.scheduler._cond:
            self.scheduler._cond.wait = self.real_wait
            self.scheduler._cond.notify()


@pytest.fixture
def controlled_wake(monkeypatch):
    wake = _ControlledWake(monkeypatch)
    try:
        yield wake
    finally:
        wake.close()


def test_shorter_interval_is_due_more_often_when_wakes_are_on_time(controlled_wake):
    wake = controlled_wake
    fast, slow = [], []
    wake.schedule_pair(lambda: fast.append(wake.now), lambda: slow.append(wake.now))

    # One microsecond per tick avoids floating-point boundary ambiguity while
    # remaining far shorter than the slow interval. No wall-clock rate assumed.
    ticks = [tick * 0.010001 for tick in range(1, 16)]
    for now in ticks:
        wake.advance(now - 0.005)
        wake.advance(now)

    assert fast == ticks
    assert slow == ticks[4::5]
    assert len(fast) > len(slow)


def test_late_wake_fires_each_due_handle_once(controlled_wake):
    wake = controlled_wake
    fast, slow = [], []
    release_fast = threading.Event()

    def fast_callback():
        # Force the last slow worker to finish first. The driver must wait for
        # this worker too, not compare live counters when only slow has finished.
        if len(fast) == 2:
            assert release_fast.wait(5.0)
        fast.append(wake.now)

    def slow_callback():
        slow.append(wake.now)
        if len(slow) == 3:
            release_fast.set()

    wake.schedule_pair(fast_callback, slow_callback)
    try:
        for now in (0.07, 0.14, 0.21):
            wake.advance(now)
            assert fast == slow == [t for t in (0.07, 0.14, 0.21) if t <= now]
    finally:
        release_fast.set()


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
