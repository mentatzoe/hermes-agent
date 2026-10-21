"""Synchronization helpers for hosted-room runtime contract tests."""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import pytest

from tui_gateway import hosted_room_driver as driver

WAIT_TIMEOUT = 30.0


class RecordingTurnLocks:
    """Record the profile lock and expose ownership to the fake RPC."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.local = threading.local()

    @contextmanager
    def __call__(self, profile: str):
        self.events.append(("lock-enter", profile))
        self.local.profile = profile
        try:
            yield
        finally:
            self.events.append(("lock-exit", profile))
            self.local.profile = None

    def held_for(self, profile: str) -> bool:
        return getattr(self.local, "profile", None) == profile


def wait_for(predicate, *, timeout: float = WAIT_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


class DeadlineClock:
    """Advance the attempt deadline only after its first observation, not during I/O.

    Patch the driver's module reference, never the process-wide time module: pytest,
    threading and the test's own bounded progress waits keep their real clocks.
    """

    def __init__(self, monkeypatch) -> None:
        self.now = 100.0
        monkeypatch.setattr(driver, "time", self)

    def monotonic(self) -> float:
        return self.now

    def expire(self) -> None:
        self.now += 1.0


@pytest.fixture(autouse=True)
def managed_runtimes(monkeypatch):
    """A failed handshake must not leave a worker using the next test's fixtures."""
    started = []
    original = driver.HostedRoomRuntime.start

    def start(runtime):
        started.append(runtime)
        original(runtime)

    monkeypatch.setattr(driver.HostedRoomRuntime, "start", start)
    yield
    for runtime in reversed(started):
        assert runtime.stop(timeout=5.0), runtime.status()
