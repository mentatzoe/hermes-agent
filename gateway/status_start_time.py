"""Comparators for :func:`gateway.status.get_process_start_time` fingerprints.

``gateway.status`` re-exports these names, so callers keep importing them from there.
"""

from __future__ import annotations

from typing import Any


def _start_times_agree(current: Any, *recorded: Any) -> bool:
    """Same process object: all fingerprints > 0 and within 1ms of ``current``; raises on junk."""
    cur = float(current)
    return cur > 0 and all(r > 0 and abs(r - cur) <= 0.001 for r in map(float, recorded))


# Same-host start-time readings can drift by ~1 s between the claim-time and a later liveness read
# (macOS ``kern.boottime`` adjustment, #117505). Both fingerprint scales are ×100 (Linux /proc ticks,
# psutil centiseconds), so 200 means 2 s on either platform — a recycled PID is essentially never
# that close to the original's start time.
START_TIME_DRIFT_TOLERANCE = 200


def start_time_fingerprints_match(recorded: Any, current: Any, tolerance: int = START_TIME_DRIFT_TOLERANCE) -> bool:
    """Liveness-reconciliation comparator for :func:`get_process_start_time` fingerprints: the
    recorded owner and the current reading are the same incarnation when they agree within
    ``tolerance``. Raises on junk; callers decide what an unreadable (``None``) side means."""
    return abs(int(current) - int(recorded)) <= tolerance
