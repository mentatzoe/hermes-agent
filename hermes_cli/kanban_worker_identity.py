"""Whether a live PID is still the kanban worker whose fingerprint was recorded at spawn.

Split out of ``hermes_cli.kanban_db_dispatch``, which keeps the decisions (``_worker_alive``,
``_pid_recycled``) and composes the fingerprint (``_process_fingerprint``); this owns the start-time
reading and the comparison.
"""

from __future__ import annotations

import sys
from typing import Any
from typing import Optional


def process_start_time(pid: int) -> Optional[int]:
    """``gateway.status.get_process_start_time``, read so that every process gets the same value.

    On macOS psutil's ``create_time()`` adds ``|boot time when psutil was imported - boot time now|``
    (whole seconds) to the start time the kernel stored at fork. ``kern.boottime`` moves: on 21 Oct
    2026 it went from ...909 to ...906 and back within an hour, so a gateway that had imported psutil
    at ...909 read every process's start 3 s later than a fresh ``hermes kanban dispatch`` did, and
    later than it had itself read an hour before (aleph-vault #215). The kernel's own value
    (``create_time(monotonic=True)``, which psutil uses for its own process identity) never moves, so
    it replaces the adjusted reading whenever both are available. It is psutil's private API (pinned
    7.2.2); if a later psutil drops it, this falls back to the ordinary reading, as it does off macOS.
    ``None`` when the process is unreadable.
    """
    from gateway.status import get_process_start_time

    start = get_process_start_time(int(pid))
    if start is None or sys.platform != "darwin":
        return start
    try:
        import psutil
    except ImportError:
        return start
    try:
        return int(round(psutil.Process(int(pid))._proc.create_time(monotonic=True) * 100))
    except (psutil.Error, AttributeError, TypeError, ValueError, OSError):
        return start


def fingerprints_match(recorded: Any, current: Any) -> bool:
    """True when ``current`` (read now) identifies the same process as ``recorded`` (stored at spawn).

    A composed ``"<instantiation epoch>|<start time>"`` fingerprint needs the same epoch, which
    changes on every reboot or container recreate. Its start halves may differ by up to
    ``START_TIME_DRIFT_TOLERANCE`` (2 s), the comparator the other owner-liveness checks use. With
    ``process_start_time`` both halves are normally equal; the tolerance still covers fingerprints
    recorded before it, which carry the recording process's psutil adjustment (1 s on 20 Oct, read as
    a recycled PID by an exact comparison; upstream #135344). Without ``/proc`` the epoch is empty on
    both sides, but the start time is then wall-clock time, so a process from an earlier boot can't
    come within 2 s of it.

    An integer fingerprint (rows written before the epoch was added) keeps the exact comparison:
    with no epoch, only an exact start time rules out a process from a previous boot. Spawns since the
    epoch was added write the composed form, so such a row ends with the worker that carries it.
    Nothing unreadable or malformed matches.
    """
    from gateway.status import _start_times_agree, start_time_fingerprints_match

    if current is None:
        return False
    if not (isinstance(recorded, str) and "|" in recorded):
        try:
            return _start_times_agree(current, recorded)
        except (TypeError, ValueError):
            return False
    if not (isinstance(current, str) and "|" in current):
        return False
    recorded_epoch, _, recorded_start = recorded.rpartition("|")
    current_epoch, _, current_start = current.rpartition("|")
    if recorded_epoch != current_epoch:
        return False
    try:
        return start_time_fingerprints_match(int(recorded_start), int(current_start))
    except (TypeError, ValueError):
        return False
