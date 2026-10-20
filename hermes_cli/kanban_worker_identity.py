"""Whether a live PID is still the kanban worker whose fingerprint was recorded at spawn.

Split out of ``hermes_cli.kanban_db_dispatch``, which keeps the decisions (``_worker_alive``,
``_pid_recycled``) and the fingerprint itself (``_process_fingerprint``); this owns the comparison.
"""

from __future__ import annotations

from typing import Any


def fingerprints_match(recorded: Any, current: Any) -> bool:
    """True when ``current`` (read now) identifies the same process as ``recorded`` (stored at spawn).

    A composed ``"<instantiation epoch>|<start time>"`` fingerprint needs the same epoch, which
    changes on every reboot or container recreate. Its start halves may differ by up to
    ``START_TIME_DRIFT_TOLERANCE`` (2 s): on macOS psutil shifts ``create_time()`` by however far
    ``kern.boottime`` has moved since psutil was imported, so a gateway that has been up for a day and
    a fresh ``hermes kanban dispatch`` read the same worker's start about a second apart. Compared
    exactly, each read the other's live workers as recycled PIDs, reclaimed their cards and started a
    second worker beside the first (aleph-vault #215; upstream #135344). Without ``/proc`` the epoch
    is empty on both sides, but the start time is then wall-clock time, so a process from an earlier
    boot can't come within 2 s of it.

    An integer fingerprint (rows written before the epoch was added) keeps the exact comparison:
    with no epoch, only an exact start time rules out a process from a previous boot. Nothing
    unreadable or malformed matches.
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
