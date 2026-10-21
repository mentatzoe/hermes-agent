"""Whether a live PID is still the kanban worker whose fingerprint was recorded at spawn.

Split out of ``hermes_cli.kanban_db_dispatch``, which keeps the decisions (``_worker_alive``,
``_pid_recycled``) and composes the fingerprint (``_process_fingerprint``); this owns the comparison
and the compatibility reading for fingerprints recorded before the unadjusted macOS start time.
"""

from __future__ import annotations

import sys
from typing import Any
from typing import Optional


def public_start_time(pid: int) -> Optional[int]:
    """psutil's public ``create_time()`` reading of ``pid`` on macOS, in the fingerprint's units
    (centiseconds since the epoch); ``None`` elsewhere or when the process is unreadable.

    ``gateway.status.get_process_start_time`` reads the kernel's stored start on macOS
    (``create_time(monotonic=True)``), which no ``kern.boottime`` movement changes. Before that,
    it read this value, which psutil adjusts by ``|boot time when psutil was imported - boot time
    now|`` once they differ by 1 s or more (aleph-vault #215). Fingerprints recorded by a process
    running the older code carry that adjustment, so ``recorded_fingerprint_matches`` also accepts
    them against this reading. Off macOS ``get_process_start_time`` never changed, so there is
    nothing to be compatible with.
    """
    if sys.platform != "darwin":
        return None
    try:
        import psutil
    except ImportError:
        return None
    try:
        return int(round(psutil.Process(int(pid)).create_time() * 100))
    except (psutil.Error, OSError, TypeError, ValueError):
        return None


def recorded_fingerprint_matches(recorded: Any, current: Any, public: Optional[int]) -> bool:
    """True when ``recorded`` matches ``current``, the reading of the PID now, or (only while
    ``current`` is readable) the same PID's public, psutil-adjusted start time ``public``
    (``public_start_time``), put in ``current``'s form: the same epoch prefix for a composed
    fingerprint, the bare value for an integer one.

    Compatibility rule for the switch to the unadjusted macOS start time: a row written by an
    older process carries the writer's psutil adjustment, so comparing it with the unadjusted value
    alone could turn a live worker the older code kept alive into a "recycled" PID and reclaim its
    card. Accepting either reading makes the decision the union of the old rule (public reading)
    and the new one (unadjusted reading): never stricter than either version, so mixed versions
    can't cause a false crash, and never looser than one of the two versions by itself. Both arms
    use ``fingerprints_match``, so the epoch half and the integer-row exactness still apply. An
    unreadable current process never matches, whatever ``public`` says.
    """
    if current is None:
        return False
    if fingerprints_match(recorded, current):
        return True
    if public is None:
        return False
    if isinstance(current, str) and "|" in current:
        return fingerprints_match(recorded, f"{current.rpartition('|')[0]}|{public}")
    return fingerprints_match(recorded, public)


def fingerprints_match(recorded: Any, current: Any) -> bool:
    """True when ``current`` (read now) identifies the same process as ``recorded`` (stored at spawn).

    A composed ``"<instantiation epoch>|<start time>"`` fingerprint needs the same epoch, which
    changes on every reboot or container recreate. Its start halves may differ by up to
    ``START_TIME_DRIFT_TOLERANCE`` (2 s), the comparator the other owner-liveness checks use. Since
    ``get_process_start_time`` reads the unadjusted start on macOS, both halves are normally equal;
    the tolerance still covers fingerprints recorded before that, which carry the recording
    process's psutil adjustment (1 s on 20 Oct, read as a recycled PID by an exact comparison;
    upstream #135344). Without ``/proc`` the epoch is empty on
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
