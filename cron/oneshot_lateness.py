"""One-shot cron jobs that fall due while dispatch is held, and the notice for one removed unrun.

The due scan retires a one-shot more than ``ONESHOT_GRACE_SECONDS`` late instead of firing it
hours late (#93526). A gateway restart breaks that rule's premise: ``/restart`` waits up to 30
minutes for running work, the ticker skips every tick meanwhile (``can_dispatch``), and the
replacement gateway's first scan found a reminder due in that wait already past its grace and
removed it unrun, leaving only a file in its output folder (aleph-vault#221).

* While dispatch is held, the ticker keeps ``cron/dispatch_held`` (``<since> <last>``: when the
  hold began, and its latest stamp, refreshed once a minute) in each store it ticks. The file
  outlives the process, so the next scan that may dispatch fires a one-shot that fell due during
  the hold, or in the restart gap after it, ONCE, if it is at most
  ``HELD_ONESHOT_MAX_LATE_SECONDS`` late. That scan's save deletes the file (the same pattern as
  the store-outage window in ``cron/store_health.py``).
* A one-shot fired late is stamped in ``last_dispatch`` and its prompt opens with when it was due.
* A one-shot retired unrun also queues a notice in the durable delivery queue
  (``cron/deliveries.db``), which the gateway sends through the job's failure lane.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("cron.jobs")

# How late a held one-shot may still fire: the recurring catch-up ceiling (``_MAX_GRACE_SECONDS``).
HELD_ONESHOT_MAX_LATE_SECONDS = 7200
_MARKER = "dispatch_held"
# A held ticker re-stamps the marker this often, so its "last" is at most a minute stale.
_REFRESH_SECONDS = 60.0
# How long after the last stamp the hold still covers: the stopping gateway's last minute, its
# exit, and the replacement's boot up to its first tick. A longer gap is downtime, not a restart.
_TAIL_SECONDS = 15 * 60.0
# Store key -> [since, last stamp] for the holds THIS process is in.
_holding: Dict[str, List[float]] = {}
# Job fields the delivery path reads; the notice must not carry the prompt or script.
_NOTICE_JOB_FIELDS = ("id", "name", "deliver", "failure_deliver", "origin", "attach_to_session")


def _cron_dir():
    from cron.jobs import _current_cron_store
    return _current_cron_store().cron_dir


def _read_marker() -> Optional[Tuple[float, float]]:
    try:
        fields = (_cron_dir() / _MARKER).read_text(encoding="utf-8-sig").split()
        return float(fields[0]), float(fields[-1])
    except (OSError, ValueError, IndexError):
        return None


def note_dispatch_held() -> None:
    """The ticker is skipping dispatch for the active store (restart wait, drain, stop): stamp the
    marker, at most once a minute. A write failure costs this protection, never the tick."""
    from cron.jobs import _hermes_now, _write_marker

    cron_dir = _cron_dir()
    key = os.path.realpath(cron_dir)
    now = _hermes_now().timestamp()
    state = _holding.get(key)
    if state is not None and now - state[1] < _REFRESH_SECONDS and (cron_dir / _MARKER).exists():
        return
    if state is None:
        # A hold the previous process left open moments ago is the same stretch without dispatch.
        previous = _read_marker()
        since = previous[0] if previous and now - previous[1] <= _TAIL_SECONDS else now
        state = _holding[key] = [since, now]
        logger.info("Cron dispatch held: a one-shot that falls due now fires once when it resumes")
    state[1] = now
    _write_marker(_MARKER, f"{state[0]} {now}\n", ".held_")


def note_dispatch_resumed() -> None:
    """Dispatch is allowed again in this process. The marker stays until the next due scan's save
    lands, so a one-shot that fell due during the hold still fires once."""
    _holding.pop(os.path.realpath(_cron_dir()), None)


def hold_covers(due_at: float, now: float) -> bool:
    """Whether a past-grace one-shot due at ``due_at`` fell due while dispatch was held (or in the
    restart gap right after) and is still recent enough to fire."""
    from cron.jobs import ONESHOT_GRACE_SECONDS

    marker = _read_marker()
    if marker is None:
        return False
    since, last = marker
    return (since - ONESHOT_GRACE_SECONDS <= due_at <= last + _TAIL_SECONDS
            and now - due_at <= HELD_ONESHOT_MAX_LATE_SECONDS)


def end_hold_window() -> None:
    """A due scan's save landed: the one-shots the hold covered are claimed or fired, so normal
    grace applies again. A store this process still holds keeps its marker."""
    cron_dir = _cron_dir()
    if os.path.realpath(cron_dir) in _holding:
        return
    try:
        (cron_dir / _MARKER).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not clear %s: %s", cron_dir / _MARKER, exc)


def late_dispatch_stamp(next_run: str, now, lateness: float) -> Dict[str, Any]:
    """``last_dispatch`` for a one-shot fired past its grace (store outage or held dispatch); the
    same ``catch_up`` kind a recurring job's past-grace run gets."""
    return {"scheduled_at": next_run, "dispatched_at": now.isoformat(),
            "lateness_seconds": round(lateness, 1), "kind": "catch_up"}


def _human_time(value: Any) -> str:
    from cron.jobs import _parse_aware

    dt = _parse_aware(value)
    return f"{dt:%H:%M} on {dt:%a} {dt.day} {dt:%b}" if dt is not None else str(value)


def late_run_note(job: Dict[str, Any]) -> str:
    """Opening paragraph for the scan's fire of a one-shot it let through late; "" otherwise
    (on-time runs, manual runs, a re-armed job's later occurrence)."""
    from cron.occurrences import scheduled_instant

    stamp = job.get("last_dispatch")
    schedule = job.get("schedule")
    if (not isinstance(schedule, dict) or schedule.get("kind") != "once"
            or not isinstance(stamp, dict) or stamp.get("kind") != "catch_up"
            or not job.get("_scheduled_instant")
            or scheduled_instant(stamp.get("scheduled_at")) != job["_scheduled_instant"]):
        return ""
    return (f"[Running late: this one-off job was due at {_human_time(stamp['scheduled_at'])}, but "
            "the scheduler couldn't run it then (the gateway was restarting or unavailable), so "
            f"it's running now, at {_human_time(stamp.get('dispatched_at'))}. Do the task as "
            "written. If its wording assumes the original time, make it fit the actual time, or "
            "reply exactly [SILENT] if it's no longer useful.]\n\n")


def queue_missed_notice(job: Dict[str, Any], next_run: str) -> None:
    """Queue the notice for a one-shot retired unrun; the gateway sends it on the failure lane.

    Runs once the scan's save lands (``_DueScan.on_saved``) and is keyed by job and instant, so a
    scan repeated after a failed save cannot queue the same miss twice. Nothing is queued for a
    ``local`` lane. Best effort: the diagnostic file is the durable record."""
    name = job.get("name") or job.get("id")
    try:
        from cron import delivery_queue
        from cron.occurrences import scheduled_instant
        from cron.scheduler_delivery import _delivery_lane_value, _normalize_deliver_value
        from cron.scheduler_failure_copy import missed_oneshot_notice

        if _normalize_deliver_value(_delivery_lane_value(job, for_failure=True)) == "local":
            return
        delivery_queue.enqueue(
            f"oneshot-missed:{job['id']}:{scheduled_instant(next_run) or next_run}",
            {key: job[key] for key in _NOTICE_JOB_FIELDS if key in job},
            missed_oneshot_notice(name, job["id"], _human_time(next_run)), for_failure=True)
    except Exception:  # an on_saved callback must not abort the scan or the callbacks after it
        logger.warning("Job '%s': could not queue its missed-run notice", name, exc_info=True)


def write_missed_oneshot_diagnostic(job: Dict[str, Any], next_run: str) -> None:
    """Trace for a never-ran one-shot retired outside the grace window (else it would just vanish).
    """
    from cron.jobs import ONESHOT_GRACE_SECONDS, _hermes_now, _write_oneshot_diagnostic

    _write_oneshot_diagnostic(
        job,
        "# Cron job removed before firing (run time outside grace window)\n\n"
        f"- job id: {job.get('id')}\n"
        f"- name: {job.get('name')}\n"
        f"- scheduled run time: {next_run}\n"
        f"- grace window: {ONESHOT_GRACE_SECONDS}s\n"
        f"- removed at: {_hermes_now().isoformat()}\n\n"
        "This one-shot's run time is more than the grace window in the "
        "past (scheduler down past the window, host asleep, or jobs.json "
        "edited), which is outside the 'will never fire' contract "
        "enforced at create/update/resume time. The job was removed "
        "without running; recreate it (or use the Run button) to "
        "schedule it again.\n",
        "missed-oneshot")
