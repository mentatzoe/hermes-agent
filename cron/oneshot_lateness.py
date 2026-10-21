"""One-shot cron jobs that fall due while dispatch is held, and the notice for one removed unrun.

The due scan retires a one-shot more than ``ONESHOT_GRACE_SECONDS`` late instead of firing it
hours late (#93526). A gateway restart breaks that rule's premise: ``/restart`` waits up to 30
minutes for running work, the ticker skips every tick meanwhile (``can_dispatch``), and the
replacement gateway's first scan found a reminder due in that wait already past its grace and
removed it unrun, leaving only a file in its output folder (aleph-vault#221).

* **Hold marker.** Every held tick stamps ``cron/dispatch_held`` (``<since> <last>``, epoch
  seconds) in each store the ticker visits. The file is the only state, so it outlives the process
  and every process's scan reads the same answer. A one-shot past its grace still fires once when
  it fell due between ``since`` and shortly after ``last``, the marker was stamped recently (a
  restart, not downtime), and it is at most ``HELD_ONESHOT_MAX_LATE_SECONDS`` late. A scan deletes
  the marker only once nobody has stamped it for ``_RELEASE_SECONDS``: a CLI scan during the hold,
  or one right after the hold ends, leaves it for the next gateway's first scan.
* **Late run.** A one-shot fired past its grace is stamped in ``last_dispatch`` and its prompt opens
  with when it was due. A hold-admitted claim carries ``until`` (due + the 2 h ceiling), which
  ``clear_run_claim`` keeps, so a failed dispatch is retried only inside that window.
* **Skip notice.** A one-shot retired unrun also queues a notice in the durable delivery queue
  (``cron/deliveries.db``), which the gateway sends through the job's failure lane.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("cron.jobs")

# How late a held one-shot may still fire: the recurring catch-up ceiling (``_MAX_GRACE_SECONDS``).
HELD_ONESHOT_MAX_LATE_SECONDS = 7200
_MARKER = "dispatch_held"
# A held ticker stamps on every tick (60 s); a marker this long unstamped belongs to no live hold.
_RELEASE_SECONDS = 150.0
# How long after its last stamp a hold still covers one-shots: the stopping gateway's last minute,
# its exit, and the replacement's boot up to its first tick. Longer is downtime, not a restart.
_TAIL_SECONDS = 15 * 60.0
# Job fields the delivery path reads; the notice must not carry the prompt or script.
_NOTICE_JOB_FIELDS = ("id", "name", "deliver", "failure_deliver", "origin", "attach_to_session")


def _marker_path():
    from cron.jobs import _current_cron_store
    return _current_cron_store().cron_dir / _MARKER


def _read_marker() -> Optional[Tuple[float, float]]:
    try:
        fields = _marker_path().read_text(encoding="utf-8-sig").split()
        return float(fields[0]), float(fields[-1])
    except (OSError, ValueError, IndexError):
        return None


def note_dispatch_held() -> None:
    """The ticker is skipping dispatch for the active store (restart wait, drain, stop): stamp the
    marker. A hold stamped moments ago (by this process or the previous one) is the same gap and
    keeps its start. A write failure costs this protection, never the tick."""
    from cron.jobs import _hermes_now, _write_marker

    now = _hermes_now().timestamp()
    previous = _read_marker()
    since = previous[0] if previous and 0 <= now - previous[1] <= _TAIL_SECONDS else now
    if previous is None or since != previous[0]:
        logger.info("Cron dispatch held: a one-shot that falls due now fires once when it resumes")
    _write_marker(_MARKER, f"{since} {now}\n", ".held_")


def hold_covers(due_at: float, now: float) -> bool:
    """Whether a past-grace one-shot due at ``due_at`` fell due while dispatch was held (or in the
    restart gap right after), the hold is recent, and the job is still young enough to fire."""
    from cron.jobs import ONESHOT_GRACE_SECONDS

    marker = _read_marker()
    if marker is None:
        return False
    since, last = marker
    return (now - last <= _TAIL_SECONDS
            and since - ONESHOT_GRACE_SECONDS <= due_at <= last + _TAIL_SECONDS
            and now - due_at <= HELD_ONESHOT_MAX_LATE_SECONDS)


def end_hold_window(now: float) -> None:
    """After a due scan's save: once no ticker has stamped the marker for ``_RELEASE_SECONDS``, the
    hold is over and a scan that could dispatch has claimed or fired what it covered, so normal
    grace applies again. A marker that is still being stamped stays."""
    marker = _read_marker()
    if marker is not None and now - marker[1] < _RELEASE_SECONDS:
        return
    path = _marker_path()
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not clear %s: %s", path, exc)


def late_claim_fields(until: Optional[float]) -> Dict[str, Any]:
    """Extra ``run_claim`` fields for a one-shot dispatched past its grace: ``outage`` keeps it
    covered if this dispatch fails, and a hold-admitted job's ``until`` ends that cover."""
    return {"outage": True} if until is None else {"outage": True, "until": until}


def retry_claim(claim: Any) -> Optional[Dict[str, Any]]:
    """What ``clear_run_claim`` keeps of a failed dispatch's claim: a late one-shot's outage marker
    (no ``at``: stale, so it re-dispatches) with its ``until``; nothing for an on-time one."""
    if not (isinstance(claim, dict) and claim.get("outage")):
        return None
    return {key: claim[key] for key in ("outage", "until") if key in claim}


def unexpired_claim(job: Dict[str, Any], now: float) -> Any:
    """The job's ``run_claim``, or None once a hold-admitted claim is past its ``until``: from then
    on the job is a miss like any other."""
    claim = job.get("run_claim")
    until = claim.get("until") if isinstance(claim, dict) else None
    return None if isinstance(until, (int, float)) and now > until else claim


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
