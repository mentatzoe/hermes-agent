"""Occurrence-bound one-shot restart admission, separate from ownership leases.

Only lifecycle-recorded, clean restart windows authorize late dispatch, at most
2h after the due instant. Retries keep the same admission; edits cannot reuse it.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("cron.jobs")
MAX_LATE_SECONDS = 7200
_NOTICE_FIELDS = ("id", "name", "deliver", "origin", "attach_to_session")


def admission_for(job: dict, due: str) -> dict | None:
    """Match an exact occurrence, never an owner or a mutable schedule definition."""
    admission = job.get("restart_admission")
    if (isinstance(admission, dict) and admission.get("job_id") == job.get("id")
            and admission.get("due_at") == due):
        return admission
    return None


def admit_restart(d) -> bool:
    """Called under the due scan's jobs lock, including inside ordinary grace."""
    from cron.jobs import ONESHOT_GRACE_SECONDS
    from gateway.lifecycle_cron import valid_window
    from gateway import lifecycle_ledger as ledger
    from hermes_constants import get_hermes_home

    if d.job.get("last_run_at") or d.job.get("fire_claim") or d.job.get("manual_run_at") == d.next_run:
        return False
    due, now = d.next_run_dt.timestamp(), d.scan.now.timestamp()
    if not 0 <= now - due <= MAX_LATE_SECONDS:
        return False
    admission = admission_for(d.job, d.next_run)
    if admission is None:
        record = ledger._read_json(ledger.get_lifecycle_sentinel_path()) or {}
        home = str(get_hermes_home().resolve())
        retained = record.get("restart_windows")
        windows = retained if isinstance(retained, list) else []
        covering = next((w for w in windows
                         if valid_window(w) and home in w["homes"]
                         and w["start"] - ONESHOT_GRACE_SECONDS <= due <= w["end"] <= now), None)
        if covering is None:
            return False
        admission = {"job_id": d.job["id"], "due_at": d.next_run,
                     "admitted_at": d.scan.now.isoformat(), "window": dict(covering)}
        d.job["restart_admission"] = admission
        d.scan.persist(d.job["id"], restart_admission=admission)
    stamp = {"scheduled_at": d.next_run, "dispatched_at": d.scan.now.isoformat(),
             "lateness_seconds": round(now - due, 1), "kind": "catch_up", "reason": "gateway_restart"}
    d.job["last_dispatch"] = stamp
    d.scan.persist(d.job["id"], last_dispatch=stamp)
    return True


def oneshot_run_claim(d) -> dict:
    """Lease only owns dispatch; actual store outages keep their pre-existing exemption."""
    from cron import store_health
    from cron.jobs import ONESHOT_GRACE_SECONDS, _current_cron_store, _machine_id

    claim = {"at": d.scan.now.isoformat(), "by": _machine_id()}
    late = d.scan.now.timestamp() - d.next_run_dt.timestamp() > ONESHOT_GRACE_SECONDS
    outage = store_health.outage_covers(
        _current_cron_store().cron_dir, d.next_run_dt.timestamp(), ONESHOT_GRACE_SECONDS)
    prior = d.job.get("run_claim")
    if late and (outage or (isinstance(prior, dict) and prior.get("outage"))
                 or not admission_for(d.job, d.next_run)):
        claim["outage"] = True
    return claim


def restart_dispatch_expired(job: dict, now, *, manual=False) -> bool:
    """A queued scheduled worker must still satisfy the cap when it really starts."""
    from cron.jobs import _parse_aware

    if (manual or job.get("schedule", {}).get("kind") != "once"
            or (job.get("run_claim") or {}).get("outage")):
        return False
    admission = admission_for(job, job.get("next_run_at"))
    due = _parse_aware(admission["due_at"]) if admission else None
    return due is not None and now.timestamp() - due.timestamp() > MAX_LATE_SECONDS


def retirement_claim(job: dict, now) -> Any:
    """Never mistake a dead dispatch lease for admission after expiry or an edit."""
    from cron.jobs import _claim_is_live, _oneshot_run_claim_ttl_seconds

    claim = job.get("run_claim")
    if (job.get("restart_admission") and not (isinstance(claim, dict) and claim.get("outage"))
            and not _claim_is_live(claim, now, _oneshot_run_claim_ttl_seconds())):
        return None
    return claim


def late_run_note(job: dict) -> str:
    """No prompt mutation at rest; manual and edited occurrences never inherit this note."""
    from cron.occurrences import scheduled_instant

    admission = admission_for(job, job.get("next_run_at"))
    if (not admission or job.get("schedule", {}).get("kind") != "once"
            or not job.get("_scheduled_instant")
            or scheduled_instant(admission["due_at"]) != job["_scheduled_instant"]):
        return ""
    return (f"[Running late: this one-off was due at {admission['due_at']}, while the gateway "
            "was restarting and cron dispatch was stopped. It is running now. Do the task as "
            "written, adjusting time-dependent wording; say [SILENT] if it is no longer useful.]\n\n")


def queue_missed_notice(job: dict, next_run: str) -> None:
    """Durable, idempotent notice via the job target, or home for targetless/local jobs."""
    from cron import delivery_queue
    from cron.occurrences import scheduled_instant
    from cron.scheduler_delivery import _normalize_deliver_value

    name = job.get("name") or job["id"]
    target = {key: job[key] for key in _NOTICE_FIELDS if key in job}
    if _normalize_deliver_value(target.get("deliver")) == "local":
        target["deliver"] = "origin"
        target.pop("origin", None)  # deliver=origin with no origin resolves the home channel
    try:
        delivery_queue.enqueue(
            f"oneshot-missed:{job['id']}:{scheduled_instant(next_run) or next_run}", target,
            f"One-off '{name}' ({job['id']}) due {next_run} was skipped: too late to run safely "
            "and no eligible restart window remains.", for_failure=False)
    except Exception:
        logger.warning("Job '%s': could not queue its skipped-run notice", name, exc_info=True)


def write_missed_oneshot_diagnostic(job: dict, next_run: str) -> None:
    """Existing diagnostic, kept alongside the new user-visible notice."""
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
