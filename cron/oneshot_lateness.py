"""Opt-in, stateless late-one-shot policy; normal leases still own execution.

No restart inference, occurrence admission stamps or persistent delivery consumer.
Skip delivery stays in the scanning ticker's profile and adapter scope.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import math

logger = logging.getLogger("cron.scheduler")
_skip_delivery: ContextVar[list | None] = ContextVar("oneshot_skip_delivery", default=None)
_LATE_PREFIX = "This one-off job is running late:"


def max_lateness_seconds():
    from cron.jobs import _cron_config_number
    value = _cron_config_number("oneshot_max_lateness_seconds", 0, float)
    return value if math.isfinite(value) and value > 0 else 0


def outage_exempt(d):
    from cron import jobs
    claim = d.job.get("run_claim")
    return jobs.store_health.outage_covers(
        jobs._current_cron_store().cron_dir, d.next_run_dt.timestamp(), jobs.ONESHOT_GRACE_SECONDS
    ) or (isinstance(claim, dict) and claim.get("outage") and not d.job.get("fire_claim"))


def retire_expired_oneshot(d):
    from cron import jobs
    late = jobs._elapsed_seconds(d.scan.now, d.next_run_dt)
    if late <= jobs.ONESHOT_GRACE_SECONDS or outage_exempt(d):
        return False
    automatic = d.job.get("manual_run_at") != d.next_run and not d.job.get("last_run_at")
    if automatic and not d.job.get("fire_claim") and late <= max_lateness_seconds():
        return False
    # Preserve upstream's in-flight guard: a stale lease is not proof that a run never started.
    if not (d.job.get("run_claim") or d.job.get("fire_claim")):
        jobs._write_missed_oneshot_diagnostic(d.job, d.next_run)
        d.scan.retire(d.job["id"])
        d.scan.on_saved.append(lambda: jobs.record_cron_missed(d.job))
        if automatic and jobs._cron_config_number("notify_missed_oneshots", False, lambda v: v is True):
            text = (f"Skipped one-off job '{_one_line(d.label)}': due {d.next_run}; "
                    f"{late:.0f} seconds late (outside the allowed lateness window).")
            d.scan.on_saved.append(lambda: _notice_after_save(d.job, text))
    return True


def _one_line(value):
    return " ".join(str(value).splitlines())


def _send_skip(job, text, adapters=None, loop=None):
    from cron import scheduler
    try:
        with scheduler._fire_secret_scope():
            error = scheduler._deliver_result(job, text, adapters=adapters, loop=loop, for_failure=True)
        if error:
            logger.error("One-off skip notice delivery failed for %s: %s", job["id"], error)
    except Exception:
        logger.exception("One-off skip notice delivery failed for %s", job["id"])


def _notice_after_save(job, text):
    pending = _skip_delivery.get()
    if pending is None:
        _send_skip(job, text)
    else:
        pending.append((job, text))


@contextmanager
def skip_delivery_scope(adapters=None, loop=None):
    """Collect only this scan's committed skips, then deliver outside the jobs-store lock.

    The caller is the existing per-profile ticker, not a launch-home queue consumer.
    """
    pending = []
    token = _skip_delivery.set(pending)
    try:
        yield
    finally:
        _skip_delivery.reset(token)
        for job, text in pending:
            _send_skip(job, text, adapters, loop)


def late_note(job, *, agent=False):
    """Read the claimed record's CURRENT due time, never an old dispatch snapshot."""
    from cron import jobs
    schedule = job.get("schedule")
    claim = job.get("run_claim")
    if (not isinstance(schedule, dict) or schedule.get("kind") != "once" or job.get("last_run_at")
            or job.get("_scheduled_instant") is None
            or job.get("manual_run_at") == job.get("next_run_at")
            or (isinstance(claim, dict) and claim.get("outage"))
            or not max_lateness_seconds()):
        return ""
    due = jobs._parse_aware(job.get("next_run_at"))
    if due is None:
        return ""
    late = jobs._elapsed_seconds(jobs._hermes_now(), due)
    if late <= jobs.ONESHOT_GRACE_SECONDS:
        return ""
    text = f"{_LATE_PREFIX} due {job['next_run_at']}; {late:.0f} seconds late."
    if agent:
        text += " Adjust time-bound wording to the actual time, or stay silent if it is no longer useful."
    return text + ("\n\n" if agent else "\n")


def script_failure_delivery(job, content, final_response):
    """Keep the script's start-time note when failure composition replaces its alert.

    Do not resurrect an acknowledged/suppressed failure, or recalculate after a long script.
    """
    if (job.get("no_agent") and content and isinstance(final_response, str)
            and final_response.startswith(_LATE_PREFIX)):
        return final_response.partition("\n")[0] + "\n" + content
    return content
