"""One-shots that fall due while cron dispatch is held (restart wait, drain) fire once afterwards.

Regression for aleph-vault#221. On 20 Oct 2026 a one-shot reminder was due at 10:00 while a
``/restart`` waited its full 30 minutes for running work (09:42-10:12). The ticker skips every tick
while ``can_dispatch`` is false, so at 10:13 the replacement gateway's first scan found the job 13
minutes late, past ``ONESHOT_GRACE_SECONDS``, and retired it unrun with only a diagnostic file.

These tests drive the real ticker loop and the real due scan in a temporary store:
  - due during a hold   -> fires once on the next process's first scan, stamped late
  - due long before     -> retired as before, plus a "did not run" notice in the delivery queue
  - a late run's prompt opens with when it was due
"""

import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest

from cron import delivery_queue, oneshot_lateness
from cron import jobs as cronjobs
from cron.jobs import get_due_jobs, load_jobs, save_jobs

WAIT_START = datetime(2026, 10, 20, 9, 42, tzinfo=timezone.utc)
DUE_IN_WAIT = datetime(2026, 10, 20, 10, 0, tzinfo=timezone.utc)
NEW_GATEWAY_FIRST_SCAN = datetime(2026, 10, 20, 10, 13, 24, tzinfo=timezone.utc)


@pytest.fixture()
def cron_store(tmp_path, monkeypatch):
    """Temp cron store, a settable clock, no hold left over from another test."""
    clock = {"now": WAIT_START}
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: clock["now"])
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", tmp_path / "cron" / "deliveries.db")
    return clock


def _oneshot(jid, due, **extra):
    job = {
        "id": jid, "name": jid, "prompt": "Check whether the GP has replied.",
        "schedule": {"kind": "once", "run_at": due.isoformat()},
        "next_run_at": due.isoformat(), "last_run_at": None, "enabled": True,
        "state": "scheduled", "repeat": {"times": 1, "completed": 0},
        "deliver": "discord:1496360045733085244", "run_claim": None, "fire_claim": None,
    }
    job.update(extra)
    return job


def _queued_notices():
    path = delivery_queue.queue_path()
    if not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        return conn.execute(
            "SELECT execution_id, content, for_failure FROM deliveries ORDER BY execution_id").fetchall()


def _hold_through_restart_wait(clock, monkeypatch, end=datetime(2026, 10, 20, 10, 12, tzinfo=timezone.utc)):
    """Run the real in-process ticker with dispatch held, one loop per minute of the restart wait,
    then stop it (the old gateway exits)."""
    from cron import scheduler
    from cron.scheduler_provider import InProcessCronScheduler

    ticks = []
    monkeypatch.setattr(scheduler, "tick", lambda *a, **k: ticks.append(1) or 0)
    minutes = int((end - clock["now"]).total_seconds() // 60) + 1
    held = {"calls": 0}
    stop = threading.Event()

    def can_dispatch():
        held["calls"] += 1
        if held["calls"] > minutes:
            stop.set()
        else:
            clock["now"] = WAIT_START + timedelta(minutes=held["calls"] - 1)
        return False

    thread = threading.Thread(
        target=InProcessCronScheduler().start, args=(stop,),
        kwargs={"interval": 0, "can_dispatch": can_dispatch}, daemon=True)
    thread.start()
    thread.join(timeout=30)
    assert not thread.is_alive(), "ticker did not stop"
    assert ticks == [], "no tick may run while dispatch is held"


def test_oneshot_due_during_restart_wait_fires_once_after_restart(cron_store, monkeypatch):
    save_jobs([_oneshot("gp-reply-check", DUE_IN_WAIT)])
    _hold_through_restart_wait(cron_store, monkeypatch)

    cron_store["now"] = NEW_GATEWAY_FIRST_SCAN
    due = get_due_jobs()

    assert [d["id"] for d in due] == ["gp-reply-check"]
    stamp = due[0]["last_dispatch"]
    assert stamp["scheduled_at"] == DUE_IN_WAIT.isoformat() and stamp["kind"] == "catch_up"
    assert stamp["lateness_seconds"] == pytest.approx(804.0)
    assert [j["id"] for j in load_jobs()] == ["gp-reply-check"]  # kept for its run to complete
    assert load_jobs()[0]["last_dispatch"] == stamp  # visible to `hermes cron list` / `status`
    assert _queued_notices() == []  # it runs, so nobody is told it didn't
    # Once: the next scan neither re-dispatches it nor treats a later miss as held.
    cron_store["now"] += timedelta(minutes=1)
    assert get_due_jobs() == []


def test_oneshot_due_well_before_the_hold_is_retired_with_a_notice(cron_store, monkeypatch):
    long_before = datetime(2026, 10, 20, 6, 30, tzinfo=timezone.utc)
    save_jobs([_oneshot("stale", long_before), _oneshot("local-only", long_before, deliver="local")])
    _hold_through_restart_wait(cron_store, monkeypatch)

    cron_store["now"] = NEW_GATEWAY_FIRST_SCAN
    assert get_due_jobs() == []

    assert load_jobs() == []
    assert list((cronjobs.OUTPUT_DIR / "stale").glob("*.md")), "the diagnostic file is still written"
    notices = _queued_notices()
    assert [n[0] for n in notices] == ["oneshot-missed:stale:2026-10-20T06:30:00+00:00"]
    content, for_failure = notices[0][1], notices[0][2]
    assert for_failure == 1 and "'stale' did not run" in content and "06:30" in content
    # A rescan after a failed save cannot queue the same miss twice (keyed by job and instant).
    oneshot_lateness.queue_missed_notice(_oneshot("stale", long_before), long_before.isoformat())
    assert len(_queued_notices()) == 1


def test_without_a_hold_a_late_oneshot_is_still_retired(cron_store):
    """A gateway that was simply down (no hold recorded) keeps the #93526 rule: no late fire."""
    save_jobs([_oneshot("down", DUE_IN_WAIT)])
    cron_store["now"] = NEW_GATEWAY_FIRST_SCAN
    assert get_due_jobs() == []
    assert load_jobs() == []
    assert [n[0] for n in _queued_notices()] == ["oneshot-missed:down:2026-10-20T10:00:00+00:00"]


def test_a_held_oneshot_more_than_two_hours_late_is_not_fired(cron_store, monkeypatch):
    """The hold is still live (stamped a minute ago); only the 2 h ceiling stops the fire."""
    save_jobs([_oneshot("very-late", DUE_IN_WAIT)])
    late = DUE_IN_WAIT + timedelta(seconds=oneshot_lateness.HELD_ONESHOT_MAX_LATE_SECONDS + 60)
    _hold_through_restart_wait(cron_store, monkeypatch, end=late - timedelta(minutes=1))
    cron_store["now"] = late
    assert get_due_jobs() == []
    assert load_jobs() == []


def test_hold_window_closes_once_the_marker_goes_unstamped(cron_store, monkeypatch):
    """A scan removes the marker once it can no longer cover a job (15 min unstamped), so a one-shot
    missed later while nothing is held is retired as before."""
    save_jobs([])
    _hold_through_restart_wait(cron_store, monkeypatch)
    cron_store["now"] = NEW_GATEWAY_FIRST_SCAN
    get_due_jobs()
    assert (cronjobs.CRON_DIR / "dispatch_held").exists()
    cron_store["now"] = datetime(2026, 10, 20, 10, 27, 1, tzinfo=timezone.utc)
    get_due_jobs()
    assert not (cronjobs.CRON_DIR / "dispatch_held").exists()

    missed_later = datetime(2026, 10, 20, 10, 40, tzinfo=timezone.utc)
    save_jobs([_oneshot("later", missed_later)])
    cron_store["now"] = missed_later + timedelta(minutes=10)
    assert get_due_jobs() == []
    assert load_jobs() == []


def test_a_marker_still_being_stamped_is_not_closed_by_a_scan(cron_store):
    """While a ticker keeps stamping the marker (a CLI `cron tick` scan can run meanwhile), it
    survives, so the post-restart scan still sees the hold."""
    oneshot_lateness.note_dispatch_held()
    save_jobs([])
    cron_store["now"] += timedelta(minutes=8)  # a backed-off ticker's gap between stamps
    get_due_jobs()
    assert (cronjobs.CRON_DIR / "dispatch_held").exists()
    cron_store["now"] += timedelta(minutes=8)
    get_due_jobs()
    assert not (cronjobs.CRON_DIR / "dispatch_held").exists()


def test_late_oneshot_prompt_opens_with_when_it_was_due(cron_store, monkeypatch):
    from cron.scheduler_prompt import _build_job_prompt

    save_jobs([_oneshot("gp-reply-check", DUE_IN_WAIT)])
    _hold_through_restart_wait(cron_store, monkeypatch)
    cron_store["now"] = NEW_GATEWAY_FIRST_SCAN
    late_job = get_due_jobs()[0]

    prompt = _build_job_prompt(late_job)
    assert prompt.startswith("[Running late: this one-off job was due at 10:00 on Tue 20 Oct")
    assert "Check whether the GP has replied." in prompt
    # An on-time run's prompt carries no note.
    on_time = _oneshot("on-time", NEW_GATEWAY_FIRST_SCAN + timedelta(minutes=30))
    save_jobs([on_time])
    cron_store["now"] = NEW_GATEWAY_FIRST_SCAN + timedelta(minutes=30)
    assert not _build_job_prompt(get_due_jobs()[0]).startswith("[Running late")
    # A hand-edited record with a string schedule still builds (no note, no error).
    assert oneshot_lateness.late_run_note({"id": "x", "schedule": "*/5 * * * *"}) == ""


def test_hold_marker_survives_a_ticker_stop_mid_hold(cron_store, monkeypatch):
    """The stopping gateway's ticker may be stopped while held; the marker still records the
    hold's start, and a fresh process that holds again keeps that start (one continuous gap)."""
    oneshot_lateness.note_dispatch_held()
    first = (cronjobs.CRON_DIR / "dispatch_held").read_text(encoding="utf-8").split()[0]
    cron_store["now"] += timedelta(minutes=3)
    oneshot_lateness.note_dispatch_held()
    assert (cronjobs.CRON_DIR / "dispatch_held").read_text(encoding="utf-8").split()[0] == first
