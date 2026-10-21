"""Restart-window catch-up and visible retirement (aleph-vault#221)."""
import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from cron import jobs, delivery_queue, scheduler
from gateway import lifecycle_ledger as ledger
from hermes_constants import get_hermes_home


@pytest.fixture
def clock(monkeypatch):
    current = [datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: current[0])
    monkeypatch.setattr(ledger.time, "time", lambda: current[0].timestamp())
    monkeypatch.setattr(ledger, "_process_hermes_home", get_hermes_home)
    monkeypatch.setattr(ledger, "detect_unclean_exit", lambda home=None: None)
    return current


def at(clock, hour, minute=0):
    clock[0] = clock[0].replace(hour=hour, minute=minute, second=0)


def create(clock, hour=10, minute=0, **kw):
    due = clock[0].replace(hour=hour, minute=minute)
    return jobs.create_job(prompt="Check the GP reply", schedule=due.isoformat(), **kw)


def window(clock, start=(9, 42), end=(10, 13), clean=True):
    at(clock, *start)
    ledger.record_startup()
    path = ledger.get_lifecycle_sentinel_path()
    record = json.loads(path.read_text())
    record.update(dispatch_stopped_at=clock[0].timestamp(), dispatch_stopped_pid=os.getpid(),
                  dispatch_stopped_homes=[str(get_hermes_home().resolve())])
    path.write_text(json.dumps(record))
    if clean:
        ledger.mark_exited(75)
    at(clock, *end)
    ledger.record_startup()


def notice(job):
    from cron.occurrences import scheduled_instant
    key = f"oneshot-missed:{job['id']}:{scheduled_instant(job['next_run_at'])}"
    row = delivery_queue.get_status(key)
    assert row is not None
    assert row["status"] == "pending"
    assert "skipped" in row["content"].lower()
    return json.loads(row["job_json"])


def test_due_in_restart_runs_once_with_late_prompt(clock):
    job = create(clock)
    window(clock)
    due = jobs.get_due_jobs()
    assert [j["id"] for j in due] == [job["id"]]
    claimed = jobs.claim_job_for_fire(job["id"], return_job=True)
    from cron.scheduler_prompt import _build_job_prompt
    prompt = _build_job_prompt(claimed)
    assert "Running late" in prompt and "restart" in prompt
    jobs.mark_job_run(job["id"], success=True)
    assert jobs.get_due_jobs() == []
    assert jobs.get_job(job["id"])["state"] == "completed"


def test_before_restart_is_skipped_to_own_target(clock):
    job = create(clock, 9, 20, deliver="discord:123")
    window(clock)
    assert jobs.get_due_jobs() == []
    assert jobs.get_job(job["id"]) is None
    assert notice(job)["deliver"] == "discord:123"
    assert list((jobs.get_cron_output_dir() / job["id"]).glob("*.md"))


def test_unclean_exit_does_not_authorize_catchup(clock):
    job = create(clock)
    window(clock, clean=False)
    assert jobs.get_due_jobs() == []
    assert notice(job)["deliver"] == "origin"


def test_schedule_edit_cannot_reuse_old_admission(clock):
    job = create(clock)
    window(clock)
    assert jobs.get_due_jobs()
    # Edit before dispatch starts, then lose the dispatch. Neither a deadline nor an
    # owner-shaped admission may survive as authorization for the new occurrence.
    at(clock, 10, 14)
    jobs.update_job(job["id"], {"schedule": clock[0].replace(minute=16).isoformat()})
    at(clock, 10, 50)
    assert jobs.get_due_jobs() == []
    changed = dict(job, next_run_at=clock[0].replace(minute=16).isoformat())
    notice(changed)
    assert jobs.get_job(job["id"]) is None


def test_consecutive_restarts_preserve_both_windows(clock):
    first = create(clock, 10, 11)
    second = create(clock, 10, 18)
    window(clock)
    assert [j["id"] for j in jobs.get_due_jobs()] == [first["id"]]
    jobs.clear_run_claim(first["id"])
    window(clock, (10, 14), (10, 21))
    due = jobs.get_due_jobs()
    assert {j["id"] for j in due} == {first["id"], second["id"]}
    sentinel = json.loads(ledger.get_lifecycle_sentinel_path().read_text())
    assert len(sentinel["restart_windows"]) == 2


def test_failed_dispatch_retries_same_admission_not_outage(clock):
    job = create(clock)
    window(clock)
    due = jobs.get_due_jobs()[0]
    pool = Mock()
    pool.submit.side_effect = RuntimeError("executor unavailable")
    assert scheduler._submit_with_guard(due, pool, lambda _: True) is None
    admission = jobs.get_job(job["id"])["restart_admission"]
    assert not (jobs.get_job(job["id"]).get("run_claim") or {}).get("outage")
    at(clock, 10, 20)
    retried = jobs.get_due_jobs()[0]
    assert retried["restart_admission"] == admission
    assert jobs.get_due_jobs() == []  # a live run lease still fences the retry


def test_expired_admission_retires_after_failed_dispatch(clock):
    job = create(clock)
    window(clock)
    assert jobs.get_due_jobs()
    jobs.clear_run_claim(job["id"])
    at(clock, 12, 1)
    assert jobs.get_due_jobs() == []
    notice(job)


def test_queued_worker_cannot_start_after_cap_but_manual_can(clock):
    job = create(clock)
    window(clock)
    assert jobs.get_due_jobs()
    at(clock, 12, 1)
    assert not jobs.claim_job_for_fire(job["id"])
    assert jobs.claim_job_for_fire(job["id"], manual=True)


def test_once_admitted_is_not_readmitted_when_history_rotates(clock):
    job = create(clock)
    window(clock)
    assert jobs.get_due_jobs()
    admission = jobs.get_job(job["id"])["restart_admission"]
    jobs.clear_run_claim(job["id"])
    # Existing admission is occurrence-bound, independent of retained history.
    path = ledger.get_lifecycle_sentinel_path()
    record = json.loads(path.read_text())
    record["restart_windows"] = []
    path.write_text(json.dumps(record))
    at(clock, 10, 20)
    assert jobs.get_due_jobs()[0]["restart_admission"] == admission


def test_live_run_keeps_completion_record_beyond_cap(clock):
    job = create(clock)
    window(clock)
    assert jobs.get_due_jobs()
    assert jobs.claim_job_for_fire(job["id"])
    at(clock, 12, 1)
    assert jobs.get_due_jobs() == []
    assert jobs.get_job(job["id"]) is not None


def test_recurring_and_store_outage_unchanged(clock, monkeypatch):
    recurring = jobs.create_job(prompt="recurring", schedule="every 1h")
    one = create(clock)
    window(clock, clean=False)
    monkeypatch.setattr(jobs.store_health, "outage_covers", lambda *args: True)
    due = jobs.get_due_jobs()
    assert {j["id"] for j in due} == {recurring["id"], one["id"]}
    assert not jobs.get_job(recurring["id"]).get("restart_admission")
    assert jobs.get_job(one["id"])["run_claim"]["outage"]


def test_initial_admission_cannot_fire_hours_late(clock):
    job = create(clock)
    window(clock)
    at(clock, 12, 1)
    assert jobs.get_due_jobs() == []
    notice(job)


def test_dead_dispatch_lease_does_not_hide_expired_admission(clock):
    job = create(clock)
    window(clock)
    assert jobs.get_due_jobs()
    at(clock, 12, 1)
    assert jobs.get_due_jobs() == []
    notice(job)


def test_later_store_outage_retains_its_own_exemption(clock, monkeypatch):
    job = create(clock)
    window(clock)
    assert jobs.get_due_jobs()
    jobs.clear_run_claim(job["id"])
    at(clock, 12, 1)
    monkeypatch.setattr(jobs.store_health, "outage_covers", lambda *args: True)
    assert jobs.get_due_jobs()
    assert jobs.get_job(job["id"])["run_claim"]["outage"]
    assert jobs.claim_job_for_fire(job["id"])


def test_local_skip_resolves_to_home_not_creation_origin(clock, monkeypatch):
    from cron import scheduler_delivery as delivery
    job = create(clock, deliver="local", origin={"platform": "discord", "chat_id": "elsewhere"})
    at(clock, 10, 13)
    assert jobs.get_due_jobs() == []
    target = notice(job)
    monkeypatch.setattr(delivery, "_iter_home_target_platforms", lambda: ["discord"])
    monkeypatch.setattr(delivery, "_get_home_target_chat_id", lambda p: "home-channel")
    monkeypatch.setattr(delivery, "_get_home_target_thread_id", lambda p: None)
    assert delivery._resolve_delivery_targets(target)[0]["chat_id"] == "home-channel"


def test_no_timestamp_no_window_and_local_still_notifies_home(clock):
    job = create(clock, deliver="local")
    ledger.record_startup()
    ledger.mark_exited(0)
    at(clock, 10, 13)
    ledger.record_startup()
    assert jobs.get_due_jobs() == []
    assert notice(job)["deliver"] == "origin"
