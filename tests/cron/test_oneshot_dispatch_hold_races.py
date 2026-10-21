"""Regressions for the review of the dispatch-hold catch-up (aleph-vault#221, hermes-agent#18).

The first version kept the hold in process memory and gave a hold-admitted claim an unbounded
retry. Each test below is one finding of that review, driven through the real due scan; the
"other process" scans run in a separate Python process against the same store.

  1. a one-shot another process is running (fire claim) is never deleted by the hold catch-up
  2. a hold-admitted claim that fails to dispatch is retried only until due + 2 h
  3. a marker last stamped before real downtime admits nothing
  4. another process's scan during a live hold leaves the marker for the next gateway
  5. a hold that ended long ago does not lend its start to a new one
"""

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cron import delivery_queue, oneshot_lateness
from cron import jobs as cronjobs
from cron.jobs import claim_dispatch, claim_job_for_fire, clear_run_claim, get_due_jobs, load_jobs, save_jobs

REPO = Path(__file__).resolve().parents[2]


def _at(hh, mm, ss=0):
    return datetime(2026, 10, 20, hh, mm, ss, tzinfo=timezone.utc)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A temp HERMES_HOME whose cron store this process and a child process share, and a clock."""
    home = tmp_path / ".hermes"
    (home / "cron").mkdir(parents=True)
    clock = {"now": _at(9, 42)}
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(cronjobs, "CRON_DIR", home / "cron")
    monkeypatch.setattr(cronjobs, "JOBS_FILE", home / "cron" / "jobs.json")
    monkeypatch.setattr(cronjobs, "OUTPUT_DIR", home / "cron" / "output")
    monkeypatch.setattr(cronjobs, "_hermes_now", lambda: clock["now"])
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", home / "cron" / "deliveries.db")
    return {"home": home, "clock": clock}


def _oneshot(jid, due):
    return {
        "id": jid, "name": jid, "prompt": "Check whether the GP has replied.",
        "schedule": {"kind": "once", "run_at": due.isoformat()},
        "next_run_at": due.isoformat(), "last_run_at": None, "enabled": True,
        "state": "scheduled", "repeat": {"times": 1, "completed": 0},
        "deliver": "local", "run_claim": None, "fire_claim": None,
    }


def _hold(store, start, end):
    """A held ticker's stamps, once a minute from ``start`` to ``end`` inclusive."""
    t = start
    while t <= end:
        store["clock"]["now"] = t
        oneshot_lateness.note_dispatch_held()
        t += timedelta(minutes=1)


def _scan_in_another_process(store, now):
    """A due scan by a separate process (a CLI `hermes cron tick`, another gateway) at ``now``."""
    code = (
        "import sys; from datetime import datetime; import cron.jobs as j\n"
        "now = datetime.fromisoformat(sys.argv[1]); j._hermes_now = lambda: now\n"
        "print(len(j.get_due_jobs()))\n")
    env = {**os.environ, "HERMES_HOME": str(store["home"]), "HERMES_REAL_HOME": str(store["home"].parent),
           "HOME": str(store["home"].parent)}
    done = subprocess.run([sys.executable, "-c", code, now.isoformat()], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    assert done.returncode == 0, done.stderr
    return int(done.stdout.split()[-1])


def _ids():
    return [j["id"] for j in load_jobs()]


def test_1_a_running_oneshot_with_a_fire_claim_is_never_deleted(store):
    """Another process runs the held one-shot by hand (fire claim, dispatch budget spent). The due
    scan must keep its record, so the run's heartbeat and mark_job_run land."""
    save_jobs([_oneshot("gp", _at(10, 0))])
    _hold(store, _at(9, 42), _at(10, 12))
    store["clock"]["now"] = _at(10, 13)
    assert claim_job_for_fire("gp", manual=True)
    assert claim_dispatch("gp")

    store["clock"]["now"] = _at(10, 13, 24)
    assert get_due_jobs() == []
    assert _ids() == ["gp"]
    assert load_jobs()[0]["fire_claim"]


def test_2_a_failed_hold_admitted_dispatch_is_retried_only_inside_two_hours(store):
    save_jobs([_oneshot("gp", _at(10, 0))])
    _hold(store, _at(9, 42), _at(10, 12))
    store["clock"]["now"] = _at(10, 13, 24)
    assert [d["id"] for d in get_due_jobs()] == ["gp"]
    assert clear_run_claim("gp")  # dispatch failed (shutdown, executor error)
    kept = load_jobs()[0]["run_claim"]
    assert kept["outage"] is True and kept["until"] == _at(12, 0).timestamp()

    store["clock"]["now"] = _at(10, 20)  # inside the window: retried, keeping its deadline
    due = get_due_jobs()
    assert [d["id"] for d in due] == ["gp"] and due[0]["run_claim"]["until"] == kept["until"]
    clear_run_claim("gp")

    store["clock"]["now"] = _at(10, 0) + timedelta(days=1)  # past it: a miss, not a day-late run
    assert get_due_jobs() == []
    assert _ids() == []


def test_3_a_marker_from_before_real_downtime_admits_nothing(store):
    """Hold stamps stop at 09:55 (the process died), the gateway is down until 11:30: a job due at
    10:00 was missed by downtime, not skipped by a hold."""
    save_jobs([_oneshot("gp", _at(10, 0))])
    _hold(store, _at(9, 42), _at(9, 55))
    store["clock"]["now"] = _at(11, 30)
    assert get_due_jobs() == []
    assert _ids() == []


def test_4_another_process_scan_during_a_live_hold_keeps_the_marker(store):
    """The ticker holds through 10:11; a separate process scans at 10:11:50 while a 10:12 job is
    still future; the holding gateway stops before its next tick. The replacement's 10:15 scan
    must still fire the job once."""
    save_jobs([_oneshot("gp", _at(10, 12))])
    _hold(store, _at(9, 42), _at(10, 11))
    assert _scan_in_another_process(store, _at(10, 11, 50)) == 0
    assert (cronjobs.CRON_DIR / "dispatch_held").exists()

    store["clock"]["now"] = _at(10, 15)
    assert [d["id"] for d in get_due_jobs()] == ["gp"]
    assert not (cronjobs.CRON_DIR / "dispatch_held").exists()  # unstamped for 4 min: released


def test_5_a_hold_that_ended_long_ago_does_not_cover_a_new_one(store):
    """A profile's hold ends at 09:50 when it leaves the multiplex tick set; another process scans
    it at 10:15. A job due at 10:30 is missed with nothing held. A new hold from 11:00 must start at
    11:00, so the 11:10 scan retires the 10:30 job instead of admitting it."""
    from cron.scheduler_ownership import register_ticked_homes

    save_jobs([_oneshot("missed", _at(10, 30))])
    register_ticked_homes([store["home"]])
    _hold(store, _at(9, 42), _at(9, 50))
    register_ticked_homes([])
    assert _scan_in_another_process(store, _at(10, 15)) == 0

    register_ticked_homes([store["home"]])
    _hold(store, _at(11, 0), _at(11, 9))
    since = float((cronjobs.CRON_DIR / "dispatch_held").read_text(encoding="utf-8").split()[0])
    assert since == _at(11, 0).timestamp()
    store["clock"]["now"] = _at(11, 10)
    assert get_due_jobs() == []
    assert _ids() == []
    register_ticked_homes([])
