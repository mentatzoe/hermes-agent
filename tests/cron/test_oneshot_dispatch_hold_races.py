"""Regressions for the review of the dispatch-hold catch-up (aleph-vault#221, hermes-agent#18).

The first version kept the hold in process memory and gave a hold-admitted claim an unbounded
retry. Each test below is one finding of that review, driven through the real due scan; the
"other process" scans run in a separate Python process against the same store.

  1. a one-shot another process is running (fire claim) is never deleted by the hold catch-up
  2. a hold-admitted claim that fails to dispatch is retried only until due + 2 h
  3. a marker last stamped before real downtime admits nothing
  4. another process's scan during a live hold leaves the marker for the next gateway
  5. a hold that ended long ago does not lend its start to a new one

and of the second review (da2fb56c):

  6. a scan's delete and a ticker's stamp are serialized by the jobs lock
  7. a backed-off ticker's 8-minute gap between stamps does not end its hold
  8. a queued scheduled run past its claim's ``until`` is not started
  9. a marker stamped in the future (clock skew) admits nothing
"""

import os
import subprocess
import sys
import threading
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


def test_6_a_stamp_waits_for_a_scan_that_holds_the_jobs_lock(store):
    """end_hold_window reads, then unlinks. A ticker's stamp must not land between the two, or the
    delete erases a hold that has just been renewed."""
    _hold(store, _at(9, 0), _at(9, 1))  # an old hold: the next scan drops its marker
    store["clock"]["now"] = _at(10, 0)
    stamped = threading.Event()

    def stamp():
        oneshot_lateness.note_dispatch_held()
        stamped.set()

    with cronjobs._jobs_lock():
        ticker = threading.Thread(target=stamp, daemon=True)
        ticker.start()
        assert not stamped.wait(0.5), "the stamp ran inside a scan's critical section"
        oneshot_lateness.end_hold_window(_at(10, 0).timestamp())
    assert stamped.wait(10)
    since, last = (cronjobs.CRON_DIR / "dispatch_held").read_text(encoding="utf-8").split()
    assert float(since) == float(last) == _at(10, 0).timestamp()


def test_7_a_backed_off_ticker_keeps_its_hold_between_stamps(store):
    """EMFILE backoff stretches the tick to 8 min: stamps at 10:00 and 10:08. A scan by another
    process at 10:03 must leave the marker, and the 10:09 scan fires the 10:04 reminder."""
    save_jobs([_oneshot("gp", _at(10, 4))])
    _hold(store, _at(10, 0), _at(10, 0))
    assert _scan_in_another_process(store, _at(10, 3)) == 0
    _hold(store, _at(10, 8), _at(10, 8))
    since = float((cronjobs.CRON_DIR / "dispatch_held").read_text(encoding="utf-8").split()[0])
    assert since == _at(10, 0).timestamp()
    store["clock"]["now"] = _at(10, 9)
    assert [d["id"] for d in get_due_jobs()] == ["gp"]


def test_8_a_queued_run_past_its_deadline_is_not_started(store):
    """Admitted at 10:13, the run waits behind busy workers; its worker reaches the fire claim at
    12:10, after due + 2 h. It must not start; a hand-run still may."""
    save_jobs([_oneshot("gp", _at(10, 0))])
    _hold(store, _at(9, 42), _at(10, 12))
    store["clock"]["now"] = _at(10, 13, 24)
    assert [d["id"] for d in get_due_jobs()] == ["gp"]

    store["clock"]["now"] = _at(12, 10)
    assert not claim_job_for_fire("gp")
    assert load_jobs()[0]["fire_claim"] is None
    assert claim_job_for_fire("gp", manual=True)


def test_9_a_marker_stamped_in_the_future_admits_nothing(store):
    """A writer with a fast clock (or a clock corrected backwards) left last=11:00. At 10:30 that
    is no evidence of a live hold: the 10:00 reminder is a miss and the marker goes."""
    save_jobs([_oneshot("gp", _at(10, 0))])
    (cronjobs.CRON_DIR / "dispatch_held").write_text(
        f"{_at(9, 42).timestamp()} {_at(11, 0).timestamp()}\n", encoding="utf-8")
    store["clock"]["now"] = _at(10, 30)
    assert get_due_jobs() == []
    assert _ids() == []
    assert not (cronjobs.CRON_DIR / "dispatch_held").exists()
