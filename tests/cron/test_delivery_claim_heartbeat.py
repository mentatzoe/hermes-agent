"""Real script/store/ledger coverage for renewal during slow delivery.

Only the external delivery boundary is replaced: no bot/model calls or live
monthly job. The scheduler, heartbeat, fences, script subprocess, output and
terminal ledger writes are real and isolated under the test's HERMES_HOME.
"""
import threading

import pytest


@pytest.mark.parametrize("script_exit,delivery_error,cancel", [
    (0, None, False),
    (1, None, False),
    (0, "consumer unavailable", False),
    (0, None, True),
])
def test_terminal_result_after_delivery_heartbeat(
    tmp_path, monkeypatch, script_exit, delivery_error, cancel,
):
    from cron import executions, jobs, scheduler

    home = tmp_path / "home"
    scripts = home / "scripts"
    scripts.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", home / "cron" / "executions.db")
    (scripts / "probe.py").write_text(
        f"print('admission probe')\nraise SystemExit({script_exit})\n"
    )
    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)
    delivery_started = threading.Event()
    heartbeat_finished = threading.Event()
    cancel_event = threading.Event()
    renewals = []
    delivered = []
    real_heartbeat = jobs.heartbeat_fire_claim

    def observed_heartbeat(job_id, *, expected_owner):
        renewed = real_heartbeat(job_id, expected_owner=expected_owner)
        if (threading.current_thread().name == "cron-fire-claim-heartbeat"
                and delivery_started.is_set()):
            renewals.append(renewed)
            heartbeat_finished.set()
        return renewed

    def slow_delivery(job, content, **kwargs):
        delivered.append(content)
        delivery_started.set()
        assert heartbeat_finished.wait(3), "renewal did not return during delivery"
        delivery_started.clear()
        if cancel:
            cancel_event.set()
        return delivery_error

    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", observed_heartbeat)
    monkeypatch.setattr(scheduler, "_deliver_result", slow_delivery)
    with jobs.use_cron_store(home):
        job = jobs.create_job(
            prompt="", schedule="0 9 1 * *", script="probe.py",
            no_agent=True, deliver="bot-chat", name="isolated-admission-probe",
        )
        claimed = jobs.claim_job_for_fire(job["id"], return_job=True)
        assert isinstance(claimed, dict)
        next_run = claimed["next_run_at"]
        assert scheduler.run_one_job(claimed, cancel_event=cancel_event) is True
        stored = jobs.get_job(job["id"])
        ledger = executions.latest_execution(job["id"])
        assert stored is not None and ledger is not None
        assert renewals and all(renewals), "own delivery fence starved lease renewal"
        assert len(delivered) == 1
        assert stored["fire_claim"] is None
        assert stored["next_run_at"] == next_run
        assert list((home / "cron" / "output" / job["id"]).glob("*.md"))
        if cancel:
            assert ledger["status"] == "failed"
            assert "Interrupted by shutdown" in ledger["error"]
        elif script_exit:
            assert ledger["status"] == "failed"
            assert "Interrupted by shutdown" not in ledger["error"]
            assert "admission probe" in ledger["error"]
            assert "failed" in delivered[0].lower()
        else:
            assert ledger["status"] == "completed"
            assert stored["last_status"] == ("delivery_failed" if delivery_error else "ok")
            assert stored["last_delivery_error"] == delivery_error
            assert "admission probe" in delivered[0]

        # Same job remains usable after a recoverable failure; do not mint a
        # replacement. Renewal cannot resurrect the completed fire's claim.
        assert jobs.heartbeat_fire_claim(
            job["id"], expected_owner=claimed["fire_claim"]["by"],
        ) is False
        retry = jobs.claim_job_for_fire(job["id"], return_job=True)
        assert isinstance(retry, dict)
        cancel = False
        delivery_error = None
        cancel_event.clear()
        heartbeat_finished.clear()
        (scripts / "probe.py").write_text("print('recovered admission probe')\n")
        assert scheduler.run_one_job(retry, cancel_event=cancel_event) is True
        recovered = executions.latest_execution(job["id"])
        assert recovered is not None
        assert recovered["status"] == "completed"
        assert recovered["id"] != ledger["id"]
        assert len(delivered) == 2
        assert delivered[-1] == "recovered admission probe"
        assert len(jobs.load_jobs()) == 1
