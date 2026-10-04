"""Renewal must never use the jobs lock's degraded, process-local mode."""

import subprocess
import sys

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def test_renewal_fails_closed_during_other_process_terminal_write(temp_home, monkeypatch):
    from cron import jobs

    job = jobs.create_job(prompt="x", schedule="every 5m", name="renewal-overlap")
    claimed = jobs.claim_job_for_fire(job["id"], return_job=True)
    assert isinstance(claimed, dict)
    owner = claimed["fire_claim"]["by"]
    child = subprocess.Popen(
        [sys.executable, "-c", """
import sys
from cron import jobs
with jobs._jobs_lock():
    print('locked', flush=True)
    sys.stdin.readline()
    jobs.mark_job_run(sys.argv[1], True, expected_fire_owner=sys.argv[2])
""", job["id"], owner],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "locked"
        before = jobs.JOBS_FILE.read_bytes()
        monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.05)
        with pytest.raises(TimeoutError, match="renew"):
            jobs.heartbeat_fire_claim(job["id"], expected_owner=owner)
        assert jobs.JOBS_FILE.read_bytes() == before
        _, stderr = child.communicate("finish\n", timeout=5)
        assert child.returncode == 0, stderr
    finally:
        if child.poll() is None:
            child.terminate()
            child.communicate(timeout=5)

    persisted = jobs.get_job(job["id"])
    assert persisted is not None
    assert persisted["last_status"] == "ok"
    assert not persisted.get("fire_claim")
    assert not jobs.heartbeat_fire_claim(job["id"], expected_owner=owner)
    recovered = jobs.claim_job_for_fire(job["id"], return_job=True)
    assert isinstance(recovered, dict)
    assert jobs.heartbeat_fire_claim(job["id"], expected_owner=recovered["fire_claim"]["by"])


def test_renewal_refuses_nested_degraded_lock(temp_home, monkeypatch):
    from cron import jobs

    job = jobs.create_job(prompt="x", schedule="every 5m", name="degraded-renewal")
    claimed = jobs.claim_job_for_fire(job["id"], return_job=True)
    assert isinstance(claimed, dict)
    before = jobs.JOBS_FILE.read_bytes()
    monkeypatch.setattr(jobs, "fcntl", None)
    monkeypatch.setattr(jobs, "msvcrt", None)
    with jobs._jobs_lock():
        with pytest.raises(TimeoutError, match="renew"):
            jobs.heartbeat_fire_claim(job["id"], expected_owner=claimed["fire_claim"]["by"])
    assert jobs.JOBS_FILE.read_bytes() == before
