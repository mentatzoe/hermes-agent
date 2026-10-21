"""Runtime status belongs to its live writer (mentatzoe/aleph-vault#235)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gateway import status


_OWNER = """
import json
import sys
from gateway import status

assert status.write_runtime_status(gateway_state="running", active_agents=1)
print(json.dumps(status.read_runtime_status()), flush=True)
for command in sys.stdin:
    if command.strip() == "stop":
        assert status.write_runtime_status(gateway_state="stopped")
        print("stopped", flush=True)
        break
    assert status.write_runtime_status(active_agents=2)
    print("updated", flush=True)
"""


@pytest.fixture
def owner(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Do not carry the previous test's snapshot, including a rejected write.
    monkeypatch.setattr(status, "_runtime_status_state", None)
    process = None

    def start():
        nonlocal process
        process = subprocess.Popen(
            [sys.executable, "-c", _OWNER],
            cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "HERMES_HOME": str(tmp_path)},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        record = json.loads(process.stdout.readline())
        assert record["pid"] == process.pid
        assert record["start_time"] == status._get_process_start_time(process.pid)
        return process, record

    yield start
    status.flush_runtime_status(timeout=2.0)
    if process is not None:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


def _submit(mode, **fields):
    if mode == "write":
        return status.write_runtime_status(wait_timeout=2.0, **fields)
    generation = status.publish_runtime_status(**fields)
    return status._get_runtime_status_writer().wait(generation, timeout=2.0)


@pytest.mark.parametrize("mode", ["write", "publish"])
@pytest.mark.parametrize("cached", [False, True])
def test_foreign_status_write_preserves_live_owner_and_all_file_bytes(
    tmp_path, owner, mode, cached
):
    if cached:
        # An earlier in-memory snapshot must not bypass a later disk owner.
        status._prepare_runtime_status_update(load_existing=False, active_agents=99)
    process, record = owner()
    path = tmp_path / "gateway_state.json"
    before, mtime = path.read_bytes(), path.stat().st_mtime_ns

    assert not _submit(
        mode,
        gateway_state="stopped",
        restart_requested=True,
        platform="discord",
        platform_state="fatal",
    )
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == mtime
    assert status.read_runtime_status() == record

    # The real owner's next heartbeat still persists after the rejected write.
    process.stdin.write("update\n")
    process.stdin.flush()
    assert process.stdout.readline().strip() == "updated"
    current = status.read_runtime_status()
    assert current["pid"] == process.pid
    assert current["start_time"] == record["start_time"]
    assert current["active_agents"] == 2
    assert current["gateway_state"] == "running"


@pytest.mark.parametrize("mode", ["write", "publish"])
@pytest.mark.parametrize("graceful", [False, True])
def test_new_writer_takes_over_only_after_previous_owner_exits(
    tmp_path, owner, mode, graceful
):
    process, record = owner()
    assert not _submit(mode, gateway_state="starting")
    if graceful:
        process.stdin.write("stop\n")
        process.stdin.flush()
        assert process.stdout.readline().strip() == "stopped"
        process.wait(timeout=10)
    else:
        process.kill()
        process.wait(timeout=10)

    assert _submit(mode, gateway_state="running", active_agents=3)
    current = status.read_runtime_status()
    assert current["pid"] == os.getpid()
    assert current["start_time"] == status._get_process_start_time(os.getpid())
    assert current["pid"] != record["pid"]
    assert current["gateway_state"] == "running"
    assert current["active_agents"] == 3
    assert _submit(mode, active_agents=4)
    assert status.read_runtime_status()["active_agents"] == 4


@pytest.mark.parametrize("mode", ["write", "publish"])
def test_recycled_pid_does_not_prevent_status_takeover(tmp_path, owner, mode):
    process, record = owner()
    # The process is still alive, but this record belongs to an earlier birth.
    record["start_time"] -= 10000
    (tmp_path / "gateway_state.json").write_text(json.dumps(record))

    assert process.poll() is None
    assert _submit(mode, gateway_state="running")
    assert status.read_runtime_status()["pid"] == os.getpid()


@pytest.mark.parametrize("mode", ["write", "publish"])
@pytest.mark.parametrize("previous", [None, "not JSON", '{"platforms": {}}'])
def test_initial_status_write_accepts_missing_or_unowned_record(
    tmp_path, monkeypatch, mode, previous
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "gateway_state.json"
    if previous is not None:
        path.write_text(previous)

    assert _submit(mode, gateway_state="starting")
    record = status.read_runtime_status()
    assert record["pid"] == os.getpid()
    assert record["gateway_state"] == "starting"


def test_queued_snapshot_rechecks_owner_when_writer_reaches_disk(
    tmp_path, owner, monkeypatch
):
    import threading

    entered = threading.Event()
    release = threading.Event()
    merge = status.merge_runtime_status_on_disk

    def delayed_merge(path, payload):
        entered.set()
        assert release.wait(timeout=10)
        return merge(path, payload)

    monkeypatch.setattr(status, "merge_runtime_status_on_disk", delayed_merge)
    generation = status.publish_runtime_status(gateway_state="starting")
    try:
        assert entered.wait(timeout=2)
        _, record = owner()
        path = tmp_path / "gateway_state.json"
        before = path.read_bytes()
    finally:
        release.set()

    assert not status._get_runtime_status_writer().wait(generation, timeout=2)
    assert path.read_bytes() == before
    assert status.read_runtime_status() == record
