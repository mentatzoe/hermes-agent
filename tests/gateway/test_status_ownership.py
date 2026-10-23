"""Runtime status belongs to its live writer (mentatzoe/aleph-vault#235)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gateway import status


pytestmark = pytest.mark.spawns_gateway_lookalike


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

    def start(script=_OWNER, before_ready=None, *, as_gateway=True):
        nonlocal process
        repo = Path(__file__).resolve().parents[2]
        if as_gateway:
            # A reaped stub exercises the real command-line identity reader, not a mock.
            entry = tmp_path / "hermes"
            entry.write_text(f"import sys\nsys.path.insert(0, {str(repo)!r})\n" + script)
            command = [sys.executable, str(entry), "gateway", "run"]
        else:
            command = [sys.executable, "-c", script]
        process = subprocess.Popen(
            command,
            cwd=repo,
            env={**os.environ, "HERMES_HOME": str(tmp_path)},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if before_ready is not None:
            before_ready(process)
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
def test_unrelated_live_writer_cannot_block_gateway_startup(tmp_path, owner, mode):
    process, record = owner(as_gateway=False)
    assert process.poll() is None
    assert not status._record_matches_live_gateway_pid(record, process.pid)

    assert _submit(mode, gateway_state="starting")
    assert status.read_runtime_status()["pid"] == os.getpid()


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
    persist = status.persist_runtime_status

    def delayed_persist(path, payload, write_fn):
        entered.set()
        assert release.wait(timeout=10)
        return persist(path, payload, write_fn)

    monkeypatch.setattr(status, "persist_runtime_status", delayed_persist)
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


def test_status_validation_and_replacement_are_one_cross_process_transaction(
    tmp_path, owner
):
    paused_owner = """
from gateway import status
original = status._write_json_file

def pause_before_replace(path, payload):
    print("validated", flush=True)
    assert input() == "release"
    status._write_json_file = original
    original(path, payload)

status._write_json_file = pause_before_replace
""" + _OWNER
    generations = []

    def contend(process):
        assert process.stdout.readline().strip() == "validated"
        try:
            generation = status.publish_runtime_status(gateway_state="starting")
            generations.append(generation)
            writer = status._get_runtime_status_writer()
            # The first writer has validated but not replaced the file. A second
            # writer must stay blocked, rather than publishing its own identity.
            assert not writer.wait(generation, timeout=2.0)
            assert writer.settled(generation) is None
            assert not (tmp_path / "gateway_state.json").exists()
        finally:
            process.stdin.write("release\n")
            process.stdin.flush()

    process, record = owner(paused_owner, before_ready=contend)
    assert not status._get_runtime_status_writer().wait(generations[0], timeout=2.0)
    assert status.read_runtime_status() == record
    process.stdin.write("update\n")
    process.stdin.flush()
    assert process.stdout.readline().strip() == "updated"
    assert status.read_runtime_status()["active_agents"] == 2
