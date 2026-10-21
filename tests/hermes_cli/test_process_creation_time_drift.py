"""Creation-time consumers agree across macOS clock steps (aleph-vault #233)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from hermes_cli import active_sessions, process_identity, update_lock
from hermes_cli.local_runtime import recovery, supervisor
from hermes_cli.observability import shared_metrics_startup
from tools.bot_desktop import runtime as desktop


@pytest.mark.platforms("macos")
@pytest.mark.parametrize("shift", [3, -3])
def test_float_producers_ignore_psutil_boot_time_adjustment(monkeypatch, shift):
    import psutil._psosx as osx

    pid = os.getpid()
    raw = psutil.Process(pid)._proc.create_time(monotonic=True)
    monkeypatch.setattr(osx, "INIT_BOOT_TIME", osx.boot_time() + shift)
    assert psutil.Process(pid).create_time() != raw  # reproduce the public reading's shift
    for read in (process_identity._process_create_time, active_sessions._process_start_time,
                 update_lock.process_create_time, desktop._create_time):
        assert read(pid) == raw, read.__module__
    assert shared_metrics_startup.process_started_at() == raw
    assert process_identity._pid_alive_matches(pid, raw, strict=True) is True
    assert process_identity._pid_alive_matches(pid, raw + 10, strict=True) is False
    assert process_identity._pid_alive_matches(pid, raw + 10) is False
    monkeypatch.setattr(active_sessions, "_OWN_START", None)
    assert active_sessions._pid_liveness(pid, raw) is True
    assert active_sessions._pid_liveness(pid, raw + 10) is False
    assert recovery._same_birth(psutil.Process(pid), {"create_time": raw}) is True
    assert recovery._owner_is_dead({"owner_pid": pid, "owner_create_time": raw}) is False


_WRITER = """
import json, os, sys
import psutil._psosx as osx
from hermes_cli import active_sessions, process_identity
osx.INIT_BOOT_TIME = osx.boot_time() + float(sys.argv[1])
process_identity._ledger_path = lambda: __import__('pathlib').Path(sys.argv[2])
assert process_identity.register_self('serve')
lease, refusal = active_sessions.try_acquire_active_session(
    session_id='shared', surface='cli', config={}, registry_home=sys.argv[3], track_liveness=True)
assert lease is not None and refusal is None
print('ready', flush=True)
sys.stdin.read()
lease.release()
"""


@pytest.mark.platforms("macos")
def test_shifted_writer_and_fresh_reader_keep_the_same_spawn_ledger_owner(tmp_path):
    registry = tmp_path / "registry"
    ledger = registry / "spawn-ledger.json"
    child = subprocess.Popen([sys.executable, "-c", _WRITER, "3", str(ledger), str(registry)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        row = json.loads(ledger.read_text())[0]
        assert process_identity._pid_alive_matches(row["pid"], row["create_time"], strict=True) is True
        assert process_identity.spawner_is_dead({"spawner_pid": child.pid,
                                                "spawner_create": row["create_time"]}) is False
        assert process_identity._pid_alive_matches(child.pid, row["create_time"] + 10, strict=True) is False
        lease, refusal = active_sessions.try_acquire_active_session(
            session_id="shared", surface="tui", config={}, registry_home=registry, track_liveness=True)
        assert lease is None
        assert refusal.reason == active_sessions.SESSION_NOT_OWNED
        command = [sys.executable, str(Path(__file__).resolve().parents[2] / "scripts/start_time_transition_check.py"),
                   "--home", str(registry), "--no-profiles", "--no-lock-dir"]
        checked = subprocess.run(command, capture_output=True, text=True, timeout=15)
        assert checked.returncode == 0, checked.stdout + checked.stderr
        assert "spawn-ledger.json" in checked.stdout and "active_sessions.json" in checked.stdout
        # A retained pre-switch stamp is detected without rewriting it or ending its live owner.
        original = ledger.read_text()
        old_row = dict(row, create_time=row["create_time"] + 3)
        ledger.write_text(json.dumps([old_row]))
        checked = subprocess.run(command, capture_output=True, text=True, timeout=15)
        assert checked.returncode == 1 and "NOT DRAINED" in checked.stdout
        assert json.loads(ledger.read_text()) == [old_row]
        ledger.write_text(original)
    finally:
        child.communicate(timeout=10)
    assert process_identity._pid_alive_matches(row["pid"], row["create_time"], strict=True) is False
    lease, refusal = active_sessions.try_acquire_active_session(
        session_id="shared", surface="tui", config={}, registry_home=registry, track_liveness=True)
    assert lease is not None and refusal is None
    lease.release()


@pytest.mark.platforms("macos")
def test_supervisor_float_state_survives_a_different_readers_clock_offset(tmp_path, monkeypatch):
    import psutil._psosx as osx

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    try:
        owner_birth = psutil.Process(os.getpid())._proc.create_time(monotonic=True)
        child_birth = psutil.Process(child.pid)._proc.create_time(monotonic=True)
        managed = supervisor.LlamaServerSupervisor(Path(sys.executable), tmp_path / "models", port=18434)
        managed.proc = child
        monkeypatch.setattr(osx, "INIT_BOOT_TIME", osx.boot_time() + 3)
        managed._write_state()
        state = recovery.read_state()
        assert state["create_time"] == child_birth
        assert state["owner_create_time"] == owner_birth
        # Older modern records have float births but no integer fingerprints.
        state.pop("start_time")
        state.pop("owner_start_time")
        monkeypatch.setattr(osx, "INIT_BOOT_TIME", osx.boot_time() - 3)
        assert recovery.recorded_process(state).pid == child.pid
        assert recovery._owner_is_dead(state) is False
        state["create_time"] += 10
        assert recovery.recorded_process(state) is None
    finally:
        child.communicate(timeout=10)
