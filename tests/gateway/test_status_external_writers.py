"""External status edits must not replay a gateway identity (aleph-vault#235)."""
import json
import os
import threading
from pathlib import Path

import pytest

from gateway import status
from hermes_cli import gateway_migrate, service_manager


@pytest.mark.parametrize("editor", ["service", "migration"])
def test_external_status_edit_serializes_with_new_gateway(tmp_path, monkeypatch, editor):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "gateway_state.json"
    path.write_text(json.dumps({
        "pid": 99999999, "start_time": 1, "desired_state": "running",
        "served_profiles": ["default", "coder"], "platforms": {"coder:matrix": {}},
    }))
    entered, release = threading.Event(), threading.Event()
    read_text = Path.read_text
    errors = []

    def paused_read(self, *args, **kwargs):
        text = read_text(self, *args, **kwargs)
        if self == path and threading.current_thread() is thread:
            entered.set()
            assert release.wait(timeout=10)
        return text

    def edit():
        try:
            if editor == "service":
                service_manager._write_gateway_desired_state("gateway-default", "stopped")
            else:
                gateway_migrate._reconcile_standalone_runtime(tmp_path, {"coder"})
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(Path, "read_text", paused_read)
    thread = threading.Thread(target=edit)
    thread.start()
    generation = None
    try:
        assert entered.wait(timeout=5)
        generation = status.publish_runtime_status(gateway_state="starting")
        writer = status._get_runtime_status_writer()
        assert not writer.wait(generation, timeout=2)
        assert writer.settled(generation) is None
    finally:
        release.set()
        thread.join(timeout=10)
        if generation is not None:
            status._get_runtime_status_writer().wait(generation, timeout=5)
    assert not thread.is_alive()
    assert not errors
    current = status.read_runtime_status()
    assert current["pid"] == os.getpid()
    assert current["start_time"] == status._get_process_start_time(os.getpid())
    if editor == "service":
        assert current["desired_state"] == "stopped"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("editor", ["service", "migration"])
def test_new_coordination_lock_remains_readable_with_restrictive_umask(tmp_path, monkeypatch, editor):
    import stat

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    mask = os.umask(0o077)
    try:
        if editor == "service":
            service_manager._write_gateway_desired_state("gateway-default", "running")
        else:
            gateway_migrate._reconcile_standalone_runtime(tmp_path, set())
    finally:
        os.umask(mask)
    lock = tmp_path / "gateway_state.json.lock"
    assert stat.S_IMODE(lock.stat().st_mode) == 0o644


@pytest.mark.platforms("posix")
def test_read_only_coordination_file_allows_runtime_persistence(tmp_path, monkeypatch):
    # Models the service user's access to a privileged caller's 0644 lock.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    lock = tmp_path / "gateway_state.json.lock"
    lock.touch(mode=0o444)
    try:
        assert status.write_runtime_status(gateway_state="starting")
        assert status.read_runtime_status()["pid"] == os.getpid()
    finally:
        lock.chmod(0o600)


def test_cached_gateway_snapshot_preserves_new_operator_intent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    service_manager._write_gateway_desired_state("gateway-default", "running")
    assert status.write_runtime_status(gateway_state="running")
    service_manager._write_gateway_desired_state("gateway-default", "stopped")
    assert status.write_runtime_status(active_agents=2)
    current = status.read_runtime_status()
    assert current["pid"] == os.getpid()
    assert current["active_agents"] == 2
    assert current["desired_state"] == "stopped"
