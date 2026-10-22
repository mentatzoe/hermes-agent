"""Backend contracts for delegated shells; real bash with stand-in remote I/O."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.environments.base import BaseEnvironment
from tools.environments.local import LocalEnvironment
from tools.terminal_tool_sessions import DelegatedShellSession


@pytest.fixture
def transport(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_REAL_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_TEMP_DIR", str(tmp_path))
    monkeypatch.setenv("BASH_ENV", "")
    backend = LocalEnvironment(cwd=str(tmp_path))
    yield backend
    backend.cleanup()


def test_managed_gateway_uses_original_environment_not_parent_exports(transport, tmp_path, monkeypatch):
    from tools.environments import managed_modal as mm
    import tools.credential_files
    monkeypatch.setattr(tools.credential_files, "get_credential_file_mounts", lambda: [])
    monkeypatch.setattr(mm, "resolve_managed_tool_gateway", lambda _: SimpleNamespace(
        gateway_origin="https://invalid.example", nous_user_token="synthetic"))
    monkeypatch.setattr(mm.ManagedModalEnvironment, "_create_sandbox", lambda _: "test-sandbox")
    backend = mm.ManagedModalEnvironment(image="test", cwd=str(tmp_path))
    backend._startup_snapshot_path = str(tmp_path / "managed-startup.sh")

    def request(method, path, *, json=None, **kwargs):
        assert method == "POST"
        if path.endswith("/terminate"):
            return SimpleNamespace(status_code=200)
        assert path.endswith("/execs")
        result = transport.execute(json["command"], cwd=json["cwd"], stdin_data=json.get("stdinData"))
        return SimpleNamespace(status_code=200, json=lambda: {
            "status": "completed", "output": result["output"], "returncode": result["returncode"]})

    monkeypatch.setattr(backend, "_request", request)
    backend.execute("export ISOLATION_TEST=parent")
    shell = DelegatedShellSession(backend, str(tmp_path))
    try:
        assert shell.execute('printf "%s" "${ISOLATION_TEST-unset}"')["output"] == "unset"
        assert shell.execute("export ISOLATION_TEST=child")["returncode"] == 0
        assert shell.execute('printf "%s" "$ISOLATION_TEST"')["output"] == "child"
        assert backend.execute('printf "%s" "$ISOLATION_TEST"')["output"] == "parent"
    finally:
        shell.cleanup()
        backend.cleanup()


def test_docker_recovery_rebuilds_each_child_snapshot(transport, tmp_path, monkeypatch):
    from tools.environments.docker import DockerEnvironment
    backend = DockerEnvironment.__new__(DockerEnvironment)
    BaseEnvironment.__init__(backend, cwd=str(tmp_path), timeout=30)
    backend._container_id = "first"
    backend._persist_across_processes = True
    backend.cleanup = lambda **kwargs: None
    gone = False

    def run(command, **kwargs):
        if gone:
            return transport._run_bash("printf 'No such container'; exit 1", **kwargs)
        return transport._run_bash(command, **kwargs)

    monkeypatch.setattr(backend, "_run_bash", run)
    backend.init_session()
    first = DelegatedShellSession(backend, str(tmp_path))
    second = DelegatedShellSession(backend, str(tmp_path))
    first.execute("export ISOLATION_TEST=first")
    second.execute("export ISOLATION_TEST=second")

    def recreate():
        nonlocal gone
        gone = False
        backend._container_id = "replacement"
        for shell in (first, second):
            Path(shell._snapshot_path).unlink()
        backend.init_session()
        backend._mark_recreated()
        return True

    monkeypatch.setattr(backend, "_recreate_container", recreate)
    gone = True
    try:
        result = first.execute('printf "%s" "${ISOLATION_TEST-unset}"')
        assert result["returncode"] == 0 and result["output"] == "unset"
        result = second.execute('printf "%s" "${ISOLATION_TEST-unset}"')
        assert result["returncode"] == 0 and result["output"] == "unset"
    finally:
        first.cleanup()
        second.cleanup()
