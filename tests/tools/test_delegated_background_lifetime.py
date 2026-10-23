"""Event-gated regressions from the independent PR #33 review."""

import json
import shlex
import time

import pytest

from tests.tools.test_delegated_terminal_sessions import shells as _shell_fixture
from tools import terminal_tool as tt
from tools.delegate_tool_child_run import _close_child
from tools.process_registry import process_registry

pytestmark = pytest.mark.platforms("posix")
shells = _shell_fixture


def wait_file(path):
    deadline = time.monotonic() + 10
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"startup did not reach {path}")
        time.sleep(0.02)


@pytest.mark.parametrize("lifetime", ["control", "handoff", "persisted"])
def test_background_launch_state_survives_child_close(shells, monkeypatch, lifetime):
    root, command, child = shells
    monkeypatch.setenv("HERMES_HOME", str(root / "hermes"))
    command("parent", "printf parent")
    run = child()
    command("child", "export ISOLATION_TEST=child")
    entered, release = root / "startup-entered", root / "startup-release"
    profile = root / ".bash_profile"
    profile.write_text(f"touch {shlex.quote(str(entered))}\nwhile [ ! -f {shlex.quote(str(release))} ]; do sleep 0.02; done\n")
    result = json.loads(tt.terminal_tool(
        'printf "PAYLOAD:%s" "$ISOLATION_TEST"', task_id="child", background=True,
        persist_on_release=lifetime == "persisted", timeout=30))
    assert "session_id" in result, result
    session_id = result["session_id"]
    try:
        wait_file(entered)
        copies = list(root.glob("hermes-bg-snap-*.sh"))
        assert len(copies) == 1
        assert copies[0].stat().st_mode & 0o077 == 0
        if lifetime == "handoff":
            moved = process_registry.transfer_ownership(
                session_id, from_owner="child", to_owner="parent",
                to_task_id=tt._resolve_container_task_id("parent"), to_session_key="parent", note="regression handoff")
            assert moved is not None
        if lifetime != "control":
            _close_child(run.child, "regression child close")
        release.touch()
        completed = process_registry.wait(session_id, timeout=20)
        assert completed["exit_code"] == 0, completed
        assert "PAYLOAD:child" in completed["output"], completed
        deadline = time.monotonic() + 10
        while list(root.glob("hermes-bg-snap-*.sh")) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not list(root.glob("hermes-bg-snap-*.sh"))
    finally:
        release.touch()
        process_registry.kill_process(session_id, source="test_cleanup", consume_output=True)
        profile.unlink(missing_ok=True)


@pytest.mark.parametrize("failure", ["copy", "spawn"])
def test_failed_launch_does_not_leave_private_shell_state(shells, monkeypatch, failure):
    from pathlib import Path
    from tools.terminal_tool_background import _spawn
    from tools.terminal_tool_lifecycle import get_active_env
    root, command, child = shells
    command("parent", "printf parent")
    child()
    command("child", "export ISOLATION_TEST=child")
    env = get_active_env("child")
    assert env is not None
    called = []

    def fail_spawn(**kwargs):
        called.append(True)
        raise RuntimeError("synthetic spawn failure")

    monkeypatch.setattr(process_registry, "spawn_local", fail_spawn)
    if failure == "copy":
        Path(env._snapshot_path).unlink()
    with pytest.raises(RuntimeError):
        _spawn(process_registry, env=env, env_type="local", command="printf payload",
               cwd=str(root), effective_task_id=tt._resolve_container_task_id("child"),
               task_id="child", session_key="child", effective_pty=False)
    assert bool(called) == (failure == "spawn")
    assert not list(root.glob("hermes-bg-snap-*.sh"))
    assert command("parent", "printf alive") == "alive"


@pytest.mark.parametrize("failure", ["result", "drain", "wrapping", "running-copy", "run"])
def test_preparation_exception_removes_created_copy_before_payload(shells, monkeypatch, failure):
    from tools.terminal_tool_background import _spawn
    from tools.terminal_tool_lifecycle import get_active_env
    from tools.environments import base
    root, command, child = shells
    command("parent", "printf parent")
    child()
    command("child", "export ISOLATION_TEST=child")
    shell = get_active_env("child")
    original_run, original_wait = shell._run_bash, shell._wait_for_process
    copies, processes, spawned = [], [], []

    def copy_then_raise(proc, *args, **kwargs):
        if copies:
            return original_wait(proc, *args, **kwargs)
        if failure == "running-copy":
            wait_file(root / "copy-entered")
        else:
            original_wait(proc, *args, **kwargs)
        copies.extend(root.glob("hermes-bg-snap-*.sh"))
        assert copies
        raise OSError("synthetic preparation wait failure")

    def run_copy(command, **kwargs):
        if command.startswith("umask 077; cp"):
            if failure == "running-copy":
                command += f"; touch {shlex.quote(str(root / 'copy-entered'))}; while :; do sleep 0.02; done"
            proc = original_run(command, **kwargs)
            processes.append(proc)
            if failure == "run":
                proc.wait(timeout=10)
                raise OSError("synthetic preparation run failure")
            return proc
        return original_run(command, **kwargs)

    def unexpected_spawn(**kwargs):
        spawned.append(True)
        raise AssertionError("payload must not spawn")

    monkeypatch.setattr(process_registry, "spawn_local", unexpected_spawn)
    monkeypatch.setattr(shell, "_run_bash", run_copy)
    if failure in {"result", "running-copy"}:
        monkeypatch.setattr(shell, "_wait_for_process", copy_then_raise)
    elif failure == "drain":
        original_drain = base._start_drain_thread

        def fail_drain(proc, *args, **kwargs):
            if not copies:
                proc.wait(timeout=10)
                copies.extend(root.glob("hermes-bg-snap-*.sh"))
                assert copies
                raise RuntimeError("synthetic preparation drain failure")
            return original_drain(proc, *args, **kwargs)

        monkeypatch.setattr(base, "_start_drain_thread", fail_drain)
    elif failure == "wrapping":
        def fail_quote(cwd):
            assert list(root.glob("hermes-bg-snap-*.sh"))
            raise ValueError("synthetic preparation quote failure")
        monkeypatch.setattr(shell.backend, "_quote_cwd_for_cd", fail_quote)
    with pytest.raises((OSError, RuntimeError, ValueError), match="synthetic preparation"):
        _spawn(process_registry, env=shell, env_type="local", command="printf payload",
               cwd=str(root), effective_task_id=tt._resolve_container_task_id("child"),
               task_id="child", session_key="child", effective_pty=False)
    assert not spawned
    assert not list(root.glob("hermes-bg-snap-*.sh"))
    assert all(proc.poll() is not None for proc in processes)


@pytest.mark.parametrize("mode", ["pipe", "pty", "pty-fallback", "remote"])
@pytest.mark.parametrize("explicit_workdir", [False, True])
def test_background_cwd_survives_login_startup_cd(shells, monkeypatch, mode, explicit_workdir):
    root, command, child = shells
    monkeypatch.setenv("HERMES_HOME", str(root / "hermes"))
    command("parent", "printf parent")
    child()
    recorded, requested, startup = root / "recorded", root / "requested", root / "startup"
    for path in (recorded, requested, startup):
        path.mkdir()
    command("child", f"cd {shlex.quote(str(recorded))}")
    target = requested if explicit_workdir else recorded
    profile = root / ".bash_profile"
    profile.write_text(f"cd {shlex.quote(str(startup))}\n")
    if mode == "pty-fallback":
        def unavailable(*args, **kwargs):
            raise ImportError("synthetic unavailable PTY")
        monkeypatch.setattr(process_registry, "_spawn_local_pty", unavailable)
    if mode == "remote":
        from tools.terminal_tool_background import _spawn
        from tools.terminal_tool_lifecycle import get_active_env
        session = _spawn(
            process_registry, env=get_active_env("child"), env_type="ssh", command="pwd -P",
            cwd=str(target), effective_task_id=tt._resolve_container_task_id("child"),
            task_id="child", session_key="child", effective_pty=False)
        result = {"session_id": session.id}
    else:
        result = json.loads(tt.terminal_tool(
            "pwd -P", task_id="child", background=True, pty=mode != "pipe", timeout=30,
            workdir=str(target) if explicit_workdir else None))
    assert "session_id" in result, result
    try:
        completed = process_registry.wait(result["session_id"], timeout=20)
        assert completed["exit_code"] == 0, completed
        assert completed["output"].strip() == str(target), completed
    finally:
        profile.unlink(missing_ok=True)
        process_registry.kill_process(result["session_id"], source="test_cleanup", consume_output=True)
