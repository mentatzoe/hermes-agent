"""Real-shell regressions for mentatzoe/aleph-vault#245 (no model/network calls)."""

import json
import shlex
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools import terminal_tool as tt
from tools.delegate_tool_child_run import _ChildRun, _close_child
from tools.file_tools import _get_file_ops, clear_file_ops_cache


@pytest.fixture
def shells(tmp_path, monkeypatch):
    for name in ("HOME", "HERMES_REAL_HOME"):
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setenv("TERMINAL_TEMP_DIR", str(tmp_path))
    monkeypatch.setenv("BASH_ENV", "")
    monkeypatch.setattr(tt, "_terminal_config_bridge_attempted", True)
    monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(tt, "_pre_exec_block", lambda *a, **kw: None)
    monkeypatch.setattr(tt, "_run_approval_guards", lambda *a, **kw: SimpleNamespace(note=None, approved_run=False))
    monkeypatch.setattr(tt, "_active_environments", {})
    monkeypatch.setattr(tt, "_session_cwd", {})
    monkeypatch.setattr(tt, "_container_aliases", {})
    monkeypatch.setattr(tt, "_last_activity", {})
    monkeypatch.setattr(tt, "_task_env_overrides", {})
    runs = []

    def command(task, text):
        result = json.loads(tt.terminal_tool(text, task_id=task, timeout=30))
        assert result.get("exit_code") == 0, result
        return result["output"].strip()

    def child(parent="parent", name="child", script=None):
        agent = SimpleNamespace()
        if script is not None:
            agent.run_conversation = script
        run = _ChildRun(agent, SimpleNamespace(_current_task_id=parent), 0, "test", name, None)
        with patch("tools.delegate_tool_child_run._create_isolated_worktree", return_value=None):
            run.seed_workspace()
        runs.append(run)
        return run

    yield tmp_path, command, child
    for run in runs:
        _close_child(run.child, "test close")
    from tools.terminal_tool_lifecycle import _cleanup_env
    for env in tt._active_environments.values():
        _cleanup_env(env)
    clear_file_ops_cache()


def state(command, task):
    return command(task, 'printf "%s\\n" "$HOME" "$HERMES_HOME" "$HERMES_REAL_HOME" "${ISOLATION_TEST-unset}"; pwd -P').splitlines()


@pytest.mark.parametrize("session_key", ["", "gateway-parent"])
def test_child_exports_and_cd_are_bidirectionally_isolated(shells, monkeypatch, session_key):
    root, command, child = shells
    monkeypatch.setenv("HERMES_SESSION_KEY", session_key)
    baseline = state(command, "parent")
    parent_dir = root / "parent-dir"
    child_dir = root / "child-dir"
    parent_dir.mkdir()
    child_dir.mkdir()
    command("parent", f"export ISOLATION_TEST=parent; cd {shlex.quote(str(parent_dir))}")
    child_run = child()
    child_initial = state(command, "child")
    assert child_initial[:3] == baseline[:3]
    assert child_initial[3] == "unset"
    command("child", f"export HOME={shlex.quote(str(root / 'child-home'))} HERMES_HOME={shlex.quote(str(root / 'child-hermes'))} HERMES_REAL_HOME={shlex.quote(str(root / 'child-real'))} ISOLATION_TEST=child; cd {shlex.quote(str(child_dir))}")
    child_state = state(command, "child")
    assert child_state == [str(root / "child-home"), str(root / "child-hermes"), str(root / "child-real"), "child", str(child_dir)]
    assert state(command, "parent") == [*baseline[:3], "parent", str(parent_dir)]
    command("parent", f"export HOME={shlex.quote(str(root / 'parent-home'))} ISOLATION_TEST=changed; cd {shlex.quote(str(root))}")
    assert state(command, "child") == child_state
    assert _get_file_ops("child")._exec('printf "%s" "$HOME"').stdout == str(root / "child-home")
    _close_child(child_run.child, "test child close")
    assert state(command, "parent")[3:] == ["changed", str(root)]


def test_siblings_nested_children_and_file_first_share_files_not_shells(shells):
    root, command, child = shells
    baseline = state(command, "parent")
    for name in ("one", "two"):
        child(name=name)
    # File tools may create the shell before its first terminal call.
    _get_file_ops("one")._exec("export ISOLATION_TEST=one")
    assert state(command, "one")[3] == "one"
    assert state(command, "two") == baseline
    child(parent="one", name="nested")
    assert state(command, "nested") == baseline
    with ThreadPoolExecutor(max_workers=3) as pool:
        outputs = list(pool.map(lambda t: command(t, f"export ISOLATION_TEST={t}; printf '%s' \"$ISOLATION_TEST\""), ("one", "two", "nested")))
    assert outputs == ["one", "two", "nested"]
    command("one", f"printf shared > {shlex.quote(str(root / 'shared.txt'))}")
    assert command("two", f"read value < {shlex.quote(str(root / 'shared.txt'))}; printf '%s' \"$value\"") == "shared"
    assert state(command, "parent") == baseline
    assert len(tt._active_environments) == 1


@pytest.mark.parametrize("background", [False, True])
def test_child_worker_runs_exports_without_changing_parent(shells, background):
    import threading
    root, command, child = shells
    baseline = state(command, "parent")
    entered = threading.Event()
    resume = threading.Event()
    target = root / "worker-cwd"
    target.mkdir()

    def script(*, task_id, **kwargs):
        command(task_id, f"export HOME={shlex.quote(str(root / 'worker-home'))} ISOLATION_TEST=worker; cd {shlex.quote(str(target))}")
        entered.set()
        assert resume.wait(10)
        assert state(command, task_id)[3:] == ["worker", str(target)]
        return {"final_response": "checked", "completed": True}

    run = child(script=script)
    if background:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(run.await_child)
            assert entered.wait(10)
            assert state(command, "parent") == baseline
            command("parent", "export ISOLATION_TEST=parent-later")
            resume.set()
            result, failure, deferred = future.result(timeout=20)
        assert state(command, "parent") == [*baseline[:3], "parent-later", baseline[4]]
    else:
        resume.set()
        result, failure, deferred = run.await_child()
        assert state(command, "parent") == baseline
    assert failure is None and not deferred
    assert result["final_response"] == "checked"
    _close_child(run.child, "test child close")
    assert command("parent", "printf alive") == "alive"


def test_backend_eviction_rebinds_file_tools_and_child_shell(shells):
    from tools.terminal_tool_lifecycle import cleanup_vm
    root, command, child = shells
    baseline = state(command, "parent")
    child()
    command("child", "export ISOLATION_TEST=old")
    old = _get_file_ops("child")
    cleanup_vm(tt._resolve_container_task_id("parent"))
    new = _get_file_ops("child")
    assert new is not old
    assert state(command, "child") == baseline


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("mode", ["pipe", "pty", "pty-fallback"])
def test_background_commands_use_child_snapshot_without_publishing_mutations(shells, monkeypatch, mode):
    from tools.process_registry import process_registry
    if mode == "pty-fallback":
        def unavailable(*args, **kwargs):
            raise ImportError("synthetic missing PTY")
        monkeypatch.setattr(process_registry, "_spawn_local_pty", unavailable)
    pty = mode != "pipe"
    root, command, child = shells
    baseline = state(command, "parent")
    child()
    target = root / "background-cwd"
    target.mkdir()
    command("child", f"export HOME={shlex.quote(str(root / 'bg-home'))} HERMES_HOME={shlex.quote(str(root / 'bg-hermes'))} HERMES_REAL_HOME={shlex.quote(str(root / 'bg-real'))} ISOLATION_TEST=foreground; cd {shlex.quote(str(target))}")
    result = json.loads(tt.terminal_tool(
        'printf "CHECK:%s:%s:%s:%s\\n" "$HOME" "$HERMES_HOME" "$HERMES_REAL_HOME" "$ISOLATION_TEST"; pwd -P; export ISOLATION_TEST=background',
        task_id="child", background=True, pty=pty, timeout=30))
    assert "session_id" in result, result
    if mode == "pty":
        assert process_registry.get(result["session_id"])._pty is not None
    completed = process_registry.wait(result["session_id"], timeout=20)
    assert completed["status"] == "exited", completed
    assert completed["exit_code"] == 0, completed
    assert f"CHECK:{root / 'bg-home'}:{root / 'bg-hermes'}:{root / 'bg-real'}:foreground" in completed["output"]
    assert str(target) in completed["output"]
    assert state(command, "child")[3] == "foreground"
    assert state(command, "parent") == baseline
