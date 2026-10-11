"""Spawn/queue/checkpoint provenance with real terminal control flow, fake I/O."""
from contextlib import nullcontext
from types import SimpleNamespace
import json
import os

import pytest

from agent.delegation_context import delegated_child_context
from tests.tools.test_terminal_tool_pty_fallback import _base_config
from tools.process_registry import ProcessRegistry, ProcessSession
import tools.process_registry as pr
import tools.terminal_tool as tt


@pytest.mark.parametrize("child", [False, True])
def test_terminal_stamps_execution_not_session_history(monkeypatch, tmp_path, child):
    registry = ProcessRegistry()
    session = ProcessSession(id="proc_provenance", command="true", pid=os.getpid())
    monkeypatch.setattr(pr, "process_registry", registry)
    monkeypatch.setattr(registry, "spawn_local", lambda **kwargs: session)
    monkeypatch.setattr(tt, "_get_env_config", lambda: _base_config(tmp_path))
    monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(tt, "_check_all_guards", lambda *a, **k: {"approved": True})
    monkeypatch.setattr("gateway.session_context.async_delivery_supported", lambda: True)
    monkeypatch.setitem(tt._active_environments, "default", SimpleNamespace(env={}))
    monkeypatch.setitem(tt._last_activity, "default", 0.0)
    with delegated_child_context() if child else nullcontext():
        result = json.loads(tt.terminal_tool(command="true", background=True, notify_on_complete=True))
    assert not result.get("error"), result
    assert session.delegated_child is child


@pytest.mark.parametrize("child", [None, False, True])
def test_provenance_survives_queue_and_checkpoint(monkeypatch, tmp_path, child):
    checkpoint = tmp_path / "processes.json"
    monkeypatch.setattr(pr, "CHECKPOINT_PATH", checkpoint)
    registry = ProcessRegistry()
    session = ProcessSession(id="proc_provenance", command="synthetic", pid=os.getpid(),
                             parent_session_id="spawn-session", delegated_child=child,
                             notify_on_complete=True)
    registry._running[session.id] = session
    registry._write_checkpoint()
    recovered = ProcessRegistry()
    assert recovered.recover_from_checkpoint() == 1
    assert recovered.get(session.id).delegated_child is child
    assert recovered.get(session.id).parent_session_id == "spawn-session"
    session.exited = True
    session.exit_code = 0
    registry._move_to_finished(session)
    evt = registry.completion_queue.get_nowait()
    assert evt["delegated_child"] is child
    assert evt["parent_session_id"] == "spawn-session"
