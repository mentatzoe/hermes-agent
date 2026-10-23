"""Delegated terminals cannot arm a parent-chat watcher (aleph-vault#256)."""

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.delegation_context import delegated_child_context
from gateway.session_context import clear_session_vars, set_session_vars
from tools.process_registry import ProcessRegistry, ProcessSession
from tools.terminal_tool_background import spawn_background_process


@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize("serving", [False, True])
def test_spawn_arms_only_main_session_watchers(monkeypatch, tmp_path, child, serving):
    import gateway.run as gateway_run
    import tools.process_registry as registry_module
    import tools.terminal_tool_background as background

    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_ensure_heartbeat_thread", lambda: None)
    session = ProcessSession(id="proc_spawned", command="work", started_at=1)
    monkeypatch.setattr(registry_module, "process_registry", registry)
    monkeypatch.setattr(background, "_spawn", lambda *_a, **_kw: session)
    arm = Mock(return_value=serving)
    runner = SimpleNamespace(arm_process_watcher=arm)
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
    tokens = set_session_vars(
        platform="telegram", chat_id="123", session_id="owning-session",
        session_key="agent:main:telegram:dm:123", async_delivery=True,
    )
    try:
        with delegated_child_context() if child else nullcontext():
            result = json.loads(spawn_background_process(
                command="work", env=SimpleNamespace(), env_type="local",
                effective_task_id="owner", task_id="owner", session_key="owning-session",
                workdir=str(tmp_path), cwd=str(tmp_path), effective_pty=False,
                notify_on_complete=True, watch_patterns=None, approval_note=None,
                pty_disabled_reason=None, heartbeat_seconds=60,
            ))
    finally:
        clear_session_vars(tokens)

    assert result.get("error") is None
    assert result["notify_on_complete"] is not child
    assert arm.call_count == (0 if child else 1)
    assert len(registry.pending_watchers) == (0 if child or serving else 1)
    if child:
        assert result["subagent_note"]
        assert session.watcher_interval == 0
        assert session.heartbeat_seconds == 0
        # Keep registry accounting and explicit child-to-parent handoff available.
        assert session.notify_on_complete is True
