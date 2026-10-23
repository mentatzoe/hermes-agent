"""Explicit handoff replaces a suppressed child watcher (aleph-vault#256)."""

import asyncio
import json
import threading
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agent.delegation_context import delegated_child_context
from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session_context import clear_session_vars, set_session_vars
from tools.process_registry import ProcessRegistry, _handle_process
from tools.terminal_tool_background import spawn_background_process


@pytest.mark.asyncio
@pytest.mark.parametrize("serving", [False, True])
@pytest.mark.parametrize("parent_is_child", [False, True])
async def test_handoff_arms_only_the_new_main_owner(monkeypatch, tmp_path, serving, parent_is_child):
    import gateway.run as gateway_run
    import tools.delegate_tool_registry as delegates
    import tools.process_registry as registry_module
    import tools.terminal_tool_background as background

    registry = ProcessRegistry()
    monkeypatch.setattr(registry_module, "process_registry", registry)
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    child_id = "sa-0-1234abcd"
    route = "agent:main:telegram:dm:123"
    parent = SimpleNamespace(
        session_id="parent-session", _current_task_id="sa-1-5678abcd" if parent_is_child else "parent-turn",
        _gateway_session_key=route,
    )
    child = SimpleNamespace(_delegate_parent_ref=lambda: parent)
    monkeypatch.setitem(delegates._active_subagents, child_id, {"agent": child})
    arm = Mock(return_value=serving)
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: SimpleNamespace(arm_process_watcher=arm))

    def spawn(*_args, **_kwargs):
        session = registry._new_session("work", child_id, child_id, "child-key", str(tmp_path))
        registry._running[session.id] = session
        return session

    monkeypatch.setattr(background, "_spawn", spawn)
    tokens = set_session_vars(
        platform="telegram", chat_id="123", session_id="parent-session",
        session_key="stale-inherited-key", async_delivery=True,
    )
    try:
        with delegated_child_context("child-session"):
            result = json.loads(spawn_background_process(
                command="work", env=SimpleNamespace(), env_type="local", effective_task_id=child_id,
                task_id=child_id, session_key="child-key", workdir=str(tmp_path), cwd=str(tmp_path),
                effective_pty=False, notify_on_complete=True, watch_patterns=None,
                approval_note=None, pty_disabled_reason=None,
            ))
            arm.assert_not_called()
            assert registry.pending_watchers == []
            session = registry.get(result["session_id"])
            handed = json.loads(_handle_process(
                {"action": "handoff", "session_id": session.id, "data": "Check the build result"}, task_id=child_id,
            ))
    finally:
        clear_session_vars(tokens)

    assert handed["status"] == "handed_off"
    if parent_is_child:
        arm.assert_not_called()
        assert registry.pending_watchers == []
        return
    arm.assert_called_once()
    watcher = arm.call_args.args[0] if serving else registry.pending_watchers[0]
    assert watcher["session_key"] == route
    assert watcher["parent_session_id"] == "parent-session"
    assert session.delegated_child is False
    session.exited, session.exit_code, session.output_buffer = True, 0, "build passed"

    def accept(event):
        event._gateway_accepted = True

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=accept), send=AsyncMock())
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SimpleNamespace(_ensure_loaded=lambda: None, _entries={})
    runner._session_source_cache = {}
    runner._session_db = SimpleNamespace(get_session=AsyncMock(return_value={"ended_at": None}))
    runner._completion_delivery_lock = threading.Lock()
    runner._completion_delivery_retention = 2048
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._background_tasks = set()
    runner._load_background_notifications_mode = lambda: "off"
    watcher["check_interval"] = 0
    await asyncio.wait_for(runner._run_process_watcher(watcher), timeout=10)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.internal is True
    assert event.metadata["gateway_session_id"] == "parent-session"
    assert "build passed" in event.text
    adapter.send.assert_not_awaited()
