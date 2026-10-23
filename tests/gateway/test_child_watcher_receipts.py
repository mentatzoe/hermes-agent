"""Child process receipts must not bypass internal ownership (aleph-vault#256)."""

import asyncio
import threading
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture
def watcher_case(monkeypatch, tmp_path):
    import tools.process_registry as registry_module

    registry = ProcessRegistry()
    monkeypatch.setattr(registry_module, "process_registry", registry)
    monkeypatch.setattr(registry_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    session = ProcessSession(
        id="proc_owned", command="printf private-output", started_at=1,
        task_id="task", owner_task_id="owner", session_key="agent:main:telegram:dm:123",
        output_buffer="private-output\n", exited=True, exit_code=0, delegated_child=False,
    )
    registry._finished[session.id] = session
    adapter = SimpleNamespace(send=AsyncMock(), handle_message=AsyncMock())
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SimpleNamespace(_ensure_loaded=lambda: None, _entries={})
    runner._session_source_cache = {}
    runner._completion_delivery_lock = threading.Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    runner._background_tasks = set()
    runner._load_background_notifications_mode = lambda: "concise"
    watcher = {
        "session_id": session.id, "check_interval": 0, "platform": "telegram",
        "chat_id": "123", "session_key": session.session_key, "notify_on_complete": True,
    }
    adapter._active_sessions = {session.session_key: asyncio.Event()}
    return runner, adapter, registry, session, watcher


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["concise", "all", "result", "error"])
@pytest.mark.parametrize("agent_notify", [False, True])
async def test_child_completion_never_sends_to_busy_parent(watcher_case, mode, agent_notify):
    runner, adapter, _, session, watcher = watcher_case
    session.delegated_child = True
    session.exit_code = 1  # Include failure receipts in error mode.
    watcher["notify_on_complete"] = agent_notify
    runner._load_background_notifications_mode = lambda: mode

    await runner._run_process_watcher(watcher)

    adapter.send.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("provenance", ["explicit", "legacy_owner", "durable_child", "unknown"])
async def test_direct_watcher_send_checks_ownership(watcher_case, provenance):
    runner, adapter, _, session, watcher = watcher_case
    if provenance == "explicit":
        session.delegated_child = True
    else:
        session.delegated_child = None
        if provenance == "legacy_owner":
            session.owner_task_id = "sa-12-1234abcd"
        else:
            session.parent_session_id = "child-session"
            row = {"source": "subagent"} if provenance == "durable_child" else None
            runner._session_db = SimpleNamespace(get_session=AsyncMock(return_value=row))

    await runner._send_watcher_message("telegram", "123", None, "private-output", watcher)

    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_rejected_internal_delivery_has_no_busy_receipt(watcher_case):
    runner, adapter, _, _, watcher = watcher_case
    runner._enqueue_process_completion_notification = AsyncMock(return_value=None)

    await runner._run_process_watcher(watcher)

    runner._enqueue_process_completion_notification.assert_awaited_once()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_retryable_delivery_retries_before_main_session_receipt(watcher_case):
    runner, adapter, _, _, watcher = watcher_case
    runner._enqueue_process_completion_notification = AsyncMock(side_effect=[False, True])

    await runner._run_process_watcher(watcher)

    assert runner._enqueue_process_completion_notification.await_count == 2
    adapter.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_off_keeps_main_session_internal_followthrough(watcher_case):
    runner, adapter, _, _, watcher = watcher_case
    runner._load_background_notifications_mode = lambda: "off"

    def accept(event):
        event._gateway_accepted = True

    adapter.handle_message.side_effect = accept
    await runner._run_process_watcher(watcher)

    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].internal is True
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_child_running_output_never_sends_to_parent(watcher_case):
    runner, adapter, registry, session, watcher = watcher_case
    session.delegated_child = True
    session.exited = False
    watcher["notify_on_complete"] = False
    runner._load_background_notifications_mode = lambda: "all"
    calls = 0
    original_get = registry.get

    def running_then_gone(session_id):
        nonlocal calls
        calls += 1
        return original_get(session_id) if calls <= 2 else None

    registry.get = running_then_gone
    await runner._run_process_watcher(watcher)

    adapter.send.assert_not_awaited()
