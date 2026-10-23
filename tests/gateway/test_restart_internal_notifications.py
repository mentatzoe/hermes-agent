"""Internal notices survive the restart admission boundary (aleph-vault#241)."""

import asyncio
import pytest

from evals.heartbeat_idle_wire import WireAdapter
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore, build_session_key
from gateway.wake import deliver_wake
from tests.gateway.test_kanban_wake_acceptance import drain


def route(tmp_path, *, restarting=True):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=False)
    runner._running = True
    runner._draining = restarting
    runner._restart_requested = restarting
    runner._startup_restore_in_progress = False
    runner._running_agents = {}
    runner._background_tasks = set()
    runner._shutdown_event = asyncio.Event()
    runner._is_session_running = lambda key: False
    runner._resolve_profile_home_for_source = lambda source: tmp_path
    runner._status_action_gerund = lambda: "restarting"
    runner.session_store = SessionStore(sessions_dir=tmp_path / "sessions", config=runner.config)
    adapter = WireAdapter(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
    adapter.wire = []
    adapter.gateway_runner = runner
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._delivery_adapter_for = lambda source: adapter
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm")
    key = build_session_key(source)
    entry = runner.session_store.get_or_create_session(source)
    received = []

    async def handler(event):
        if runner._draining:
            handled, reply, _ = await runner._hm_dispatch_quick_and_plugin_commands(event, event.source, None)
            assert handled
            return reply
        event._heartbeat_execution_started = True  # substitute model boundary
        received.append((event.text, event.metadata.copy(), event.internal))
        from gateway.restart_notifications import complete_notice
        await complete_notice(runner, event)
        return None

    adapter.set_message_handler(handler)
    return runner, adapter, source, key, entry.session_id, received


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["kanban", "background-process", "delegation", "cron-delivery"])
async def test_internal_notice_is_silent_then_replayed_once_to_original_session(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    old, adapter, source, key, session_id, old_received = route(tmp_path)
    await deliver_wake(adapter, text=f"{kind} completed", source=source)
    await drain(adapter)
    assert adapter.wire == [], "a notice is not a human message and must not get the restart busy reply"
    assert old_received == []

    from gateway.restart_notifications import replay_pending

    new, fresh_adapter, _, _, fresh_session_id, received = route(tmp_path, restarting=False)
    assert fresh_session_id == session_id
    await replay_pending(new)
    await drain(fresh_adapter)
    await replay_pending(new)
    await drain(fresh_adapter)
    assert [row[0] for row in received] == [f"{kind} completed"]
    assert received[0][1]["gateway_session_key"] == key
    assert received[0][1]["gateway_session_id"] == session_id
    assert received[0][2] is True
    assert fresh_adapter.wire == []

    # A fresh replay consumer must not re-inject an already handled notice.
    third, third_adapter, _, _, _, third_received = route(tmp_path, restarting=False)
    await replay_pending(third)
    await drain(third_adapter)
    assert third_received == []


@pytest.mark.asyncio
async def test_human_message_keeps_busy_reply(tmp_path):
    _, adapter, source, _, _, received = route(tmp_path)
    await adapter.handle_message(MessageEvent(text="Are you there?", source=source))
    await drain(adapter)
    assert len(adapter.wire) == 1
    assert "not accepting new work" in adapter.wire[0]
    assert received == []


@pytest.mark.asyncio
async def test_scheduled_notice_caught_when_restart_starts_before_dispatch(tmp_path):
    from gateway.restart_notifications import replay_pending

    old, adapter, source, _, _, received = route(tmp_path, restarting=False)
    await deliver_wake(adapter, text="scheduled before drain", source=source)
    old._draining = old._restart_requested = True
    await drain(adapter)
    assert received == []
    assert adapter.wire == []
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    await replay_pending(new)
    await drain(fresh)
    assert [row[0] for row in received] == ["scheduled before drain"]


@pytest.mark.asyncio
async def test_failed_handoff_is_not_acknowledged(tmp_path, monkeypatch):
    from gateway import restart_notifications as notices
    from gateway.wake import WakeNotAccepted, admit_internal_event

    _, adapter, source, _, _, _ = route(tmp_path)
    def fail(*args):
        raise OSError("disk unavailable")
    monkeypatch.setattr(notices, "_save", fail)
    event = MessageEvent(text="retain me", source=source, internal=True)
    with pytest.raises(WakeNotAccepted, match="handoff failed"):
        await admit_internal_event(adapter, event)
    assert event._gateway_accepted is False
    assert not adapter._background_tasks
    assert adapter.wire == []


@pytest.mark.asyncio
async def test_replay_waits_for_restore_and_reconnect(tmp_path):
    from gateway.restart_notifications import replay_pending

    _, adapter, source, _, _, _ = route(tmp_path)
    await deliver_wake(adapter, text="wait for connection", source=source)
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    new._startup_restore_in_progress = True
    await replay_pending(new)
    assert not fresh._background_tasks
    new._startup_restore_in_progress = False
    handler = fresh._message_handler
    fresh.set_message_handler(None)
    await replay_pending(new)
    assert received == []
    fresh.set_message_handler(handler)
    await asyncio.gather(replay_pending(new), replay_pending(new))
    await drain(fresh)
    assert len(received) == 1


@pytest.mark.asyncio
async def test_second_restart_before_replay_execution_keeps_one_notice(tmp_path):
    from gateway.restart_notifications import replay_pending

    _, adapter, source, _, _, _ = route(tmp_path)
    await deliver_wake(adapter, text="two restarts", source=source)
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    await replay_pending(new)
    new._draining = new._restart_requested = True
    await drain(fresh)
    assert received == []
    third, fresh, _, _, _, received = route(tmp_path, restarting=False)
    await replay_pending(third)
    await drain(fresh)
    await replay_pending(third)
    assert len(received) == 1


@pytest.mark.asyncio
async def test_kanban_producer_advances_only_after_durable_restart_handoff(tmp_path, monkeypatch):
    from gateway.kanban_watchers_notifier import _KanbanNotification, _notifier_collect
    from gateway.restart_notifications import replay_pending
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    old, adapter, _, key, _, _ = route(tmp_path)
    old._kanban_dispatcher_lock_handle = object()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="restart result", assignee="worker", session_id=key)
        kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="42",
                           user_id="42", chat_type="dm", delivery_mode="wake")
        kb.complete_task(conn, tid, summary="actual Kanban handoff")
    finally:
        conn.close()
    deliveries = await asyncio.to_thread(_notifier_collect, old, kb,
        notifier_profile=None, gc_due=False, gc_retention_days=30)
    for delivery in deliveries:
        await _KanbanNotification(old, delivery, platform_cls=Platform, sub_fail_counts={}).deliver()
    assert adapter.wire == []
    conn = kbc.connect()
    try:
        assert not kbn.unseen_events_for_sub(conn, task_id=tid, platform="telegram", chat_id="42",
                                             kinds=["completed"])[1]
    finally:
        conn.close()
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    await replay_pending(new)
    await drain(fresh)
    await replay_pending(new)
    assert len(received) == 1
    assert "actual Kanban handoff" in received[0][0]


@pytest.mark.asyncio
async def test_background_process_injection_survives_restart(tmp_path):
    from unittest.mock import AsyncMock
    from gateway.restart_notifications import replay_pending

    old, adapter, source, key, sid, _ = route(tmp_path)
    old._process_notice_ownership = AsyncMock(return_value="deliver")
    old._build_process_event_source = lambda event: source
    old._resolve_injection_adapter = lambda platform, target: adapter
    assert await old._inject_watch_notification("process exited 0", {
        "type": "completion", "session_key": key, "parent_session_id": sid,
        "session_id": "process-test", "started_at": 1, "exit_code": 0,
    }) is True
    await drain(adapter)
    assert adapter.wire == []
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    await replay_pending(new)
    await drain(fresh)
    await replay_pending(new)
    assert len(received) == 1
    assert "process exited 0" in received[0][0]
    assert received[0][1]["gateway_session_id"] == sid
