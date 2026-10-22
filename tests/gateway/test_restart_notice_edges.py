"""Race, ownership and queue-drain checks for restart notice retention."""

import asyncio
import copy
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.event import MessageEvent
from gateway.restart_notifications import replay_pending
from gateway.session import build_session_key
from gateway.wake import deliver_wake
from tests.gateway.test_kanban_wake_acceptance import drain
from tests.gateway.test_restart_internal_notifications import route


@pytest.mark.asyncio
async def test_restart_during_profile_scope_await_does_not_send_busy_reply(tmp_path):
    old, adapter, source, _, _, received = route(tmp_path, restarting=False)
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed(event):
        entered.set()
        await release.wait()
        _, response, _ = await old._hm_dispatch_quick_and_plugin_commands(event, event.source, None)
        return response

    adapter.set_message_handler(delayed)
    await deliver_wake(adapter, text="profile awaited", source=source)
    await asyncio.wait_for(entered.wait(), 2)
    old._draining = old._restart_requested = True
    release.set()
    await drain(adapter)
    assert adapter.wire == []
    assert received == []
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    await replay_pending(new)
    await drain(fresh)
    assert [row[0] for row in received] == ["profile awaited"]


@pytest.mark.asyncio
async def test_capacity_refusal_retains_replayed_notice_for_next_attempt(tmp_path):
    _, adapter, source, _, _, _ = route(tmp_path)
    await deliver_wake(adapter, text="capacity retry", source=source)
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    handler = fresh._message_handler
    fresh.set_message_handler(AsyncMock(return_value="too many sessions"))
    await replay_pending(new)
    await drain(fresh)
    assert fresh.wire == []
    fresh.set_message_handler(handler)
    await replay_pending(new)
    await drain(fresh)
    assert len(received) == 1


@pytest.mark.asyncio
async def test_stale_pending_snapshot_cannot_repeat_completed_notice(tmp_path, monkeypatch):
    from gateway import restart_notifications as notices

    _, adapter, source, _, _, _ = route(tmp_path)
    await deliver_wake(adapter, text="finish once", source=source)
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    entered, release = asyncio.Event(), asyncio.Event()
    original_handler = fresh._message_handler

    async def held(event):
        entered.set()
        await release.wait()
        return await original_handler(event)

    fresh.set_message_handler(held)
    await replay_pending(new)
    await asyncio.wait_for(entered.wait(), 2)
    snapshot_taken, return_snapshot = threading.Event(), threading.Event()
    original_pending = notices._pending

    def stale(path):
        rows = original_pending(path)
        snapshot_taken.set()
        assert return_snapshot.wait(5)
        return rows

    monkeypatch.setattr(notices, "_pending", stale)
    replay = asyncio.create_task(replay_pending(new))
    try:
        assert await asyncio.to_thread(snapshot_taken.wait, 5)
        release.set()
        await drain(fresh)
    finally:
        return_snapshot.set()
    await replay
    await drain(fresh)
    assert len(received) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("teardown", [False, True])
async def test_queued_head_and_fifo_tail_survive_restart_drain(tmp_path, teardown):
    old, adapter, source, key, _, _ = route(tmp_path, restarting=False)
    first = MessageEvent(text="queued head", source=source, internal=True)
    second = MessageEvent(text="queued tail", source=source, internal=True)
    adapter._pending_messages[key] = first
    old._session_state(key).conversation.queued_events.append(second)
    old._draining = old._restart_requested = True
    if teardown:
        await adapter.cancel_background_tasks()
    else:
        pending, text = await old._run_agent_drain_pending({"final_response": "done"}, adapter, source, key)
        assert pending is None and text is None
    assert key not in adapter._pending_messages
    assert not old._overflow_queue(key)
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    await replay_pending(new)
    await drain(fresh)
    await replay_pending(new)
    await drain(fresh)
    assert [row[0] for row in received] == ["queued head", "queued tail"]


@pytest.mark.asyncio
async def test_relay_provenance_is_restored_not_replaced_by_native_transport(tmp_path):
    old, adapter, source, _, _, _ = route(tmp_path)
    source.delivered_via_upstream_relay = True
    await deliver_wake(adapter, text="relay result", source=source)
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    original_delivery = new._delivery_adapter_for
    new._delivery_adapter_for = lambda restored: original_delivery(restored) if restored.delivered_via_upstream_relay else None
    await replay_pending(new)
    await drain(fresh)
    assert len(received) == 1


@pytest.mark.asyncio
async def test_unavailable_first_page_does_not_starve_later_destination(tmp_path):
    from gateway import restart_notifications as notices

    old, adapter, source, _, _, _ = route(tmp_path)
    lost = copy.copy(source)
    lost.chat_id = "offline"
    old.session_store.get_or_create_session(lost)
    await deliver_wake(adapter, text="offline result", source=lost)
    path = tmp_path / "restart_notifications" / "pending.sqlite3"
    payload = notices._pending(path)[0][1]
    for index in range(100):
        notices._save(path, f"offline-{index}", payload)
    await deliver_wake(adapter, text="reachable result", source=source)
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    new._delivery_adapter_for = lambda restored: fresh if restored.chat_id == "42" else None
    await replay_pending(new)
    await drain(fresh)
    assert [row[0] for row in received] == ["reachable result"]


@pytest.mark.asyncio
@pytest.mark.parametrize("end_reason", ["idle", "new", "compression"])
async def test_real_session_resolver_never_retargets_to_idle_or_user_replacement(tmp_path, end_reason):
    runner, _, source, key, sid, _ = route(tmp_path, restarting=False)
    entry = runner.session_store.lookup_by_session_key(key)
    entry.session_id = "replacement"
    runner._async_session_store = SimpleNamespace(_store=runner.session_store,
        lookup_by_session_key=AsyncMock(return_value=entry))
    runner._session_db = SimpleNamespace(get_session=AsyncMock(return_value={"ended_at": 1, "end_reason": end_reason}))
    runner._resolve_compression_lineage_target = AsyncMock(return_value="replacement")
    runner._cache_session_source = lambda key, source: None
    runner._is_telegram_topic_lane = lambda source: False
    runner._resolve_async_delegation_session = AsyncMock(side_effect=AssertionError("generic retarget must not run"))
    event = MessageEvent(text="pinned", source=source, internal=True,
                         metadata={"gateway_session_key": key, "gateway_session_id": sid})
    event._restart_notification_id = "trusted-journal-receipt"
    resolved = await runner._hmwa_resolve_session(event, source)
    if end_reason == "compression":
        assert resolved[1].session_id == "replacement"
    else:
        assert resolved is None
    runner._resolve_async_delegation_session.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("restart_during_delivery", [False, True])
async def test_process_watcher_sends_no_concise_receipt_during_restart(tmp_path, monkeypatch, restart_during_delivery):
    import tools.process_registry as registry
    from tests.gateway.test_background_process_notifications import _FakeRegistry, _build_runner, _watcher_dict

    session = SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0,
                              command="echo done", started_at=None)
    monkeypatch.setattr(registry, "process_registry", _FakeRegistry([session], consumed=False))
    runner = _build_runner(monkeypatch, tmp_path, "concise")
    runner._draining = not restart_during_delivery
    runner._launching_turn_active = AsyncMock(return_value=True)
    async def deliver(*args):
        runner._draining = True
        return True
    runner._enqueue_process_completion_notification = AsyncMock(side_effect=deliver)
    watcher = {**_watcher_dict(), "check_interval": 0, "notify_on_complete": True}
    await runner._run_process_watcher(watcher)
    runner._enqueue_process_completion_notification.assert_awaited_once()
    from gateway.config import Platform
    runner.adapters[Platform.TELEGRAM].send.assert_not_awaited()
