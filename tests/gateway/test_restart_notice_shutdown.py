"""Shutdown and topic-ownership regressions from independent restart-notice review."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from gateway.platforms.event import MessageEvent
from gateway.restart_notifications import replay_pending
from gateway.wake import deliver_wake
from hermes_state import AsyncSessionDB, SessionDB
from tests.gateway.test_kanban_wake_acceptance import drain
from tests.gateway.test_restart_internal_notifications import route


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["before_task_start", "profile_load", "drain_before_start"])
async def test_accepted_notice_survives_shutdown_before_execution(tmp_path, monkeypatch, stage):
    old, adapter, source, key, _, old_received = route(tmp_path, restarting=False)
    entered, release = asyncio.Event(), asyncio.Event()

    @asynccontextmanager
    async def loading_profile(home):
        entered.set()
        await release.wait()
        yield

    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_async_profile_runtime_scope", loading_profile)
    adapter.set_message_handler(old._make_default_profile_message_handler())
    if stage == "drain_before_start":
        event = MessageEvent(text="accepted before restart", source=source, internal=True)
        event._gateway_accepted = True
        adapter._active_sessions[key] = asyncio.Event()
        adapter._spawn_drain_task(event, key)
    else:
        await deliver_wake(adapter, text="accepted before restart", source=source)
    if stage == "profile_load":
        await asyncio.wait_for(entered.wait(), 2)
    old._draining = old._restart_requested = True
    await adapter.cancel_background_tasks()
    assert old_received == []
    assert adapter.wire == []
    assert adapter._restart_notification_tasks == {}

    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    await replay_pending(new)
    await drain(fresh)
    await replay_pending(new)
    await drain(fresh)
    assert [item[0] for item in received] == ["accepted before restart"]


@pytest.mark.asyncio
async def test_shutdown_does_not_replay_a_notice_that_already_executed(tmp_path):
    old, adapter, source, _, _, old_received = route(tmp_path, restarting=False)
    entered, release = asyncio.Event(), asyncio.Event()
    handler = adapter._message_handler

    async def executed_then_waiting(event):
        await handler(event)
        entered.set()
        await release.wait()

    adapter.set_message_handler(executed_then_waiting)
    await deliver_wake(adapter, text="already executed", source=source)
    await asyncio.wait_for(entered.wait(), 2)
    old._draining = old._restart_requested = True
    await adapter.cancel_background_tasks()
    assert [item[0] for item in old_received] == ["already executed"]
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    await replay_pending(new)
    await drain(fresh)
    assert received == []


@pytest.mark.asyncio
@pytest.mark.parametrize("ownership", ["restart", "strict", "delegation", "human"])
async def test_topic_binding_heal_cannot_replace_an_explicit_owner(tmp_path, ownership):
    runner, _, source, _, _, _ = route(tmp_path, restarting=False)
    source.thread_id = "10"
    entry = runner.session_store.get_or_create_session(source)
    key, pinned = entry.session_key, entry.session_id
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id=pinned, source="telegram", user_id="42")
    db.create_session(session_id="unrelated-session", source="telegram", user_id="42")
    db.bind_telegram_topic(chat_id="42", thread_id="10", user_id="42",
                          session_key=key, session_id="unrelated-session")
    assert db.get_compression_tip(pinned) == pinned
    runner._session_db = AsyncSessionDB(db)
    runner._cache_session_source = lambda session_key, source: None
    runner._is_telegram_topic_lane = lambda source: True
    runner._telegram_topic_profile_name = lambda source: "default"
    metadata: dict = {} if ownership == "human" else {
        "gateway_session_key": key, "gateway_session_id": pinned,
    }
    if ownership == "strict":
        metadata["gateway_session_strict"] = True
    event = MessageEvent(text="pinned", source=source, internal=ownership != "human", metadata=metadata)
    if ownership == "restart":
        event._restart_notification_id = "review-receipt"
    try:
        resolved = await runner._hmwa_resolve_session(event, source)
        if ownership == "human":
            assert resolved is not None
            assert resolved[1].session_id == "unrelated-session"
        else:
            assert resolved is None or resolved[1].session_id == pinned
            current = runner.session_store.lookup_by_session_key(key)
            assert current is not None and current.session_id == pinned
    finally:
        db.close()
