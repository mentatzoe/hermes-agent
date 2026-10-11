"""Regression for a child process notice taking over its parent's DM (7 Oct)."""
from collections import OrderedDict
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
import asyncio
import threading

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import AsyncSessionStore, SessionEntry, SessionSource, SessionStore
from hermes_state import AsyncSessionDB, SessionDB

KEY = "agent:main:discord:dm:123"


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("home", source="discord")
    # Source can be rewritten when a child is accidentally adopted by a gateway.
    # The durable delegate marker survives and is not a compression-parent link.
    db.create_session("child", source="discord", parent_session_id="home",
                      model_config={"_delegate_from": "home"})
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    store._db = db
    store._ensure_loaded()
    source = SessionSource(platform=Platform.DISCORD, chat_type="dm", chat_id="123")
    entry = SessionEntry(session_key=KEY, session_id="home", created_at=datetime.now(),
                         updated_at=datetime.now(), platform=Platform.DISCORD,
                         chat_type="dm", origin=source)
    store._entries[KEY] = entry
    store._save()
    runner = object.__new__(GatewayRunner)
    runner._session_db = AsyncSessionDB(db)
    runner.config = GatewayConfig()
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._session_source_cache = {}
    runner._completion_delivery_lock = threading.Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 20
    turns = []

    async def accept(event):
        current = store.lookup_by_session_key(KEY)
        pin = event.metadata.get("gateway_session_id")
        resolved = await runner._resolve_async_delegation_session(current, pin) if pin else current
        if resolved:
            turns.append((resolved.session_id, event.text))

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=accept))
    runner.adapters = {Platform.DISCORD: adapter}
    yield runner, db, store, adapter, turns
    db.close()


def event(owner="child", kind="completion", process="proc_child"):
    return dict(type=kind, session_id=process, session_key=KEY, platform="discord",
                chat_type="dm", chat_id="123", parent_session_id=owner,
                started_at=1.0, exit_code=0, completion_reason="exited")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["completion", "watch_match", "watch_disabled"])
@pytest.mark.parametrize("adopted", [False, True])
async def test_child_notice_cannot_start_gateway_turn(harness, kind, adopted):
    runner, db, store, adapter, turns = harness
    if adopted:
        store.switch_session(KEY, "child")
    before = store.lookup_by_session_key(KEY).session_id
    before_home = db.get_session("home")
    result = await runner._inject_watch_notification("child output", event(kind=kind))
    assert result is None
    adapter.handle_message.assert_not_awaited()
    assert turns == []
    assert store.lookup_by_session_key(KEY).session_id == before
    assert db.get_session("home")["ended_at"] == before_home["ended_at"]


@pytest.mark.asyncio
async def test_resolver_cannot_pin_route_to_delegate_child(harness):
    runner, db, store, adapter, turns = harness
    result = await runner._resolve_async_delegation_session(store.lookup_by_session_key(KEY), "child")
    assert result is None
    assert store.lookup_by_session_key(KEY).session_id == "home"
    assert db.get_session("home")["ended_at"] is None


@pytest.mark.asyncio
async def test_route_switch_does_not_discard_parent_delegation(harness):
    runner, db, store, adapter, turns = harness
    store.switch_session(KEY, "child")
    assert db.get_session("home")["end_reason"] == "session_switch"
    assert await runner._classify_completion_target("home") == "deliver"
    result = await runner._deliver_completion_notification("batch result", event(owner="home", kind="async_delegation"))
    assert result is True
    assert turns == [("child", "batch result")]
    assert store.lookup_by_session_key(KEY).session_id == "child"
    assert db.get_session("home")["end_reason"] == "session_switch"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["session_reset", "new_session", "user_exit"])
async def test_real_user_close_remains_terminal(harness, reason):
    runner, db, store, adapter, turns = harness
    db.end_session("home", end_reason=reason)
    assert await runner._classify_completion_target("home") == "terminal"
    result = await runner._deliver_completion_notification("stale batch", event(owner="home", kind="async_delegation"))
    assert result is None
    assert turns == []


@pytest.mark.asyncio
async def test_child_notice_does_not_leak_through_parent_batch(harness):
    runner, db, store, adapter, turns = harness
    await asyncio.gather(
        runner._enqueue_process_completion_notification("parent output", event(owner="home", process="proc_home")),
        runner._enqueue_process_completion_notification("child output", event()),
    )
    assert turns == [("home", "parent output")]
    assert store.lookup_by_session_key(KEY).session_id == "home"
    assert db.get_session("home")["ended_at"] is None


@pytest.mark.asyncio
async def test_compression_child_is_not_a_delegate(harness):
    runner, db, store, adapter, turns = harness
    db.end_session("home", end_reason="compression")
    db.create_session("compressed", source="discord", parent_session_id="home")
    store.advance_compression_session(KEY, "home", "compressed")
    assert await runner._inject_watch_notification("ordinary output", event(owner="compressed")) is True
    assert turns == [("compressed", "ordinary output")]
