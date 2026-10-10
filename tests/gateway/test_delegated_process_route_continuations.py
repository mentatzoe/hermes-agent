"""Follow-up regressions from independent route-fork review."""
import pytest

from tests.gateway.test_delegated_process_route_isolation import harness, event, KEY
from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["switch", "compression"])
async def test_adopted_child_delegation_follows_later_user_switch(harness, transition):
    runner, db, store, adapter, turns = harness
    store.switch_session(KEY, "child")
    if transition == "switch":
        db.create_session("other", source="discord")
        store.switch_session(KEY, "other")
        assert db.get_session("child")["end_reason"] == "session_switch"
    else:
        db.end_session("child", end_reason="compression")
        db.create_session("other", source="discord", parent_session_id="child")
        store.advance_compression_session(KEY, "child", "other")
        assert db.get_compression_tip("child") == "other"
    assert await runner._classify_completion_target("child") == "deliver"
    result = await runner._deliver_completion_notification(
        "adopted child delegation result", event(owner="child", kind="async_delegation")
    )
    assert result is True
    assert turns == [("other", "adopted child delegation result")]
    assert store.lookup_by_session_key(KEY).session_id == "other"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason,verdict", [("session_switch", "deliver"), ("session_reset", "terminal"), ("user_exit", "terminal")])
async def test_compressed_parent_follows_tip_lifecycle(harness, reason, verdict):
    runner, db, store, adapter, turns = harness
    db.end_session("home", end_reason="compression")
    db.create_session("tip", source="discord", parent_session_id="home")
    store.advance_compression_session(KEY, "home", "tip")
    store.switch_session(KEY, "child")
    if reason != "session_switch":
        db.reopen_session("tip")
        db.end_session("tip", end_reason=reason)
    assert db.get_session("tip")["end_reason"] == reason
    assert db.get_compression_tip("home") == "tip"
    assert await runner._classify_completion_target("home") == verdict
    result = await runner._deliver_completion_notification(
        "compressed parent result", event(owner="home", kind="async_delegation")
    )
    if verdict == "deliver":
        assert result is True
        assert turns == [("child", "compressed parent result")]
    else:
        assert result is None
        assert turns == []


@pytest.mark.asyncio
async def test_producer_watch_event_stays_child_owned_after_record_pruned(harness, monkeypatch):
    runner, db, store, adapter, turns = harness
    registry = ProcessRegistry()
    monkeypatch.setattr("tools.process_registry.process_registry", registry)
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    child_process = ProcessSession(
        id="proc_child_review", command="synthetic-only", task_id="default",
        owner_task_id="sa-0-deadbeef", parent_session_id="child", session_key=KEY,
    )
    child_process.watch_patterns = ["ready"]
    child_process.watcher_platform = "discord"
    child_process.watcher_chat_id = "123"
    registry._running[child_process.id] = child_process
    registry._check_watch_patterns(child_process, "ready\n")
    evt = registry.completion_queue.get_nowait()
    assert evt["owner_task_id"] == "sa-0-deadbeef"
    registry._running.pop(child_process.id)
    child_process.exited = True
    registry._finished[child_process.id] = child_process
    child_process.started_at = 0
    with registry._lock:
        registry._prune_if_needed()
    assert registry.get(child_process.id) is None
    assert await runner._inject_watch_notification("child watch output", evt) is None
    adapter.handle_message.assert_not_awaited()
    assert turns == []


@pytest.mark.asyncio
async def test_legacy_event_retains_explicit_subagent_owner(harness):
    runner, db, store, adapter, turns = harness
    evt = event(kind="watch_match")
    evt.pop("parent_session_id")
    evt.update(task_id="default", owner_task_id="sa-0-deadbeef")
    assert await runner._inject_watch_notification("old child watch", evt) is None
    assert turns == []


@pytest.mark.asyncio
async def test_unadopted_child_async_delivery_is_not_acknowledged(harness):
    runner, db, store, adapter, turns = harness
    assert await runner._classify_completion_target("child") == "terminal"
    assert await runner._deliver_completion_notification("child result", event(kind="async_delegation")) is None
    adapter.handle_message.assert_not_awaited()
    assert store.lookup_by_session_key(KEY).session_id == "home"
