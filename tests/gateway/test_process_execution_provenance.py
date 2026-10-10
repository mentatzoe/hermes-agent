"""Execution provenance must survive a worker transcript becoming a user chat."""
import pytest
from tests.gateway.test_delegated_process_route_isolation import harness, event, KEY


@pytest.mark.asyncio
async def test_top_level_process_in_adopted_chat_still_delivers(harness):
    runner, db, store, adapter, turns = harness
    store.switch_session(KEY, "child")
    evt = event()
    evt["delegated_child"] = False
    assert await runner._inject_watch_notification("new main process", evt) is True
    assert turns == [("child", "new main process")]


@pytest.mark.asyncio
async def test_explicit_child_execution_drops_without_session_marker(harness):
    runner, db, store, adapter, turns = harness
    evt = event(owner="home")
    evt["delegated_child"] = True
    assert await runner._inject_watch_notification("worker process", evt) is None
    adapter.handle_message.assert_not_awaited()
    assert turns == []


@pytest.mark.asyncio
async def test_watcher_completion_keeps_main_execution_after_registry_prune(harness):
    from tools.process_registry import ProcessSession
    runner, db, store, adapter, turns = harness
    store.switch_session(KEY, "child")
    session = ProcessSession(id="proc_pruned", command="synthetic", task_id="default",
        owner_task_id="main-turn", delegated_child=False, parent_session_id="child", exit_code=0)
    evt = runner._build_process_completion_event(event(), session, session.id)
    assert evt["owner_task_id"] == "main-turn"
    assert evt["delegated_child"] is False
    assert await runner._inject_watch_notification("main output", evt) is True
