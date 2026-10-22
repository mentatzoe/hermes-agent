"""Execution and shutdown handoff must stay ordered across journal I/O awaits."""

import asyncio
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import restart_notifications as notices
from gateway.platforms.event import MessageEvent, ProcessingOutcome
from gateway.turn_context import TurnContext
from gateway.wake import deliver_wake
from tests.gateway.test_kanban_wake_acceptance import drain
from tests.gateway.test_restart_internal_notifications import route


@pytest.mark.asyncio
@pytest.mark.parametrize("production_turn", [False, True])
@pytest.mark.parametrize("handoff_await", ["lookup", "save"])
async def test_execution_and_preservation_have_one_owner(tmp_path, monkeypatch, production_turn, handoff_await):
    old, adapter, source, key, _, old_received = route(tmp_path, restarting=False)
    entered, execute, progressed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    io_entered, io_release = threading.Event(), threading.Event()
    original_lookup, original_save = old.session_store.lookup_by_session_key, notices._save

    async def handler(event):
        entered.set()
        await execute.wait()
        event._heartbeat_execution_started = True
        old_received.append(event.text)
        progressed.set()

    if production_turn:
        entry = original_lookup(key)
        prepared = old._PreparedTurn([], "", "executed during preservation", None, None, "notice")
        old._hmwa_resolve_session = AsyncMock(return_value=(source, entry, key))
        old._hmwa_prepare_turn = AsyncMock(return_value=(prepared, None))

        async def hook(*args):
            entered.set()
            await execute.wait()
            progressed.set()

        async def model(**kwargs):
            old_received.append(kwargs["message"])
            return {}

        old.hooks = SimpleNamespace(emit=hook)
        old._run_agent = model
        old._pinned_channel_inputs = lambda *args, **kwargs: (None, source)
        old._reply_anchor_for_event = lambda event: None
        old._hmwa_stop_typing_for_turn = AsyncMock()
        old._is_session_run_current = lambda *args: True
        old._hmwa_shape_agent_response = AsyncMock(return_value=(None, True, []))
        old._hmwa_prepend_reasoning = lambda *args: None
        old._hmwa_runtime_footer_line = lambda *args: None
        old._hmwa_post_turn_hooks = AsyncMock()
        old._hmwa_classify_turn_failure = lambda *args: (False, False, False)
        old._hmwa_compression_exhaustion_reset = AsyncMock(return_value=(None, entry))
        old._hmwa_persist_turn_transcript = AsyncMock()
        old._hmwa_deliver_turn_response = AsyncMock(return_value=None)
        old._clear_session_env = lambda tokens: None

        async def handler(event):
            return await old._handle_message_with_agent(event, event.source, key, 1)

    def blocked_lookup(session_key):
        io_entered.set()
        assert io_release.wait(10)
        return original_lookup(session_key)

    def blocked_save(*args):
        io_entered.set()
        assert io_release.wait(10)
        return original_save(*args)

    adapter.set_message_handler(handler)
    await deliver_wake(adapter, text="executed during preservation", source=source)
    await asyncio.wait_for(entered.wait(), 2)
    old._draining = old._restart_requested = True
    if handoff_await == "lookup":
        monkeypatch.setattr(old.session_store, "lookup_by_session_key", blocked_lookup)
    else:
        monkeypatch.setattr(notices, "_save", blocked_save)
    shutdown = asyncio.create_task(adapter.cancel_background_tasks())
    try:
        assert await asyncio.to_thread(io_entered.wait, 10)
        execute.set()
        await asyncio.wait_for(progressed.wait(), 2)
        await asyncio.sleep(0)
    finally:
        io_release.set()
    await shutdown
    await drain(adapter)
    assert adapter.wire == []
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    await notices.replay_pending(new)
    await drain(fresh)
    await notices.replay_pending(new)
    await drain(fresh)
    assert old_received + [item[0] for item in received] == ["executed during preservation"]


def test_restart_notifications_can_be_imported_before_platform_adapters():
    result = subprocess.run(
        [sys.executable, "-I", "-c", "import os, sys; sys.path.insert(0, os.getcwd()); "
         "import gateway.restart_notifications; import gateway.platforms.base_shutdown"],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_queued_followup_hands_off_if_restart_starts_during_preparation(tmp_path, monkeypatch):
    old, adapter, source, key, session_id, _ = route(tmp_path, restarting=False)
    entered, release = asyncio.Event(), asyncio.Event()
    phases = []

    async def cache_refresh(*args):
        entered.set()
        await release.wait()

    async def processing_hook(adapter, event, phase, *outcome):
        phases.append((phase, outcome))

    import gateway.run_turn_followup_ack as ack
    monkeypatch.setattr(ack, "_run_followup_processing_hook", processing_hook)
    old._refresh_agent_cache_message_count = cache_refresh
    old._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="late followup")
    old._is_goal_continuation_event = lambda event: False
    old._pinned_channel_inputs = lambda key, prompt, source, **kwargs: (prompt, source)
    old._reply_anchor_for_event = lambda event: None
    old._intake_adapter_for = lambda source: adapter
    old._run_agent = AsyncMock(return_value={})
    ctx = TurnContext(source=source, session_key=key, session_id=session_id,
                      run_generation=1, history=[], context_prompt="context")
    event = MessageEvent(text="late followup", source=source, internal=True)
    result = {"final_response": "done", "messages": [], "interrupted": True}
    task = asyncio.create_task(old._run_agent_queued_followup(
        ctx, adapter, event.text, event, "done", result, None))
    await asyncio.wait_for(entered.wait(), 2)
    old._draining = old._restart_requested = True
    release.set()
    assert await task is result
    old._run_agent.assert_not_awaited()
    assert phases == [("on_processing_start", ()),
                      ("on_processing_complete", (ProcessingOutcome.CANCELLED,))]
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    await notices.replay_pending(new)
    await drain(fresh)
    await notices.replay_pending(new)
    await drain(fresh)
    assert [item[0] for item in received] == ["late followup"]
