"""A notice remains owned after dequeue and through follow-up preparation."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway import restart_notifications as notices
from gateway.turn_context import TurnContext
from gateway.wake import deliver_wake
from tests.gateway.test_kanban_wake_acceptance import drain
from tests.gateway.test_restart_internal_notifications import route


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["prepare", "cache", "first_response", "voice"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_dequeued_notice_survives_preparation_shutdown(tmp_path, stage, cancel):
    old, adapter, source, key, sid, _ = route(tmp_path, restarting=False)
    initial_entered, queued, paused, release = (asyncio.Event() for _ in range(4))
    original_work = []

    async def pause():
        paused.set()
        await release.wait()

    async def prepare(**kwargs):
        if stage == "prepare":
            await pause()
            return None  # Refusal must not erase a notice during restart.
        return kwargs["event"].text

    async def cache(*args):
        if stage == "cache":
            await pause()

    async def deliver_first(*args):
        if stage == "first_response":
            await pause()

    async def transcribe(event, *args, **kwargs):
        await pause()
        return event.text, None

    old._prepare_profile_scoped_inbound_message_text = prepare
    old._refresh_agent_cache_message_count = cache
    old._run_agent_deliver_first_response = deliver_first
    if stage == "voice":
        old._pending_event_audio_paths = lambda event: ["synthetic.wav"]
        old._transcribe_and_echo_pending_voice = transcribe
    old._is_goal_continuation_event = lambda event: False
    old._pinned_channel_inputs = lambda key, prompt, source, **kwargs: (prompt, source)
    old._reply_anchor_for_event = lambda event: None
    old._intake_adapter_for = lambda source: adapter
    old._run_agent = AsyncMock(return_value={})
    ctx = TurnContext(source=source, session_key=key, session_id=sid,
                      run_generation=1, history=[], context_prompt="context")
    result = {"final_response": "done", "messages": [], "interrupted": stage != "first_response"}

    async def handler(event):
        assert await notices.begin_execution(old, event, key)
        original_work.append(event.text)
        await notices.complete_notice(old, event)
        initial_entered.set()
        await queued.wait()
        pending_event, pending = await old._run_agent_drain_pending(result, adapter, source, key)
        if pending_event is not None:
            await old._run_agent_queued_followup(ctx, adapter, pending, pending_event, "done", result, None)

    adapter.set_message_handler(handler)
    adapter.set_busy_session_handler(old._handle_active_session_busy_message)
    await deliver_wake(adapter, text="initial turn", source=source)
    await asyncio.wait_for(initial_entered.wait(), 2)
    await deliver_wake(adapter, text="accepted queued notice", source=source)
    accepted = adapter._pending_messages[key]
    assert accepted._gateway_accepted
    queued.set()
    await asyncio.wait_for(paused.wait(), 2)
    assert key not in adapter._pending_messages
    assert not old._overflow_queue(key)
    assert not getattr(accepted, "_heartbeat_execution_started", False)
    old._draining = old._restart_requested = True
    if not cancel:
        release.set()
        await drain(adapter)
        # Completion callbacks have run before teardown inspects ownership.
        await asyncio.sleep(0)
    await adapter.cancel_background_tasks()
    old._run_agent.assert_not_awaited()
    assert original_work == ["initial turn"]
    assert adapter.wire == []
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    await notices.replay_pending(new)
    await drain(fresh)
    await notices.replay_pending(new)
    await drain(fresh)
    assert [item[0] for item in received] == ["accepted queued notice"]
