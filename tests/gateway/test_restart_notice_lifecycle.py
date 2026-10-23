"""One journal owns admitted notices across every restart cut (aleph-vault#241)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import restart_notifications as notices
from gateway.turn_context import TurnContext
from gateway.wake import deliver_wake
from tests.gateway.test_kanban_wake_acceptance import drain
from tests.gateway.test_restart_internal_notifications import route


def production_turns(runner, adapter, source, key, sid, pause, worker):
    """Keep real entrypoint, worker-return, drain and recursion; replace external work."""
    entry = runner.session_store.lookup_by_session_key(key)
    runner._hmwa_resolve_session = AsyncMock(return_value=(source, entry, key))

    async def prepare(event, *args):
        await pause("prepare", event.text)
        return runner._PreparedTurn([], "context", event.text, None, None, "internal_notification"), None

    async def hook(*args):
        await pause("hook", "original")

    async def inbound(**kwargs):
        await pause("followup_prepare", kwargs["event"].text)
        return kwargs["event"].text

    async def cache(*args):
        await pause("cache", "followup")

    async def first_response(*args):
        await pause("first_response", "followup")

    runner._hmwa_prepare_turn = prepare
    runner.hooks = SimpleNamespace(emit=hook)
    runner._prepare_profile_scoped_inbound_message_text = inbound
    runner._refresh_agent_cache_message_count = cache
    runner._run_agent_deliver_first_response = first_response
    runner._is_goal_continuation_event = lambda event: False
    runner._pinned_channel_inputs = lambda key, prompt, source, **kwargs: (prompt, source)
    runner._reply_anchor_for_event = lambda event: None
    runner._intake_adapter_for = lambda source: adapter
    runner._clear_session_env = lambda tokens: None
    runner._get_proxy_url = lambda: None
    runner._run_agent_display_settings = lambda source: SimpleNamespace(
        needs_progress_queue=False, log_mode_enabled=False, _native_slack_task_cards=False)

    def context(*args, **kwargs):
        ctx = TurnContext(source=source, session_key=key, session_id=sid,
                          run_generation=1, history=[], context_prompt="context")
        ctx.message = kwargs["message"]
        ctx.mute_notification_reply = True
        return ctx, SimpleNamespace(run_sync=lambda: None), adapter

    runner._run_agent_build_turn_context = context
    runner._run_agent_bind_turn_wiring = lambda *args: None
    runner._run_agent_stream_consumer_task = AsyncMock()
    runner._run_agent_track_agent = AsyncMock()
    runner._run_agent_monitor_for_interrupt = AsyncMock()
    worker_future = []

    def start_worker(*args):
        future = asyncio.get_running_loop().create_future()
        worker_future.append(future)
        return SimpleNamespace(executor_task=future)

    runner._run_agent_start_turn_worker = start_worker

    async def await_worker(_worker, ctx, *args):
        result = await worker(ctx.message)
        if result is None:
            result = {"final_response": "done", "messages": [], "completed": True, "interrupted": False}
        ctx.result_holder[0] = result
        _worker.executor_task.set_result(result)
        return result

    runner._run_agent_await_turn_worker = await_worker
    runner._run_agent_evict_on_fallback = lambda *args: None
    runner._run_agent_finalize_streaming_tts = AsyncMock()
    runner._run_agent_cleanup_turn_tasks = AsyncMock()
    runner._run_agent_mark_streamed_delivery = AsyncMock()
    runner._run_agent_schedule_bubble_cleanup = lambda *args: None
    runner._run_agent = AsyncMock(side_effect=runner._run_agent_inner)
    runner._hmwa_stop_typing_for_turn = AsyncMock()
    runner._is_session_run_current = lambda *args: True
    runner._hmwa_shape_agent_response = AsyncMock(return_value=(None, True, []))
    runner._hmwa_prepend_reasoning = lambda *args: None
    runner._hmwa_runtime_footer_line = lambda *args: None
    runner._hmwa_post_turn_hooks = AsyncMock()
    runner._hmwa_classify_turn_failure = lambda *args: (False, False, False)
    runner._hmwa_compression_exhaustion_reset = AsyncMock(return_value=(None, entry))
    runner._hmwa_persist_turn_transcript = AsyncMock()
    runner._hmwa_deliver_turn_response = AsyncMock(return_value=None)
    adapter.set_message_handler(lambda event: runner._handle_message_with_agent(event, event.source, key, 1))
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)


@pytest.mark.asyncio
@pytest.mark.parametrize("cut", [
    "admitted", "queued", "prepare", "hook", "dequeued", "followup_prepare",
    "first_response", "cache", "later_turn", "finished_unmarked", "marked",
])
async def test_restart_cuts_follow_only_durable_completion(tmp_path, monkeypatch, cut):
    old, adapter, source, key, sid, _ = route(tmp_path, restarting=False)
    reached, release, worker_entered, finish_worker = (asyncio.Event() for _ in range(4))
    worked = []
    originals = []
    expected_replay = ["original"]

    async def pause(stage, text):
        if stage == cut:
            reached.set()
            await release.wait()

    async def worker(text):
        worker_entered.set()
        if cut in {"queued", "dequeued", "followup_prepare", "first_response", "cache", "later_turn"}:
            if text == "original":
                await finish_worker.wait()
            elif cut == "later_turn":
                reached.set()
                await release.wait()
        worked.append(text)

    production_turns(old, adapter, source, key, sid, pause, worker)
    original_complete = notices.complete_notice

    async def complete(runner, event):
        if runner is old and event is not None and event.text == "original" and cut == "finished_unmarked":
            reached.set()
            await release.wait()
        await original_complete(runner, event)
        if runner is old and event is not None and event.text == "original" and cut == "marked":
            reached.set()
            await release.wait()

    monkeypatch.setattr(notices, "complete_notice", complete)
    original_dequeue = old._promote_queued_event

    def dequeue(*args):
        event = original_dequeue(*args)
        if cut == "dequeued" and event is not None:
            old._draining = old._restart_requested = True
            reached.set()
        return event

    old._promote_queued_event = dequeue
    await deliver_wake(adapter, text="original", source=source)
    if cut == "admitted":
        # Admission itself must already own a committed record, before any task can run.
        originals = notices._pending(tmp_path / "restart_notifications" / "pending.sqlite3")
        assert [payload["text"] for _, payload in originals] == ["original"]
    elif cut in {"queued", "dequeued", "followup_prepare", "first_response", "cache", "later_turn"}:
        await asyncio.wait_for(worker_entered.wait(), 2)
        await deliver_wake(adapter, text="followup", source=source)
        # Both are durable while the first worker is still running, not only at teardown.
        rows = notices._pending(tmp_path / "restart_notifications" / "pending.sqlite3")
        assert [payload["text"] for _, payload in rows] == ["original", "followup"]
        if cut == "queued":
            expected_replay = ["original", "followup"]
        else:
            expected_replay = ["followup"]
            finish_worker.set()
            await asyncio.wait_for(reached.wait(), 2)
    else:
        await asyncio.wait_for(reached.wait(), 2)
        if cut == "marked":
            expected_replay = []

    old._draining = old._restart_requested = True
    await adapter.cancel_background_tasks()
    assert adapter.wire == []
    rows = notices._pending(tmp_path / "restart_notifications" / "pending.sqlite3")
    assert [payload["text"] for _, payload in rows] == expected_replay
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    for _ in range(4):
        await notices.replay_pending(new)
        await drain(fresh)
    assert [row[0] for row in received] == expected_replay
    third, third_adapter, _, _, _, third_received = route(tmp_path, restarting=False)
    await notices.replay_pending(third)
    await drain(third_adapter)
    assert third_received == []
    if cut == "finished_unmarked":
        assert worked == ["original"]  # The documented, deliberately retryable window.


@pytest.mark.asyncio
async def test_completed_intermediate_notice_is_independent_of_later_turn(tmp_path):
    old, adapter, source, key, sid, _ = route(tmp_path, restarting=False)
    started, finish, paused, release = (asyncio.Event() for _ in range(4))
    worked = []

    async def pause(stage, text):
        if stage == "followup_prepare" and text == "third":
            paused.set()
            await release.wait()

    async def worker(text):
        if text == "original":
            started.set()
            await finish.wait()
        worked.append(text)

    production_turns(old, adapter, source, key, sid, pause, worker)
    await deliver_wake(adapter, text="original", source=source)
    await asyncio.wait_for(started.wait(), 2)
    await deliver_wake(adapter, text="intermediate", source=source)
    await deliver_wake(adapter, text="third", source=source)
    finish.set()
    await asyncio.wait_for(paused.wait(), 2)
    old._draining = old._restart_requested = True
    await adapter.cancel_background_tasks()
    assert worked == ["original", "intermediate"]
    rows = notices._pending(tmp_path / "restart_notifications" / "pending.sqlite3")
    assert [payload["text"] for _, payload in rows] == ["third"]
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    for _ in range(3):
        await notices.replay_pending(new)
        await drain(fresh)
    assert [row[0] for row in received] == ["third"]


@pytest.mark.asyncio
async def test_interrupted_worker_return_does_not_mark_notice_completed(tmp_path):
    old, adapter, source, key, sid, _ = route(tmp_path, restarting=False)

    async def pause(*args):
        pass

    async def worker(text):
        return {"final_response": "interrupted", "messages": [], "interrupted": True}

    production_turns(old, adapter, source, key, sid, pause, worker)
    await deliver_wake(adapter, text="unfinished", source=source)
    await drain(adapter)
    rows = notices._pending(tmp_path / "restart_notifications" / "pending.sqlite3")
    assert [payload["text"] for _, payload in rows] == ["unfinished"]
    assert not old._restart_notification_inflight
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    await notices.replay_pending(new)
    await drain(fresh)
    assert [row[0] for row in received] == ["unfinished"]


@pytest.mark.asyncio
async def test_marked_row_wins_over_stale_snapshot_without_local_cache(tmp_path, monkeypatch):
    _, adapter, source, _, _, _ = route(tmp_path)
    await deliver_wake(adapter, text="marked elsewhere", source=source)
    path = tmp_path / "restart_notifications" / "pending.sqlite3"
    snapshot = notices._pending(path)
    notices._complete(path, snapshot[0][0])
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    monkeypatch.setattr(notices, "_pending", lambda path: snapshot)
    await notices.replay_pending(new)
    await drain(fresh)
    assert received == []


@pytest.mark.asyncio
async def test_notice_journal_does_not_also_restore_human_transcript_spool(tmp_path, monkeypatch):
    from gateway.platforms.event import MessageEvent
    from gateway.shutdown_flush import flush_overflow_to_file, flush_pending_to_file, recover_pending_spool
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    old, adapter, source, key, sid, _ = route(tmp_path, restarting=False)
    adapter._active_sessions[key] = asyncio.Event()
    adapter.set_busy_session_handler(old._handle_active_session_busy_message)
    await deliver_wake(adapter, text="journal head", source=source)
    await deliver_wake(adapter, text="journal tail", source=source)
    assert flush_pending_to_file(adapter._pending_messages) == 0
    assert flush_overflow_to_file({key: old._overflow_queue(key)}) == 0
    human = MessageEvent(text="human pending", source=source)
    assert flush_pending_to_file({key: human}) == 1
    old._draining = old._restart_requested = True
    await adapter.cancel_background_tasks()
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session(session_id=sid, source="telegram", user_id="42")
        recovered, held = recover_pending_spool(db, session_resolver=lambda key, **kwargs: (sid, db))
        assert recovered == 1 and not held
        assert [row["content"] for row in db.get_messages(sid)] == ["human pending"]
    finally:
        db.close()
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    for _ in range(3):
        await notices.replay_pending(new)
        await drain(fresh)
    assert [row[0] for row in received] == ["journal head", "journal tail"]
