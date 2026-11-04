"""Reconnect retries must not hide admitted humans from the turn drain (fork #61).

Only transport/model work is replaced. Admission, busy routing, natural backoff,
post-model selection, recursive follow-ups and depth-cap restoration are real.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.turn_context import TurnContext


class Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.DISCORD)
        self.processing = None

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, **kwargs):
        pass

    async def get_chat_info(self, chat_id):
        return {}

    async def on_processing_start(self, event):
        if self.processing is not None:
            await self.processing(event)


def install(runner):
    adapter = Adapter()
    adapter.gateway_runner = runner
    runner.adapters[Platform.DISCORD] = adapter
    runner._wire_adapter_handlers(adapter)
    return adapter


async def wait(event):
    await asyncio.wait_for(event.wait(), 15)


@pytest.mark.asyncio
@pytest.mark.parametrize("thread", [False, True], ids=["dm", "thread"])
@pytest.mark.parametrize("at_depth_cap", [False, True], ids=["recursive", "depth-cap"])
async def test_reconnect_retry_leaves_humans_visible_to_turn_drain(
    monkeypatch, tmp_path, thread, at_depth_cap,
):
    import gateway.run as run

    monkeypatch.setattr(run, "_hermes_home", tmp_path)
    runner = GatewayRunner(GatewayConfig())
    runner._busy_input_mode = runner._busy_text_mode = "queue"
    monkeypatch.setattr(runner, "_is_user_authorized_for_source", lambda source: True)
    agent = SimpleNamespace(
        steer=Mock(return_value=True), interrupt=Mock(), _active_children=[],
        get_activity_summary=lambda: {"seconds_since_activity": 0},
    )
    old = install(runner)

    def source(adapter, author="human-1"):
        return adapter.build_source(
            chat_id="room", chat_type="group" if thread else "dm",
            thread_id="thread" if thread else None, user_id=author, user_name=author,
        )

    src = source(old)
    key = runner._session_key_for_source(src)
    # The model stand-in still needs the real durable session that AIAgent would
    # open; otherwise the reaped-session guard correctly evicts its missing row.
    entry = runner.session_store.get_or_create_session(src)
    runner.session_store._db.create_session(entry.session_id, source="discord")
    entered, release, backed_off, admitted, first_exited, all_done = [asyncio.Event() for _ in range(6)]
    turns, model_tasks, backoffs = [], set(), []
    first_task = None

    async def model_turn(src, sk, generation, message_id, display_kind=None, depth=0):
        turns.append((message_id, display_kind, src.user_id))
        model_tasks.add(asyncio.current_task())
        runner._session_state(sk).turn.agent = agent
        if message_id == "first":
            entered.set()
            await release.wait()
            if at_depth_cap:
                depth = runner._MAX_INTERRUPT_DEPTH
        result = {"final_response": "", "messages": []}
        ctx = TurnContext(
            source=src, session_key=sk, session_id=entry.session_id, run_generation=generation,
            history=[], context_prompt="", _interrupt_depth=depth,
            persist_user_display_kind=display_kind,
        )
        adapter = runner._delivery_adapter_for(src)
        pending_event, pending = await runner._run_agent_drain_pending(result, adapter, src, sk)
        if pending_event or pending:
            result = await runner._run_agent_queued_followup(
                ctx, adapter, pending, pending_event, result, result, None,
            )
        if len(turns) == 5:
            all_done.set()
        return result

    async def inference(event, src, sk, generation):
        try:
            await model_turn(src, sk, generation, event.message_id,
                             "internal_notification" if event.internal else None)
        finally:
            if event.message_id == "first":
                first_exited.set()
        return None

    async def recursive_model(**kwargs):
        return await model_turn(
            kwargs["source"], kwargs["session_key"], kwargs["run_generation"],
            kwargs["inbound_message_id"], kwargs["persist_user_display_kind"],
            kwargs["_interrupt_depth"],
        )

    monkeypatch.setattr(runner, "_handle_message_with_agent", inference)
    monkeypatch.setattr(runner, "_run_agent", recursive_model)
    fresh = None
    try:
        await old.handle_message(MessageEvent(text="work", source=src, message_id="first"))
        await wait(entered)
        first_task = old._session_tasks[key]
        fresh = install(runner)
        fresh._busy_text_mode = "steer"  # no text debounce; runner still queues
        real_backoff = fresh._requeue_backoff_delay

        def observe(sk, pending, dispatched):
            delay = real_backoff(sk, pending, dispatched)
            backoffs.append(delay)
            if delay:
                backed_off.set()
            # A retry may observe the human, but must leave it available for the
            # original turn's drain. Do not replace the delay or the dispatch.
            if admitted.is_set() and pending.message_id == "h1":
                release.set()
            return delay

        async def slow_processing_marker(event):
            # A platform's processing marker can await network I/O. Force the old
            # turn to finish during that await if a retry stole h1 out of its slot.
            if event.message_id == "h1" and asyncio.current_task() is not first_task:
                release.set()
                await wait(first_exited)

        fresh.processing = slow_processing_marker
        monkeypatch.setattr(fresh, "_requeue_backoff_delay", observe)
        events = [
            MessageEvent(text=name, source=source(fresh, author), message_id=name, internal=internal)
            for name, internal, author in (
                ("n1", True, "machine"), ("n2", True, "machine"),
                ("h1", False, "human-1"), ("h2", False, "human-2"),
            )
        ]
        assert all(runner._session_key_for_source(event.source) == key for event in events)
        await fresh.handle_message(events[0])
        await wait(backed_off)
        for event in events[1:]:
            await fresh.handle_message(event)
        assert all(event._gateway_accepted for event in events)
        admitted.set()
        await wait(all_done)
        for task in model_tasks:
            await asyncio.wait_for(asyncio.shield(task), 15)
        # The retry keeper may still be sleeping after the original consumer drained
        # everything. Wait for its natural exit, rather than cancelling away a replay.
        keeper = fresh._session_tasks.get(key)
        if keeper is not None:
            await asyncio.wait_for(asyncio.shield(keeper), 15)
        assert turns == [
            ("first", None, "human-1"), ("h1", None, "human-1"), ("h2", None, "human-2"),
            ("n1", "internal_notification", "machine"), ("n2", "internal_notification", "machine"),
        ]
        assert any(backoffs)
        assert runner._queue_depth(key, adapter=fresh) == 0
        assert not old._pending_messages and not fresh._pending_messages
        assert not fresh._session_tasks and not fresh._active_sessions
        agent.steer.assert_not_called()
        agent.interrupt.assert_not_called()
    finally:
        release.set()
        first_exited.set()
        await old.cancel_background_tasks()
        if fresh is not None:
            await fresh.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["stale", "reaped", "cancel-keep", "cancel-discard"])
async def test_retry_preserves_eviction_and_command_cancellation(monkeypatch, tmp_path, finish):
    """A held retry must recover a dead owner and obey command discard/keep policy."""
    import gateway.run as run

    monkeypatch.setattr(run, "_hermes_home", tmp_path)
    runner = GatewayRunner(GatewayConfig())
    monkeypatch.setattr(runner, "_is_user_authorized_for_source", lambda source: True)
    adapter = install(runner)
    src = adapter.build_source(chat_id="room", chat_type="dm", user_id="user")
    key = runner._session_key_for_source(src)
    entry = runner.session_store.get_or_create_session(src)
    runner.session_store._db.create_session(entry.session_id, source="discord")
    activity = {"seconds_since_activity": 0}
    agent = SimpleNamespace(interrupt=Mock(), _active_children=[], get_activity_summary=lambda: activity)
    state = runner._session_state(key)
    state.turn.agent, state.turn.started_ts = agent, time.time()
    backed_off, delivered = asyncio.Event(), asyncio.Event()
    handled = []
    backoff = adapter._requeue_backoff_delay

    def observe(sk, pending, dispatched):
        delay = backoff(sk, pending, dispatched)
        if delay:
            backed_off.set()
        return delay

    async def model(event, source, sk, generation):
        handled.append(event.message_id)
        delivered.set()
        return None

    monkeypatch.setattr(adapter, "_requeue_backoff_delay", observe)
    monkeypatch.setattr(runner, "_handle_message_with_agent", model)
    queued = MessageEvent(text="notice", source=src, message_id="notice", internal=True)
    try:
        await adapter.handle_message(queued)
        await wait(backed_off)
        keeper = adapter._session_tasks[key]
        assert queued._gateway_accepted
        assert adapter._pending_messages[key] is queued
        assert not handled
        if finish == "stale":
            activity["seconds_since_activity"] = 1e9
        elif finish == "reaped":
            runner.session_store._db.end_session(entry.session_id, "test_reap")
        else:
            discard = finish == "cancel-discard"
            await adapter.cancel_session_processing(key, discard_pending=discard)
            state.turn.agent = None
            if not discard:
                assert adapter._pending_messages[key] is queued
            await adapter._drain_pending_after_session_command(key, asyncio.Event())
            keeper = adapter._session_tasks.get(key)
        if finish != "cancel-discard":
            await wait(delivered)
            # The legacy retry loop hands off repeatedly; wait for the task that
            # actually delivered, not an earlier keeper that has already returned.
            keeper = adapter._session_tasks.get(key)
        if keeper is not None:
            await asyncio.wait_for(asyncio.shield(keeper), 15)
        assert handled == ([] if finish == "cancel-discard" else ["notice"])
        assert not adapter._pending_messages and not adapter._active_sessions
        assert not adapter._session_tasks
        assert runner._queue_depth(key, adapter=adapter) == 0
        if finish in {"stale", "reaped"}:
            agent.interrupt.assert_called_once()
        else:
            agent.interrupt.assert_not_called()
    finally:
        await adapter.cancel_background_tasks()
