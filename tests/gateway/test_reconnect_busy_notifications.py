"""A replaced adapter must not turn machine notices into human steering.

Regression for mentatzoe/aleph-vault#285. Transport and inference are local
stand-ins; adapter admission, runner busy routing and follow-up dispatch are real.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from gateway.session import SessionSource


class Adapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        pass

    async def get_chat_info(self, chat_id):
        return {}


def make_runner(monkeypatch, tmp_path, mode="steer", thread=False):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = GatewayRunner(GatewayConfig())
    runner._busy_input_mode = mode
    monkeypatch.setattr(runner, "_is_user_authorized_for_source", lambda source: True)
    source = SessionSource(
        platform=Platform.DISCORD, chat_id="room", chat_type="group" if thread else "dm",
        thread_id="thread" if thread else None, user_id="operator", user_name="Operator",
    )
    agent = SimpleNamespace(
        steer=Mock(return_value=True), interrupt=Mock(), _active_children=[],
        get_activity_summary=lambda: {"seconds_since_activity": 0},
    )
    return runner, source, runner._session_key_for_source(source), agent


def install_adapter(runner):
    adapter = Adapter(PlatformConfig(enabled=True), Platform.DISCORD)
    adapter.set_message_handler(runner._handle_message)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    runner.adapters[Platform.DISCORD] = adapter
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("thread", [False, True], ids=["dm", "thread"])
@pytest.mark.parametrize("mode,internal", [
    ("steer", True), ("interrupt", True), ("queue", True), ("steer", False),
])
async def test_replaced_adapter_queues_notices_but_still_steers_human_text(
    monkeypatch, tmp_path, mode, internal, thread,
):
    runner, source, key, agent = make_runner(monkeypatch, tmp_path, mode, thread)
    old_adapter = install_adapter(runner)
    first_started, release_first = asyncio.Event(), asyncio.Event()
    routed, allow_drain, followup_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    turns = []

    async def inference(event, source, session_key, generation):
        turns.append(event)
        runner._session_state(session_key).turn.agent = agent
        if event.message_id == "first":
            first_started.set()
            await release_first.wait()
        else:
            followup_started.set()
        return None

    monkeypatch.setattr(runner, "_handle_message_with_agent", inference)
    first = MessageEvent(text="work", source=source, message_id="first")
    incoming = MessageEvent(text="background result" if internal else "a correction",
                            source=source, message_id="incoming", internal=internal)
    fresh_adapter = None
    try:
        await old_adapter.handle_message(first)
        await asyncio.wait_for(first_started.wait(), 5)
        first_task = old_adapter._session_tasks[key]
        fresh_adapter = install_adapter(runner)
        assert key in old_adapter._active_sessions
        assert key not in fresh_adapter._active_sessions
        assert runner._is_session_running(key)

        async def incoming_handler(event):
            response = await runner._handle_message(event)
            if not routed.is_set():
                routed.set()
                await allow_drain.wait()
            return response

        fresh_adapter.set_message_handler(incoming_handler)
        await fresh_adapter.handle_message(incoming)
        await asyncio.wait_for(routed.wait(), 5)
        agent.interrupt.assert_not_called()
        assert turns == [first]
        if internal:
            agent.steer.assert_not_called()
            assert fresh_adapter._pending_messages[key] is incoming
            assert incoming._gateway_accepted is True
        else:
            agent.steer.assert_called_once()
            assert "a correction" in agent.steer.call_args.args[0]
            assert key not in fresh_adapter._pending_messages

        release_first.set()
        await asyncio.wait_for(asyncio.shield(first_task), 5)
        allow_drain.set()
        if internal:
            await asyncio.wait_for(followup_started.wait(), 5)
            await asyncio.wait_for(asyncio.shield(fresh_adapter._session_tasks[key]), 5)
            assert turns == [first, incoming]
            assert turns[1].internal is True
            assert turns[1].source is source
            assert turns[1].text == "background result"
            assert runner._queue_depth(key, adapter=fresh_adapter) == 0
            agent.steer.assert_not_called()
    finally:
        release_first.set()
        allow_drain.set()
        await old_adapter.cancel_background_tasks()
        if fresh_adapter is not None:
            await fresh_adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_internal_command_text_cannot_stop_a_pending_turn(monkeypatch, tmp_path):
    runner, source, key, _ = make_runner(monkeypatch, tmp_path)
    adapter = install_adapter(runner)
    runner._session_state(key).turn.agent = _AGENT_PENDING_SENTINEL
    event = MessageEvent(text="/stop", source=source, internal=True)
    assert await runner._hm_handle_running_session_message(event, source, key) is None
    assert runner._session_state(key).turn.agent is _AGENT_PENDING_SENTINEL
    assert adapter._pending_messages[key] is event


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["agent", "pending", "adapter_only", "idle"])
async def test_launching_turn_activity_survives_adapter_replacement(monkeypatch, tmp_path, state):
    runner, source, key, agent = make_runner(monkeypatch, tmp_path)
    adapter = install_adapter(runner)
    monkeypatch.setattr(runner, "_build_process_event_source", lambda watcher: source)
    if state in {"agent", "pending"}:
        runner._session_state(key).turn.agent = agent if state == "agent" else _AGENT_PENDING_SENTINEL
    elif state == "adapter_only":
        adapter._active_sessions[key] = asyncio.Event()
    assert await runner._launching_turn_active("discord", {"session_key": key}) is (state != "idle")
    assert await runner._launching_turn_active("discord", {"session_key": key + ":other"}) is False
    assert await runner._launching_turn_active("discord", {}) is False
