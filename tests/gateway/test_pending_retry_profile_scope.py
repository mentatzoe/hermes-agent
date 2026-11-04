"""Retry hydration must leave the loop and shared queue available (fork #61 / PR69).

Only external secret I/O and the downstream message handler are stand-ins. Scope
loading, runner selection/eviction and adapter ownership/cancellation remain real.
"""

import asyncio
import threading
import time
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import pytest_asyncio

from agent.secret_scope import get_secret
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.session_identity import RoutingIdentity, replace_source
from hermes_constants import get_hermes_home


class Adapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, **kwargs):
        pass

    async def get_chat_info(self, chat_id):
        return {}


class HydrationGate:
    """One bounded external-source wait, with loop-side arrival/exit signals."""

    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.loop_thread = threading.get_ident()
        self.entered, self.finished = asyncio.Event(), asyncio.Event()
        self.release = threading.Event()
        self.samples = []
        self.released = None

    def __call__(self, home):
        first = not self.samples
        self.samples.append((Path(get_hermes_home()), threading.get_ident()))
        if first:
            self.loop.call_soon_threadsafe(self.entered.set)
            self.released = self.release.wait(3)
            self.loop.call_soon_threadsafe(self.finished.set)
        return {}

    async def wait(self):
        await asyncio.wait_for(self.entered.wait(), 15)
        assert self.samples[0][1] != self.loop_thread, "hydration ran on the event loop"
        assert self.released is None, "loop did not resume until external I/O timed out"


@pytest_asyncio.fixture
async def retry(tmp_path, monkeypatch):
    import gateway.run as run
    import hermes_cli.env_loader as env_loader
    import hermes_cli.profiles as profiles

    # Real profile resolution and per-profile databases, rooted in this fixture.
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: tmp_path)
    launch, owner = tmp_path / "launch", tmp_path / "owner"
    launch.mkdir()
    owner.mkdir()
    (owner / ".env").write_text("RETRY_TEST_PROFILE=owner\n")
    (launch / ".env").write_text("RETRY_TEST_PROFILE=launch\n")
    monkeypatch.setattr(run, "_hermes_home", launch)
    runner = GatewayRunner(GatewayConfig())
    runner.config.multiplex_profiles = True
    adapter = Adapter(PlatformConfig(enabled=True), Platform.DISCORD)
    adapter.gateway_runner = runner
    source = SessionSource(platform=Platform.DISCORD, chat_id="room", chat_type="dm",
                           user_id="human", profile="owner")
    source._identity = RoutingIdentity(
        transport_profile="default", runtime_profile="owner",
        authorization_home=launch, runtime_home=owner, transport=weakref.ref(adapter),
    )
    key = runner._session_key_for_source(source)
    entry = runner.session_store.get_or_create_session(source)
    db = runner.session_store._db_for_key(key)
    db.create_session(entry.session_id, source="discord")
    activity = {"seconds_since_activity": 0}
    evictions, handled = [], []
    agent = SimpleNamespace(
        get_activity_summary=lambda: activity, _active_children=[],
        interrupt=Mock(side_effect=lambda *a, **kw: evictions.append(
            (Path(get_hermes_home()), get_secret("RETRY_TEST_PROFILE")))),
    )
    state = runner._session_state(key)
    state.turn.agent, state.turn.started_ts = agent, time.time()
    gate = HydrationGate()
    monkeypatch.setattr(env_loader, "hydrate_profile_secret_sources", gate)

    async def handler(event):
        assert Path(get_hermes_home()) == owner
        handled.append((event.message_id, event.source.user_id, event.internal))
        return None

    monkeypatch.setattr(runner, "_handle_message", handler)
    adapter.set_message_handler(runner._make_default_profile_message_handler())
    h = SimpleNamespace(runner=runner, adapter=adapter, source=source, key=key,
                        owner=owner, state=state, gate=gate, activity=activity,
                        agent=agent, evictions=evictions, handled=handled, entry=entry, db=db)
    try:
        yield h
    finally:
        gate.release.set()
        await adapter.cancel_background_tasks()
        if gate.samples:
            await asyncio.wait_for(gate.finished.wait(), 15)


def enqueue(h, name, internal=False):
    event = MessageEvent(text=name, message_id=name, internal=internal,
                         source=replace_source(h.source, user_id=name))
    h.runner._enqueue_fifo(h.key, event, h.adapter)
    return event


def start(h):
    guard = asyncio.Event()
    h.adapter._active_sessions[h.key] = guard
    h.adapter._finish_session_task(h.key, guard)
    return h.adapter._session_tasks[h.key], guard


async def settle(h):
    while (task := h.adapter._session_tasks.get(h.key)) is not None:
        await asyncio.wait_for(asyncio.shield(task), 15)
    assert not h.adapter._active_sessions
    assert h.runner._queue_depth(h.key, adapter=h.adapter) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("consume_all", [False, True], ids=["promoted-slot", "empty-slot"])
async def test_original_turn_can_drain_during_retry_hydration(retry, consume_all):
    h = retry
    events = [enqueue(h, name, internal) for name, internal in
              [("n1", True), ("n2", True), ("h1", False), ("h2", False)]]
    keeper, _ = start(h)
    await h.gate.wait()
    assert h.adapter._pending_messages[h.key] is events[2]
    consumed = []

    async def original_turn():
        for _ in range(4 if consume_all else 1):
            event, text = await h.runner._run_agent_drain_pending(
                {"final_response": "done"}, h.adapter, h.source, h.key)
            consumed.append((event.message_id, event.source.user_id, event.internal))
            assert text == event.text
        h.state.turn.agent = None

    await asyncio.create_task(original_turn())
    h.gate.release.set()
    await asyncio.wait_for(asyncio.shield(keeper), 15)
    await settle(h)
    assert consumed + h.handled == [
        ("h1", "h1", False), ("h2", "h2", False), ("n1", "n1", True), ("n2", "n2", True)]
    assert h.gate.released is True
    assert h.gate.samples[0][0] == h.owner
    assert not h.evictions


@pytest.mark.asyncio
async def test_new_human_replaces_notice_selection_during_hydration(retry):
    h = retry
    enqueue(h, "n1", True)
    enqueue(h, "n2", True)
    start(h)
    await h.gate.wait()
    enqueue(h, "h1")
    enqueue(h, "h2")
    h.state.turn.agent = None
    h.gate.release.set()
    await settle(h)
    assert h.handled == [
        ("h1", "h1", False), ("h2", "h2", False), ("n1", "n1", True), ("n2", "n2", True)]


@pytest.mark.asyncio
@pytest.mark.parametrize("discard", [False, True], ids=["keep", "discard"])
async def test_command_cancels_retry_while_external_hydration_is_blocked(retry, discard):
    h = retry
    queued = enqueue(h, "human")
    keeper, _ = start(h)
    await h.gate.wait()
    command_guard = asyncio.Event()
    h.adapter._active_sessions[h.key] = command_guard
    await h.adapter.cancel_session_processing(h.key, release_guard=False, discard_pending=discard)
    assert keeper.cancelled()
    assert not h.gate.release.is_set() and not h.gate.finished.is_set()
    assert h.adapter._active_sessions[h.key] is command_guard
    assert h.adapter._pending_messages.get(h.key) is (None if discard else queued)
    h.state.turn.agent = None
    h.gate.release.set()
    await asyncio.wait_for(h.gate.finished.wait(), 15)
    # Completing the abandoned thread must not dispatch its stale event.
    assert not h.handled
    await h.adapter._drain_pending_after_session_command(h.key, command_guard)
    await settle(h)
    assert h.handled == ([] if discard else [("human", "human", False)])
    assert not h.evictions


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["guard", "task"])
async def test_superseded_retry_cannot_claim_or_evict_after_hydration(retry, replacement):
    h = retry
    queued = enqueue(h, "human")
    keeper, guard = start(h)
    await h.gate.wait()
    # If the old task evicts before checking ownership, this agent is stale.
    h.activity["seconds_since_activity"] = 1e9
    successor_done = asyncio.Event()
    successor = None
    if replacement == "guard":
        guard = asyncio.Event()
        h.adapter._active_sessions[h.key] = guard
    else:
        successor = asyncio.create_task(successor_done.wait())
        h.adapter._track_session_task(h.key, successor)
    try:
        h.gate.release.set()
        await asyncio.wait_for(asyncio.shield(keeper), 15)
        assert h.adapter._pending_messages[h.key] is queued
        assert h.adapter._active_sessions[h.key] is guard
        assert h.adapter._session_tasks[h.key] is (successor or keeper)
        assert not h.handled and not h.evictions
        assert h.state.turn.agent is h.agent
    finally:
        successor_done.set()
        if successor is not None:
            await successor


@pytest.mark.asyncio
async def test_replacement_running_owner_is_rechecked_after_hydration(retry):
    h = retry
    queued = enqueue(h, "human")
    start(h)
    await h.gate.wait()
    replacement = SimpleNamespace(get_activity_summary=lambda: {"seconds_since_activity": 0})
    h.state.turn.agent = replacement
    backed_off = asyncio.Event()
    real_backoff = h.adapter._requeue_backoff_delay

    def observe(*args):
        backed_off.set()
        return real_backoff(*args)

    h.adapter._requeue_backoff_delay = observe
    h.gate.release.set()
    await asyncio.wait_for(backed_off.wait(), 15)
    assert h.adapter._pending_messages[h.key] is queued
    assert h.state.turn.agent is replacement
    assert not h.handled and not h.evictions
    h.state.turn.agent = None
    await settle(h)
    assert h.handled == [("human", "human", False)]


@pytest.mark.asyncio
async def test_retry_reselects_after_adapter_runner_replacement(retry):
    h = retry
    queued = enqueue(h, "human")
    start(h)
    await h.gate.wait()
    h.activity["seconds_since_activity"] = 1e9
    # A new runner owns this adapter now. Its idle state must win over the old
    # runner's stale slot, without interrupting the old runner on its behalf.
    replacement = object.__new__(GatewayRunner)
    h.adapter.gateway_runner = replacement
    h.gate.release.set()
    await settle(h)
    assert h.handled == [(queued.message_id, "human", False)]
    assert not h.evictions
    assert h.state.turn.agent is h.agent


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["stale", "reaped"])
async def test_retry_eviction_runs_in_routed_profile_scope(retry, finish):
    h = retry
    enqueue(h, "human")
    start(h)
    await h.gate.wait()
    if finish == "stale":
        h.activity["seconds_since_activity"] = 1e9
    else:
        h.db.end_session(h.entry.session_id, "test_reap")
    h.gate.release.set()
    await settle(h)
    assert h.evictions == [(h.owner, "owner")]
    h.agent.interrupt.assert_called_once()
    assert h.handled == [("human", "human", False)]
    assert h.gate.samples[0][0] == h.owner
