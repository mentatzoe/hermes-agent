"""Human follow-ups precede internal notices (aleph-vault#246)."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from gateway.turn_context import TurnContext


class Adapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        pass

    async def get_chat_info(self, chat_id):
        return {}


@pytest.fixture
def queue():
    runner = object.__new__(GatewayRunner)
    runner._draining = False
    adapter = Adapter(PlatformConfig(enabled=True), Platform.DISCORD)
    source = SessionSource(
        platform=Platform.DISCORD, chat_id="home", user_id="human", chat_type="dm"
    )
    key = build_session_key(source)
    runner._delivery_adapter_for = lambda source: adapter
    return runner, adapter, source, key


def event(source, text, *, internal=False, photo=False):
    return MessageEvent(
        text=text, source=source, message_id=text, internal=internal,
        message_type=MessageType.PHOTO if photo else MessageType.TEXT,
        media_urls=["/scratch/photo.png"] if photo else [],
        media_types=["image/png"] if photo else [],
        channel_prompt="keep the channel", metadata={"sentinel": text},
    )


async def drain(queue):
    runner, adapter, source, key = queue
    drained = []
    for _ in range(12):
        pending, text = await runner._run_agent_drain_pending(
            {"final_response": "done"}, adapter, source, key
        )
        if pending is None:
            assert text is None
            break
        drained.append(pending)
    assert runner._queue_depth(key, adapter=adapter) == 0
    return drained


@pytest.mark.asyncio
async def test_photo_overtakes_notices_without_merging_or_losing_attribution(queue):
    runner, adapter, source, key = queue
    n1, n2, h1, n3, h2 = (
        event(source, "notice 1", internal=True),
        event(source, "notice 2", internal=True),
        event(source, "photo", photo=True),
        event(source, "notice 3", internal=True),
        event(source, "human 2"),
    )
    h2.message_type = MessageType.DOCUMENT  # Independent follow-up, not a photo-burst caption.
    for pending in (n1, n2, h1, n3, h2):
        runner._queue_or_replace_pending_event(key, pending)
    assert await drain(queue) == [h1, h2, n1, n2, n3]
    assert h1.media_urls == ["/scratch/photo.png"]
    assert h1.source is source and not h1.internal
    assert h1.channel_prompt == "keep the channel"
    assert h1.metadata == {"sentinel": "photo"}
    assert all(pending._gateway_accepted for pending in (n1, n2, h1, n3, h2))


@pytest.mark.asyncio
async def test_promotion_prioritizes_humans_already_in_legacy_overflow(queue):
    runner, adapter, source, key = queue
    n1, n2, h1, h2 = (
        event(source, "notice 1", internal=True),
        event(source, "notice 2", internal=True),
        event(source, "human 1", photo=True), event(source, "human 2"),
    )
    adapter._pending_messages[key] = n1
    runner._session_state(key).conversation.queued_events.extend([n2, h1, h2])
    assert await drain(queue) == [h1, h2, n1, n2]


@pytest.mark.asyncio
@pytest.mark.parametrize("internal", [False, True])
async def test_depth_cap_restores_selected_event_before_its_own_class(queue, internal):
    runner, adapter, source, key = queue
    first = event(source, "first", internal=internal, photo=not internal)
    second = event(source, "second", internal=internal)
    notice = event(source, "other notice", internal=True)
    for pending in (first, second, notice):
        runner._enqueue_fifo(key, pending, adapter)
    selected, text = await runner._run_agent_drain_pending(
        {"final_response": "done"}, adapter, source, key
    )
    assert selected is first
    ctx = TurnContext(
        source=source, session_key=key, session_id="session", history=[],
        _interrupt_depth=runner._MAX_INTERRUPT_DEPTH,
    )
    runner._run_agent = AsyncMock()
    await runner._run_agent_queued_followup(
        ctx, adapter, text, selected, "done", {"final_response": "done"}, None
    )
    runner._run_agent.assert_not_awaited()
    assert await drain(queue) == [first, second, notice]


@pytest.mark.asyncio
async def test_human_arriving_after_depth_cap_still_precedes_deferred_notice(queue):
    runner, adapter, source, key = queue
    n1, n2 = [event(source, f"notice {i}", internal=True) for i in (1, 2)]
    for pending in (n1, n2):
        runner._enqueue_fifo(key, pending, adapter)
    selected, text = await runner._run_agent_drain_pending(
        {"final_response": "done"}, adapter, source, key
    )
    ctx = TurnContext(
        source=source, session_key=key, session_id="session", history=[],
        _interrupt_depth=runner._MAX_INTERRUPT_DEPTH,
    )
    await runner._run_agent_queued_followup(
        ctx, adapter, text, selected, "done", {"final_response": "done"}, None
    )
    human = event(source, "photo", photo=True)
    runner._queue_or_replace_pending_event(key, human)
    assert await drain(queue) == [human, n1, n2]


@pytest.mark.asyncio
@pytest.mark.parametrize("incoming_internal", [False, True])
async def test_idle_rescue_orders_incoming_event_with_orphans(queue, incoming_internal):
    runner, adapter, source, key = queue
    n1 = event(source, "notice 1", internal=True)
    n2 = event(source, "notice 2", internal=True)
    h1 = event(source, "older human", photo=True)
    incoming = event(source, "incoming", internal=incoming_internal)
    runner._session_state(key).conversation.queued_events.extend([n1, n2, h1])
    rescued, rescued_source, internal = runner._hm_rescue_orphaned_fifo(
        incoming, source, incoming_internal, key
    )
    assert rescued is h1 and rescued_source is source and not internal
    expected = [n1, n2, incoming] if incoming_internal else [incoming, n1, n2]
    assert await drain(queue) == expected


@pytest.mark.asyncio
async def test_idle_human_does_not_wait_for_only_orphaned_notices(queue):
    runner, adapter, source, key = queue
    notices = [event(source, f"notice {i}", internal=True) for i in (1, 2)]
    runner._session_state(key).conversation.queued_events.extend(notices)
    incoming = event(source, "photo", photo=True)
    rescued, _, internal = runner._hm_rescue_orphaned_fifo(incoming, source, False, key)
    assert rescued is incoming and not internal
    assert await drain(queue) == notices
