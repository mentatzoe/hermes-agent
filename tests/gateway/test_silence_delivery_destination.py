"""An answer delivered outside the originating topic must not permit silence."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.run_turn import _unexpected_silence_reply
from gateway.run_turn_runner import TurnRunner
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.gateway.test_gateway_silence_tokens import _event
from tests.gateway.test_silence_after_delivery import ANSWER, KEY, _turn
from tests.gateway.test_telegram_thread_fallback import (
    FakeBadRequest, _inject_fake_telegram, _make_adapter,
)


async def _telegram_turn(monkeypatch, tmp_path, topic_exists):
    runner, ctx, result, _ = await _turn(monkeypatch, tmp_path, "none")
    ctx.source.thread_id = "777"
    ctx._status_thread_metadata = {"thread_id": "777"}
    adapter = _make_adapter()
    adapter._retrigger_typing = AsyncMock()
    adapter._post_delivery_callbacks = {}
    adapter._last_overflow_preview = {}
    adapter._rich_messages_enabled = False
    attempts, delivered, edits = [], [], []
    delivered_event = asyncio.Event()

    async def send_message(**kwargs):
        attempts.append(kwargs)
        if not topic_exists and kwargs.get("message_thread_id") is not None:
            raise FakeBadRequest("Message thread not found")
        delivered.append(kwargs)
        delivered_event.set()
        return SimpleNamespace(message_id=42)

    async def edit_message_text(**kwargs):
        edits.append(kwargs)

    adapter._bot = SimpleNamespace(
        send_message=send_message, edit_message_text=edit_message_text,
    )
    runner._delivery_adapter_for = lambda source: adapter
    ctx._status_adapter = adapter
    ctx.stream_consumer_holder[0] = None
    return runner, ctx, result, adapter, attempts, delivered, edits, delivered_event


def _consumer(ctx, adapter):
    consumer = GatewayStreamConsumer(
        adapter, ctx.source.chat_id, StreamConsumerConfig(cursor=""),
        metadata=ctx._status_thread_metadata,
    )
    ctx.stream_consumer_holder[0] = consumer
    return consumer


async def _silence_response(runner, ctx, result, adapter, lane):
    if lane == "completion":
        await runner._run_agent_mark_streamed_delivery(result, ctx)
        runner._run_agent = AsyncMock(return_value=result)
        event = _event(reply_expected=True)
        event.source = ctx.source
        return await runner._handle_message_with_agent(event, ctx.source, KEY, 1)
    runner._deliver_queued_first_response = AsyncMock(return_value=True)
    await runner._run_agent_deliver_first_response(ctx, adapter, result, result, None)
    calls = runner._deliver_queued_first_response.await_args_list
    return calls[0].args[0] if calls else ""


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["completion", "queued"])
@pytest.mark.parametrize("delivery", ["interim", "stream", "fallback-interim"])
@pytest.mark.parametrize("topic_exists", [True, False], ids=["same-topic", "root-fallback"])
async def test_only_delivery_to_the_originating_topic_allows_silence(
    monkeypatch, tmp_path, lane, delivery, topic_exists,
):
    runner, ctx, result, adapter, attempts, delivered, _, delivered_event = (
        await _telegram_turn(monkeypatch, tmp_path, topic_exists)
    )
    if delivery == "fallback-interim":
        def unavailable(*_args, **_kwargs):
            raise RuntimeError("No consumer available")

        runner._build_stream_consumer_config = unavailable
        _, _, interim, _ = TurnRunner(runner, ctx)._setup_stream_consumer("telegram")
        interim(ANSWER)
        await asyncio.wait_for(delivered_event.wait(), timeout=2)
    else:
        consumer = _consumer(ctx, adapter)
        if delivery == "interim":
            consumer.on_commentary(ANSWER)
        else:
            consumer.on_delta(ANSWER)
            consumer.on_delta(None)
        consumer.on_delta("[SILENT]")
        consumer.finish("[SILENT]")
        await consumer.run()
        # Wrong-topic delivery remains visible to ordinary duplicate suppression.
        assert consumer.has_durably_delivered_text(ANSWER) is True

    response = await _silence_response(runner, ctx, result, adapter, lane)

    assert len(delivered) == 1, (attempts, delivered)
    assert delivered[0].get("message_thread_id") == (777 if topic_exists else None)
    assert response == ("" if topic_exists else _unexpected_silence_reply())


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["completion", "queued"])
@pytest.mark.parametrize("topic_exists", [True, False], ids=["same-topic", "root-fallback"])
async def test_edits_and_segment_resets_preserve_delivery_destination(
    monkeypatch, tmp_path, lane, topic_exists,
):
    runner, ctx, result, adapter, _, delivered, edits, _ = await _telegram_turn(
        monkeypatch, tmp_path, topic_exists,
    )
    consumer = _consumer(ctx, adapter)
    consumer._append_accumulated(ANSWER)
    assert await consumer._send_or_edit(ANSWER, finalize=False)
    consumer._append_accumulated(" More detail.")
    assert await consumer._send_or_edit(
        ANSWER + " More detail.", finalize=True, is_turn_final=False,
    )
    consumer._reset_segment_state()
    consumer.on_delta("[SILENT]")
    consumer.finish("[SILENT]")
    await consumer.run()

    response = await _silence_response(runner, ctx, result, adapter, lane)

    assert len(delivered) == 1
    assert delivered[0].get("message_thread_id") == (777 if topic_exists else None)
    assert len(edits) == 1  # Telegram's edit receipt has no fresh routing provenance.
    assert response == ("" if topic_exists else _unexpected_silence_reply())


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["completion", "queued"])
@pytest.mark.parametrize("topic_exists", [True, False], ids=["same-topic", "root-fallback"])
async def test_queued_reply_destination_controls_later_silence(
    monkeypatch, tmp_path, lane, topic_exists,
):
    runner, ctx, result, adapter, _, delivered, _, _ = await _telegram_turn(
        monkeypatch, tmp_path, topic_exists,
    )
    runner._deliver_media_from_response = AsyncMock()
    result["final_response"] = ANSWER
    await runner._run_agent_deliver_first_response(ctx, adapter, result, result, None)

    assert len(delivered) == 1
    assert delivered[0].get("message_thread_id") == (777 if topic_exists else None)
    assert result.get("visible_text_delivered", False) is topic_exists

    result["final_response"] = "[SILENT]"
    response = await _silence_response(runner, ctx, result, adapter, lane)
    if topic_exists:
        assert response in ("", None)
    else:
        assert response == _unexpected_silence_reply()


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["completion", "queued"])
@pytest.mark.parametrize("delivery", ["commentary", "stream-segment"])
@pytest.mark.parametrize("topic_exists", [True, False], ids=["same-topic", "root-fallback"])
async def test_queued_dedupe_preserves_original_destination_after_reset(
    monkeypatch, tmp_path, lane, delivery, topic_exists,
):
    runner, ctx, result, adapter, attempts, delivered, edits, _ = await _telegram_turn(
        monkeypatch, tmp_path, topic_exists,
    )
    runner._deliver_media_from_response = AsyncMock()
    consumer = _consumer(ctx, adapter)
    if delivery == "commentary":
        consumer.on_commentary(ANSWER)
    else:
        consumer.on_delta(ANSWER)
        consumer.on_delta(None)
    consumer.finish("[SILENT]")
    await consumer.run()
    assert consumer.has_durably_delivered_text(ANSWER) is True
    assert consumer.has_durably_delivered_text() is topic_exists
    assert len(delivered) == 1
    assert delivered[0].get("message_thread_id") == (777 if topic_exists else None)

    result["final_response"] = ANSWER
    await runner._run_agent_deliver_first_response(ctx, adapter, result, result, None)
    assert len(delivered) == 1, (attempts, delivered)
    assert not edits
    receipt_after_dedupe = result.get("visible_text_delivered", False)

    result["final_response"] = "[SILENT]"
    response = await _silence_response(runner, ctx, result, adapter, lane)
    if topic_exists:
        assert receipt_after_dedupe is True
        assert response in ("", None)
    else:
        assert receipt_after_dedupe is False
        assert response == _unexpected_silence_reply()
