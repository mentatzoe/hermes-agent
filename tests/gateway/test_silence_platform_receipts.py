"""Delivery evidence must mean visible prose across adapter-specific send semantics."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.run_turn import _unexpected_silence_reply
from gateway.run_turn_runner import TurnRunner
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.wecom.adapter import WeComAdapter
from tests.gateway.test_gateway_silence_tokens import _event, _source
from tests.gateway.test_line_plugin import LineAdapter, State
from tests.gateway.test_silence_after_delivery import ANSWER, KEY, _turn


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["completion", "queued"])
async def test_noneditable_idless_reply_survives_final_silence(monkeypatch, tmp_path, lane):
    runner, ctx, result, adapter = await _turn(monkeypatch, tmp_path, "none")
    adapter.send.side_effect = None
    adapter.send.return_value = SimpleNamespace(success=True, message_id=None)
    consumer = ctx.stream_consumer_holder[0]
    consumer.on_delta(ANSWER)
    consumer.on_delta(None)
    consumer.on_delta("[SILENT]")
    consumer.finish("[SILENT]")
    await consumer.run()
    adapter.send.assert_awaited_once()
    adapter.delete_message.assert_not_awaited()
    if lane == "completion":
        await runner._run_agent_mark_streamed_delivery(result, ctx)
        runner._run_agent = AsyncMock(return_value=result)
        assert await runner._handle_message_with_agent(_event(reply_expected=True), _source(), KEY, 1) == ""
    else:
        runner._deliver_queued_first_response = AsyncMock()
        await runner._run_agent_deliver_first_response(ctx, adapter, result, result, None)
        runner._deliver_queued_first_response.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("ack,answered", [
    ({"errcode": 0}, True),
    ({"errcode": 0, "errmsg": "ack_timeout_assumed_delivered", "ack_pending": True}, False),
    ({"errcode": 6000}, False),
])
async def test_wecom_optimistic_finalization_is_not_a_durable_receipt(monkeypatch, tmp_path, ack, answered):
    runner, ctx, result, _ = await _turn(monkeypatch, tmp_path, "none")
    adapter = WeComAdapter(PlatformConfig(enabled=True))
    chat = _source().chat_id
    adapter._last_chat_req_ids[chat] = "test-request"
    adapter._send_reply_queued = AsyncMock(return_value=ack)
    consumer = GatewayStreamConsumer(adapter, chat, StreamConsumerConfig(cursor=""))
    ctx.stream_consumer_holder[0] = consumer
    try:
        # Drive the real native adapter; only its transport-ACK seam is mocked.
        assert await consumer._send_frame("", finalize=False)
        consumer._native_stream_opened = consumer._use_native_streaming = True
        consumer._append_accumulated(ANSWER)
        future = asyncio.get_running_loop().create_future()
        await consumer._handle_approval_boundary(future)
        assert future.result() is True  # Existing duplicate-suppression semantics stay intact.
        await runner._run_agent_mark_streamed_delivery(result, ctx)
        runner._run_agent = AsyncMock(return_value=result)
        response = await runner._handle_message_with_agent(_event(reply_expected=True), _source(), KEY, 1)
        assert response == ("" if answered else _unexpected_silence_reply())
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["stream", "fallback-interim", "queued"])
async def test_line_pending_button_does_not_cache_a_counted_reply(monkeypatch, tmp_path, delivery):
    runner, ctx, result, _ = await _turn(monkeypatch, tmp_path, "none")
    monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
    adapter = LineAdapter(PlatformConfig(enabled=True, extra={"channel_access_token": "test", "channel_secret": "test"}))
    adapter._client = MagicMock(push=AsyncMock(), reply=AsyncMock())
    chat = _source().chat_id
    request = adapter._cache.register_pending(chat)
    adapter._pending_buttons[chat] = request
    runner._delivery_adapter_for = lambda _source: adapter
    if delivery == "queued":
        ctx.stream_consumer_holder[0] = None
        result["final_response"] = ANSWER
        runner._deliver_media_from_response = AsyncMock()
        await runner._run_agent_deliver_first_response(ctx, adapter, result, result, None)
    elif delivery == "fallback-interim":
        ctx._status_adapter = adapter
        def unavailable(*_a, **_kw):
            raise RuntimeError("consumer unavailable")
        runner._build_stream_consumer_config = unavailable
        ctx.stream_consumer_holder[0] = None
        _, _, interim, _ = TurnRunner(runner, ctx)._setup_stream_consumer("line")
        interim(ANSWER)
        await runner._run_agent_mark_streamed_delivery(result, ctx)
    else:
        consumer = GatewayStreamConsumer(adapter, chat, StreamConsumerConfig(cursor=""))
        ctx.stream_consumer_holder[0] = consumer
        consumer.on_delta(ANSWER)
        consumer.on_delta(None)
        consumer.on_delta("[SILENT]")
        consumer.finish("[SILENT]")
        await consumer.run()
        await runner._run_agent_mark_streamed_delivery(result, ctx)
    adapter._client.push.assert_awaited_once()  # A cache acceptance is not a visible delivery.
    assert adapter._cache.get(request).state is State.PENDING
    assert result.get("visible_text_delivered") is True
