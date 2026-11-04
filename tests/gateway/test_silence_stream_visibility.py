"""Silence preserves delivered prose, but native tool/status frames are not replies."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.run_turn import _unexpected_silence_reply
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.gateway.test_gateway_silence_tokens import _event, _source
from tests.gateway.test_silence_after_delivery import ANSWER, KEY, _turn
from tests.gateway.test_stream_consumer_wecom_native import _make_native_streaming_adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("assistant_text", [False, True], ids=["progress-only", "assistant-prose"])
@pytest.mark.parametrize("lane", ["completion", "queued"])
async def test_native_boundary_only_counts_assistant_prose(monkeypatch, tmp_path, assistant_text, lane):
    runner, ctx, result, _ = await _turn(monkeypatch, tmp_path, "none")
    adapter = _make_native_streaming_adapter()
    adapter._post_delivery_callbacks = {}
    consumer = GatewayStreamConsumer(adapter, _source().chat_id, StreamConsumerConfig(cursor=""))
    ctx.stream_consumer_holder[0] = consumer
    sent = asyncio.Event()
    send_frame = adapter.send_stream_frame
    text = ANSWER if assistant_text else "Searching the web"

    async def frame(content, **kwargs):
        receipt = await send_frame(content, **kwargs)
        if text in content:
            sent.set()
        return receipt

    adapter.send_stream_frame = frame
    task = asyncio.create_task(consumer.run())
    if assistant_text:
        consumer.on_delta(text)
    else:
        consumer.on_tool_progress(text)
    try:
        await asyncio.wait_for(sent.wait(), timeout=3)
        future, _ = consumer.close_for_approval_prompt(reopen=True)
        assert await asyncio.wait_for(future, timeout=3)
        consumer.finish("[SILENT]")
        await asyncio.wait_for(task, timeout=3)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    if lane == "completion":
        await runner._run_agent_mark_streamed_delivery(result, ctx)
        runner._run_agent = AsyncMock(return_value=result)
        response = await runner._handle_message_with_agent(_event(reply_expected=True), _source(), KEY, 1)
    else:
        runner._deliver_queued_first_response = AsyncMock(return_value=True)
        await runner._run_agent_deliver_first_response(ctx, adapter, result, result, None)
        calls = runner._deliver_queued_first_response.await_args_list
        response = calls[0].args[0] if calls else ""
    assert response == ("" if assistant_text else _unexpected_silence_reply())


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["completion", "queued"])
@pytest.mark.parametrize("preface,finalize_ok,fallback_ok,answered", [
    ("progress", False, False, False),
    ("dropped-prose", False, False, False),
    ("seed", True, False, True),
    ("seed", False, True, True),
])
async def test_native_receipt_requires_acknowledged_finalization(
    monkeypatch, tmp_path, lane, preface, finalize_ok, fallback_ok, answered,
):
    runner, ctx, result, _ = await _turn(monkeypatch, tmp_path, "none")
    adapter = _make_native_streaming_adapter()
    adapter._post_delivery_callbacks = {}
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=fallback_ok))
    consumer = GatewayStreamConsumer(adapter, _source().chat_id, StreamConsumerConfig(cursor=""))
    ctx.stream_consumer_holder[0] = consumer
    ready = asyncio.Event()

    async def frame(content, *, finalize=False, **kwargs):
        if (preface == "seed" and not content) or (preface != "seed" and content):
            ready.set()
        # Native APIs may accept/drop intermediate frames without delivery (WeCom).
        return finalize_ok if finalize else True

    adapter.send_stream_frame = frame
    task = asyncio.create_task(consumer.run())
    if preface == "progress":
        consumer.on_tool_progress("Searching")
    elif preface == "dropped-prose":
        consumer.on_delta(ANSWER)
    try:
        await asyncio.wait_for(ready.wait(), timeout=3)
        if preface != "dropped-prose":
            consumer.on_delta(ANSWER)
        # No yield: the boundary is consumed before the next prose frame can be pushed.
        future, _ = consumer.close_for_approval_prompt(reopen=True)
        assert await asyncio.wait_for(future, timeout=3) == (finalize_ok or fallback_ok)
        consumer.finish("[SILENT]")
        await asyncio.wait_for(task, timeout=3)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    if lane == "completion":
        await runner._run_agent_mark_streamed_delivery(result, ctx)
        runner._run_agent = AsyncMock(return_value=result)
        response = await runner._handle_message_with_agent(_event(reply_expected=True), _source(), KEY, 1)
    else:
        runner._deliver_queued_first_response = AsyncMock(return_value=True)
        await runner._run_agent_deliver_first_response(ctx, adapter, result, result, None)
        calls = runner._deliver_queued_first_response.await_args_list
        response = calls[0].args[0] if calls else ""
    assert response == ("" if answered else _unexpected_silence_reply())


@pytest.mark.asyncio
async def test_successful_segment_tail_recovery_is_a_receipt(monkeypatch, tmp_path):
    runner, ctx, result, adapter = await _turn(monkeypatch, tmp_path, "none")
    adapter.send.side_effect = [
        SimpleNamespace(success=False, message_id=None),
        SimpleNamespace(success=True, message_id="tail-1"),
    ]
    consumer = ctx.stream_consumer_holder[0]
    consumer.on_delta(ANSWER)
    consumer.on_delta(None)
    consumer.on_delta("[SILENT]")
    consumer.finish("[SILENT]")
    await consumer.run()
    assert adapter.send.await_count == 2
    await runner._run_agent_mark_streamed_delivery(result, ctx)
    runner._run_agent = AsyncMock(return_value=result)
    assert await runner._handle_message_with_agent(_event(reply_expected=True), _source(), KEY, 1) == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["sent", "refused"])
async def test_inline_steer_keeps_the_current_turn_receipt(monkeypatch, tmp_path, outcome):
    runner, ctx, result, _ = await _turn(monkeypatch, tmp_path, "interim", outcome)
    result["messages"] = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": ANSWER},
        {"role": "user", "content": "An inline steering message"},
        {"role": "assistant", "content": "[SILENT]"},
    ]
    await runner._run_agent_mark_streamed_delivery(result, ctx)
    runner._run_agent = AsyncMock(return_value=result)
    response = await runner._handle_message_with_agent(_event(reply_expected=True), _source(), KEY, 1)
    assert response == ("" if outcome == "sent" else _unexpected_silence_reply())
