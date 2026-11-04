"""A bare silence marker may end an answered turn, never an undelivered one."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import StreamingConfig
from gateway.run_turn import _unexpected_silence_reply
from gateway.run_turn_runner import TurnRunner
from gateway.stream_consumer import StreamConsumerConfig
from gateway.turn_context import TurnContext
from tests.gateway.test_gateway_silence_tokens import _event, _runner, _source


ANSWER = "I've checked it; the earlier reply is the answer."
KEY = "agent:main:telegram:group:-1001:12345"


async def _turn(monkeypatch, tmp_path, delivery, outcome="sent"):
    runner = _runner(monkeypatch, tmp_path)
    receipt = SimpleNamespace(success=outcome == "sent", message_id="reply-1")
    adapter = SimpleNamespace(
        platform=_source().platform, MAX_MESSAGE_LENGTH=4096,
        SUPPORTS_MESSAGE_EDITING=True, SUPPORTS_NATIVE_STREAMING=False,
        send=AsyncMock(return_value=receipt), edit_message=AsyncMock(return_value=receipt),
        send_draft=AsyncMock(return_value=SimpleNamespace(success=True)),
        delete_message=AsyncMock(return_value=True),
        extract_media=lambda text: ([], text),
    )
    attempted = asyncio.Event()

    async def send(*_args, **_kwargs):
        attempted.set()
        if outcome == "exception":
            raise RuntimeError("transport unavailable")
        return receipt

    adapter.send.side_effect = send
    runner._hmwa_first_contact_notes = AsyncMock()
    runner._delivery_adapter_for = lambda _source: adapter
    runner._build_stream_consumer_config = lambda *_a, **_kw: (StreamConsumerConfig(cursor=""), None)
    runner.config.streaming = StreamingConfig(enabled=delivery == "stream")
    ctx = TurnContext(
        source=_source(), session_key=KEY, session_id="sess-silent", run_generation=1,
        reply_expected=True, inbound_message_id="msg-42", history=[],
        interim_assistant_messages_enabled=True, user_config={},
        resolve_display_setting=lambda *_a: None, _run_still_current=lambda: True,
        _status_adapter=adapter, _status_chat_id=_source().chat_id,
        _loop_for_step=asyncio.get_running_loop(),
    )
    if delivery == "fallback":
        def unavailable(*_a, **_kw):
            raise RuntimeError("stream consumer setup unavailable")
        runner._build_stream_consumer_config = unavailable
    consumer, delta, interim, enabled = TurnRunner(runner, ctx)._setup_stream_consumer("telegram")
    assert enabled
    result = {"final_response": "[SILENT]", "failed": False, "messages": [], "tools": []}
    if delivery in {"interim", "pending", "fallback"}:
        interim(ANSWER)
    elif delivery == "stream":
        delta(ANSWER)
        delta(None)
        delta("[SILENT]")
    elif delivery in {"draft", "draft-segment"}:
        consumer._use_draft_streaming, consumer._draft_id = True, "draft-1"
        assert await consumer._send_or_edit(ANSWER, finalize=False)
        assert not await consumer._send_or_edit(ANSWER, finalize=True)
        if delivery == "draft-segment":
            consumer._reset_segment_state()
    if consumer is not None and delivery not in {"pending", "draft", "draft-segment"}:
        consumer.finish("[SILENT]")
        await consumer.run()
    if delivery == "fallback":
        await asyncio.wait_for(attempted.wait(), timeout=2)
    return runner, ctx, result, adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["completion", "queued"])
@pytest.mark.parametrize("delivery,outcome,answered", [
    ("interim", "sent", True),
    ("interim", "refused", False),
    ("interim", "exception", False),
    ("stream", "sent", True),
    ("fallback", "sent", True),
    ("fallback", "refused", False),
    ("fallback", "exception", False),
    ("none", "sent", False),
    ("pending", "sent", False),
    ("draft", "refused", False),
    ("draft-segment", "refused", False),
])
async def test_silence_requires_successful_visible_delivery(
    monkeypatch, tmp_path, caplog, lane, delivery, outcome, answered,
):
    runner, ctx, result, adapter = await _turn(monkeypatch, tmp_path, delivery, outcome)
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
    if answered:
        assert not any("silence marker" in r.message and r.levelname == "WARNING" for r in caplog.records)
        assert any(ANSWER in str(call) for call in adapter.send.await_args_list)
        adapter.delete_message.assert_not_awaited()  # Final silence must preserve the earlier answer.
    if delivery in {"interim", "fallback"} and outcome != "sent":
        adapter.send.assert_awaited_once()  # Reached the transport failure, not an earlier setup error.


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["completion", "queued"])
@pytest.mark.parametrize("delivery,outcome", [("none", "sent"), ("interim", "sent"), ("interim", "refused")])
async def test_notice_turns_still_allow_silence(monkeypatch, tmp_path, lane, delivery, outcome):
    runner, ctx, result, adapter = await _turn(monkeypatch, tmp_path, delivery, outcome)
    ctx.persist_user_display_kind = "internal_notification"
    if lane == "completion":
        await runner._run_agent_mark_streamed_delivery(result, ctx)
        runner._run_agent = AsyncMock(return_value=result)
        assert await runner._handle_message_with_agent(_event(internal=True), _source(), KEY, 1) == ""
    else:
        runner._deliver_queued_first_response = AsyncMock()
        await runner._run_agent_deliver_first_response(ctx, adapter, result, result, None)
        runner._deliver_queued_first_response.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("steered", [False, True], ids=["queued", "leftover-steer"])
@pytest.mark.parametrize("terminal_answered", [False, True])
async def test_followup_uses_its_own_delivery_not_the_openers(
    monkeypatch, tmp_path, steered, terminal_answered,
):
    runner, ctx, opener, adapter = await _turn(monkeypatch, tmp_path, "interim")
    terminal_runner, terminal_ctx, terminal, _ = await _turn(
        monkeypatch, tmp_path, "interim" if terminal_answered else "none",
    )
    await terminal_runner._run_agent_mark_streamed_delivery(terminal, terminal_ctx)
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._run_agent = AsyncMock(return_value=terminal)
    runner._deliver_queued_first_response = AsyncMock(return_value=True)
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._session_key_for_source = MagicMock(return_value=KEY)
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="follow-up")
    runner._refresh_agent_cache_message_count = AsyncMock()
    pending_event = None if steered else _event(reply_expected=True)
    if pending_event:
        pending_event.message_id = "msg-43"
    merged = await runner._run_agent_queued_followup(
        ctx, None, "follow-up", pending_event, opener, opener, None,
    )
    runner._deliver_queued_first_response.assert_not_awaited()  # Opener already answered.
    runner._run_agent = AsyncMock(return_value=merged)
    response = await runner._handle_message_with_agent(_event(reply_expected=True), _source(), KEY, 1)
    assert response == ("" if terminal_answered else _unexpected_silence_reply())


@pytest.mark.asyncio
async def test_delivered_queued_reply_counts_if_the_same_turn_finishes_silently(monkeypatch, tmp_path, caplog):
    runner, ctx, result, adapter = await _turn(monkeypatch, tmp_path, "none")
    result["final_response"] = ANSWER
    runner._deliver_media_from_response = AsyncMock()
    await runner._run_agent_deliver_first_response(ctx, adapter, result, result, None)
    adapter.send.assert_awaited_once()
    result["final_response"] = "[SILENT]"
    runner._run_agent = AsyncMock(return_value=result)
    assert await runner._handle_message_with_agent(_event(reply_expected=True), _source(), KEY, 1) is None
    assert not any("silence marker" in r.message and r.levelname == "WARNING" for r in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["refused", "media-only", "declined-edit"])
async def test_queued_suppression_is_not_a_text_receipt(monkeypatch, tmp_path, case):
    runner, ctx, result, adapter = await _turn(monkeypatch, tmp_path, "none", "refused")
    runner._deliver_media_from_response = AsyncMock()
    result["final_response"] = ANSWER
    if case == "media-only":
        adapter.extract_media = lambda _text: ([], "")
    elif case == "declined-edit":
        ctx.stream_consumer_holder[0]._message_id = "old-preview"
        adapter.edit_message.return_value = SimpleNamespace(
            success=False, error="telegram egress declined: target is not an approved destination for this connection",
        )
    await runner._run_agent_deliver_first_response(ctx, adapter, result, result, None)
    assert not result.get("visible_text_delivered")
    if case == "refused":
        adapter.send.assert_awaited_once()
    else:
        adapter.send.assert_not_awaited()
    if case == "declined-edit":
        adapter.edit_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_transcript_text_without_a_delivery_receipt_does_not_allow_silence(monkeypatch, tmp_path):
    runner, ctx, result, _ = await _turn(monkeypatch, tmp_path, "none")
    result["messages"] = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": ANSWER},
        {"role": "assistant", "content": "[SILENT]"},
    ]
    runner._run_agent = AsyncMock(return_value=result)
    assert await runner._handle_message_with_agent(_event(reply_expected=True), _source(), KEY, 1) == _unexpected_silence_reply()
    history = [{**row, "timestamp": 100.0} for row in result["messages"]]
    assert runner._crash_left_reply(history, 50.0, _source()) == _unexpected_silence_reply()
