"""Live-call transcripts retain voice/speaker identity while steering an active turn."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms.event import MessageType
from tests.gateway.test_busy_session_ack import _make_adapter, _make_event, _make_runner
from gateway.session import build_session_key


@pytest.mark.asyncio
@pytest.mark.parametrize("accepts", [True, False])
async def test_live_call_transcript_steers_or_queues_without_relabeling(monkeypatch, accepts):
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    runner, _ = _make_runner()
    runner._busy_input_mode = "steer"
    runner._enrich_message_with_transcription = AsyncMock()
    event = _make_event(text="Please focus on the failing test", platform_val="discord")
    event.message_type = MessageType.VOICE
    event.source.user_name = "Speaker"
    key = build_session_key(event.source)
    adapter = _make_adapter("discord")
    runner.adapters[event.source.platform] = adapter
    agent = MagicMock()
    agent.steer.return_value = accepts
    runner._running_agents[key] = agent

    await runner._handle_active_session_busy_message(event, key)

    agent.steer.assert_called_once()
    injected = agent.steer.call_args.args[0]
    assert event.text in injected
    assert "user1" in injected
    assert event.message_type is MessageType.VOICE
    assert event.source.user_name == "Speaker"
    assert not event.media_urls
    agent.interrupt.assert_not_called()
    runner._enrich_message_with_transcription.assert_not_called()
    assert (adapter._pending_messages.get(key) is event) is (not accepts)
