"""Explicit command facts and media must survive admission and FIFO fallback."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from gateway.session import SessionSource
from gateway.typed_text import typed_text_for_event
from tests.agent.test_steer import _bare_agent
from tests.gateway.test_pending_steer_drain import Adapter

WORDS = 'Use the north entrance.'
PATH = '/public-fixture/blair-recording.ogg'


def setup(monkeypatch, receiver):
    monkeypatch.setattr('gateway.run._load_gateway_config', lambda: {})
    runner = GatewayRunner(config=GatewayConfig(group_sessions_per_user=False))
    source = SessionSource(platform=Platform.DISCORD, chat_id='public-room', chat_type='group',
                           user_id='blair', user_name='Blair')
    event = MessageEvent(text='/steer ' + WORDS, source=source, message_id='command',
                         message_type=MessageType.VOICE, media_urls=[PATH], media_types=['audio/ogg'])
    adapter = Adapter(PlatformConfig(enabled=True), Platform.DISCORD)
    runner._delivery_adapter_for = lambda src: adapter
    runner._session_state('key').turn.agent = receiver
    runner._should_echo_stt_transcripts = lambda: False
    runner._transcribe_one_clip = AsyncMock(return_value=(None, '[voice transcription unavailable]'))
    return runner, event, adapter


@pytest.mark.asyncio
async def test_explicit_steer_captures_its_owned_recording_before_labels(monkeypatch):
    agent = _bare_agent()
    runner, event, _ = setup(monkeypatch, agent)
    assert WORDS in await runner._busy_steer_command(event, 'key', event.source)
    entries = agent._drain_pending_steer(with_entries=True)
    assert len(entries) == 1 and entries[0]['facts'] == {
        'sender_id': 'blair', 'typed_user_content': WORDS, 'recoverable_audio_paths': [PATH]}
    assert '[Blair]' in entries[0]['text']
    assert event.text == '/steer ' + WORDS


@pytest.mark.asyncio
@pytest.mark.parametrize('receiver', [_AGENT_PENDING_SENTINEL, SimpleNamespace()])
async def test_command_fallback_keeps_original_media_source_and_fifo(monkeypatch, receiver):
    runner, event, adapter = setup(monkeypatch, receiver)
    event._gateway_recoverable_audio_paths = [PATH]
    event.reply_to_message_id = 'earlier-public-message'
    event.channel_context = 'public channel context'
    runner._enqueue_fifo('key', MessageEvent(text='first queued', source=event.source), adapter)
    response = await runner._busy_steer_command(event, 'key', event.source)
    assert 'queued' in response.lower()
    assert adapter._pending_messages['key'].text == 'first queued'
    queued = runner._queued_events['key'][0]
    assert queued.text == WORDS and queued.source == event.source
    assert queued.media_urls == [PATH] and queued.media_types == ['audio/ogg']
    assert queued._gateway_recoverable_audio_paths == [PATH]
    assert queued.reply_to_message_id == event.reply_to_message_id
    assert queued.channel_context == event.channel_context
    assert typed_text_for_event(queued) == WORDS
    assert event.text == '/steer ' + WORDS
