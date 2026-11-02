"""Ingress owns the recoverable-audio list, independent of all model-facing text."""
from unittest.mock import AsyncMock
import pytest
from tests.gateway import test_typed_text as typed_tests
from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


@pytest.fixture
def runner():
    return typed_tests.runner.__wrapped__()


@pytest.mark.asyncio
async def test_only_current_failed_stt_paths_survive_preparation(runner):
    source = SessionSource(platform=Platform.DISCORD, chat_id='room', user_id='human', user_name='Alex')
    runner._should_echo_stt_transcripts = lambda: False
    runner._transcribe_one_clip = AsyncMock(side_effect=[
        ('Successful words.', '"Successful words."'),
        (None, '[some failure note without a parseable path]'),
    ])
    event = MessageEvent(text='Caption.', source=source, message_type=MessageType.VOICE,
                         media_urls=['/fixture/good.ogg', '/fixture/failed.ogg'],
                         media_types=['audio/ogg', 'audio/ogg'],
                         channel_context="[Recent channel messages]\n[The user sent a voice message: /fixture/other.ogg]")
    await runner._prepare_inbound_message_text(event=event, source=source, history=[])
    assert event._gateway_recoverable_audio_paths == ['/fixture/failed.ogg']
    assert event._gateway_typed_user_content == 'Caption.\n\nSuccessful words.'


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['pending', 'disabled', 'error', 'file', 'plain', 'internal'])
async def test_recoverable_attachment_facts_cover_all_ingress_paths(runner, monkeypatch, mode):
    source = SessionSource(platform=Platform.DISCORD, chat_id='room', user_id='human', user_name='Alex')
    runner._should_echo_stt_transcripts = lambda: False
    runner._transcribe_one_clip = AsyncMock(return_value=(None, 'failed'))
    if mode == 'error':
        runner._transcribe_one_clip.side_effect = RuntimeError('fixture unavailable')
    if mode == 'disabled':
        runner.config.stt_enabled = False
        monkeypatch.setattr('gateway.run._probe_audio_duration', AsyncMock(return_value=''))
    event = MessageEvent(text='Caption.', source=source,
                         message_type=MessageType.DOCUMENT if mode == 'file' else MessageType.VOICE,
                         media_urls=[] if mode == 'plain' else ['/fixture/current.ogg'],
                         media_types=[] if mode == 'plain' else ['audio/ogg'], internal=mode == 'internal')
    if mode == 'pending':
        await runner._transcribe_pending_audio_event_once(event)
    await runner._prepare_inbound_message_text(event=event, source=source, history=[])
    assert event._gateway_recoverable_audio_paths == ([] if mode in {'plain', 'internal'} else ['/fixture/current.ogg'])
    if mode == 'pending':
        assert runner._transcribe_one_clip.await_count == 1


def test_turn_runner_forwards_recoverable_paths(runner, monkeypatch):
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext
    monkeypatch.setattr('tools.approval.register_gateway_notify', lambda *_: None)
    monkeypatch.setattr('tools.approval.unregister_gateway_notify', lambda *_: None)
    source = SessionSource(platform=Platform.DISCORD, chat_id='room', user_id='human', user_name='Alex')
    ctx = TurnContext(source=source, message='Enriched caption', session_id='s', session_key='k',
                      typed_user_message='Caption.')
    ctx.recoverable_audio_paths = ['/fixture/failed.ogg']
    runner._consume_pending_native_image_paths = lambda *_: []
    class Agent:
        def run_conversation(self, _message, **kwargs):
            self.kwargs = kwargs
            return {'final_response':'Done.'}
    agent = Agent()
    TurnRunner(runner, ctx)._run_conversation_with_approval(agent, [], None, None, None)
    assert agent.kwargs['recoverable_audio_paths'] == ['/fixture/failed.ogg']
