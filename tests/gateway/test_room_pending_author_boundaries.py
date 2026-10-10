"""Residual author boundaries in 0.21.6 queue entry points."""
import pytest
from tests.gateway.test_shared_room_steer_identity import _busy_runner, _event, _zoe, _sol, _opus, _src
from tests.gateway.restart_test_helpers import RestartTestAdapter
from gateway.platforms.base import MessageType


def test_photo_followups_from_different_authors_do_not_merge():
    runner, adapter, _ = _busy_runner()
    first, second = _event('first', _zoe()), _event('second', _src(name='Other', user_id='222'))
    first.message_type = second.message_type = MessageType.PHOTO
    runner._queue_or_replace_pending_event('k', first)
    runner._queue_or_replace_pending_event('k', second)
    assert first.text == 'first'
    assert runner._peek_session_state('k').conversation.queued_events == [second]


@pytest.mark.asyncio
async def test_three_authors_survive_adapter_debounce_overflow():
    runner, _, _ = _busy_runner()
    adapter = RestartTestAdapter()
    adapter.gateway_runner = runner
    first, second, third = _event('one', _zoe()), _event('two', _sol()), _event('three', _opus())
    try:
        adapter._pending_messages['k'] = first
        await adapter._queue_text_debounce('k', second)
        await adapter._queue_text_debounce('k', third)
        assert adapter._pending_messages['k'] is first
        buffered = adapter._text_debounce_store()['k'].event
        overflow = runner._peek_session_state('k').conversation.queued_events
        assert buffered is second
        assert overflow == [third]
    finally:
        adapter._discard_text_debounce('k')


def test_priority_pending_keeps_different_human_authors():
    runner, adapter, _ = _busy_runner()
    first, second = _event('first', _zoe()), _event('second', _src(name='Other', user_id='222'))
    runner._hm_merge_pending_for_source(first.source, 'k', first, merge_text=True)
    runner._hm_merge_pending_for_source(second.source, 'k', second, merge_text=True)
    assert first.text == 'first'
    assert runner._peek_session_state('k').conversation.queued_events == [second]


@pytest.mark.asyncio
async def test_recursion_cap_keeps_different_authors():
    from types import SimpleNamespace
    runner, adapter, _ = _busy_runner()
    first, second = _event('first', _zoe()), _event('second', _sol())
    adapter._pending_messages['k'] = first
    ctx = SimpleNamespace(source=first.source, session_id='s', session_key='k', run_generation=0,
                          _interrupt_depth=100, history=[], _status_thread_metadata={}, result_holder=[None])
    await runner._run_agent_queued_followup(ctx, adapter, second.text, second, 'answer', {}, None)
    assert adapter._pending_messages['k'] is first
    assert runner._peek_session_state('k').conversation.queued_events == [second]


def test_voice_speaker_does_not_inherit_bound_bot_origin():
    from types import SimpleNamespace
    from gateway.run_voice import GatewayVoiceMixin
    class Adapter:
        _voice_sources = {1: _sol().to_dict()}
        _client = None
    adapter = Adapter()
    source = GatewayVoiceMixin._voice_input_source(adapter, 1, 123, '2')
    assert source.is_bot is False


def test_debounce_does_not_merge_synthetic_actor_with_human():
    adapter = RestartTestAdapter()
    assert not adapter._can_merge_text_debounce_events(
        _event('one', _zoe()), _event('notice', _zoe(), internal=True))
