"""Every pending-steer consumer retains per-call ownership, not rendered labels."""
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
from agent.turn_iteration_prep import _inject_steer_after_newest_tool_result
from gateway.run import GatewayRunner
from gateway.turn_context import TurnContext
from tests.agent.test_steer import _bare_agent
from tests.gateway.test_internal_event_pin_wiring import KEY, _capture, _human_source, _make_runner
from tests.gateway.test_pending_steer_drain import Adapter
from gateway.config import Platform, PlatformConfig

ENTRIES = [
    {'text': '[Alex] first', 'facts': {'sender_id': 'alex', 'typed_user_content': 'first',
                                     'recoverable_audio_paths': ['/audio/first.ogg']}},
    {'text': 'legacy text'},
    {'text': '[Blair] second', 'facts': {'sender_id': 'blair', 'typed_user_content': 'second',
                                      'recoverable_audio_paths': ['/audio/second.ogg']}},
]


@pytest.mark.parametrize('consumer', ['post-batch', 'pre-api', 'redirect-to-steer'])
def test_in_turn_consumers_preserve_each_call(consumer):
    agent = _bare_agent()
    entries = deepcopy(ENTRIES)
    agent._executing_tools = consumer == 'redirect-to-steer'
    verb = agent.redirect if agent._executing_tools else agent.steer
    for entry in entries:
        assert verb(entry['text'], **({'facts': entry['facts']} if 'facts' in entry else {}))
    messages = [{'role': 'user', 'content': 'current request'},
                {'role': 'tool', 'tool_call_id': 'call', 'content': 'done'}]
    if consumer == 'pre-api':
        _inject_steer_after_newest_tool_result(
            agent, messages, agent._drain_pending_steer(with_entries=True), current_turn_user_idx=0,
        )
    else:
        apply_pending_steer_to_tool_results(agent, messages, 1)
    assert messages[-1]['steer_entries'] == ENTRIES
    assert agent._turn_recoverable_audio_paths == ['/audio/first.ogg', '/audio/second.ogg']
    assert agent._drain_pending_steer() is None


@pytest.mark.asyncio
@pytest.mark.parametrize('delivery', ['alone', 'queued-human', 'queued-internal', 'interrupt', 'recursion-cap'])
async def test_late_consumer_preserves_facts_through_fifo_and_turn_runner(monkeypatch, delivery):
    runner = _make_runner(monkeypatch)
    runner._draining = False
    calls = []
    _capture(runner, calls)
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._persist_prompt_pins = AsyncMock()
    runner._session_key_for_source = lambda source: KEY
    source = _human_source()
    adapter = Adapter(PlatformConfig(enabled=True), Platform.DISCORD)
    runner._delivery_adapter_for = lambda source: adapter
    runner._intake_adapter_for = lambda source: None
    from gateway.platforms.event import MessageEvent
    if delivery.startswith('queued-'):
        runner._enqueue_fifo(KEY, MessageEvent(text='later input', source=source,
            internal=delivery == 'queued-internal'), adapter)
    result = {'final_response': 'done', 'messages': [],
              'pending_steer': '\n'.join(entry['text'] for entry in ENTRIES),
              'pending_steer_entries': deepcopy(ENTRIES)}
    if delivery == 'interrupt':
        result.update(interrupted=True, interrupt_message='interrupt goes first')
    event, text = await runner._run_agent_drain_pending(result, adapter, source, KEY)
    if delivery == 'interrupt':
        assert text == 'interrupt goes first'
        event, text = await runner._run_agent_drain_pending({'final_response': 'done'}, adapter, source, KEY)
    ctx = TurnContext(source=source, session_id='sess-wiring', session_key=KEY,
                      run_generation=1, history=[])
    if delivery == 'recursion-cap':
        ctx._interrupt_depth = runner._MAX_INTERRUPT_DEPTH
        await runner._run_agent_queued_followup(ctx, adapter, text, event, '', result, None)
        assert not calls
        event, text = await runner._run_agent_drain_pending({'final_response': 'done'}, adapter, source, KEY)
        ctx._interrupt_depth = 0
    await runner._run_agent_queued_followup(ctx, adapter, text, event, '', result, None)
    assert calls[-1]['message'] == '\n'.join(entry['text'] for entry in ENTRIES)
    assert calls[-1]['typed_user_message'] == 'first\n\nsecond'
    assert calls[-1]['recoverable_audio_paths'] == ['/audio/first.ogg', '/audio/second.ogg']
    assert calls[-1]['persist_user_display_metadata']['steer_entries'] == ENTRIES
    if delivery.startswith('queued-'):
        event, text = await runner._run_agent_drain_pending({'final_response': 'done'}, adapter, source, KEY)
        assert text == 'later input'
    assert runner._queue_depth(KEY, adapter=adapter) == 0
