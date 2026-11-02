"""Busy gateway events provide facts only to the parent; subagents receive text."""
import pytest

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from tests.agent.test_steer import _bare_agent
from tests.gateway.test_busy_steer_subagent_fanout import _event


@pytest.mark.asyncio
async def test_busy_facts_do_not_follow_subagent_fanout(tmp_path, monkeypatch):
    monkeypatch.setattr('gateway.run._hermes_home', tmp_path)
    runner = GatewayRunner(config=GatewayConfig())
    parent, child = _bare_agent(), _bare_agent()
    parent._active_children = [child]
    runner._session_state('key').turn.agent = parent
    event = _event('Only these human words.')
    result = await runner._resolve_busy_steer_or_redirect(event, 'key', 'steer', parent)
    assert result.steered
    parent_entries = parent._drain_pending_steer(with_entries=True)
    child_entries = child._drain_pending_steer(with_entries=True)
    assert parent_entries[0]['facts'] == {
        'sender_id': 'u', 'typed_user_content': 'Only these human words.',
        'recoverable_audio_paths': []}
    assert child_entries == [{'text': parent_entries[0]['text']}]


def test_internal_event_cannot_supply_human_attachment_facts():
    from gateway.typed_text import authored_steer_kwargs
    agent = _bare_agent()
    event = _event('[machinery]')
    event.internal = True
    event._gateway_recoverable_audio_paths = ['/audio/machinery.ogg']
    assert authored_steer_kwargs(agent.steer, event) == {'facts': {
        'sender_id': None, 'typed_user_content': '', 'recoverable_audio_paths': []}}
