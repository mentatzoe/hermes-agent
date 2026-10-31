"""A finalizer keeps late steer facts alongside its legacy text-only return."""
from copy import deepcopy

from tests.agent.test_run_agent import _mock_response
from tests.agent.test_turn_user_intervened import _loop_agent, _run


def test_finalizer_preserves_late_per_sender_facts_without_delivery():
    agent = _loop_agent()
    entries = [
        {'text': '[Alex] first', 'facts': {'sender_id': 'alex', 'typed_user_content': 'first',
                                          'recoverable_audio_paths': ['/audio/first.ogg']}},
        {'text': 'legacy'},
        {'text': '[Blair] second', 'facts': {'sender_id': 'blair', 'typed_user_content': 'second',
                                            'recoverable_audio_paths': ['/audio/second.ogg']}},
    ]
    expected = deepcopy(entries)

    def model_call(_):
        for entry in entries:
            agent.steer(entry['text'], **({'facts': entry['facts']} if 'facts' in entry else {}))
        entries[0]['facts']['typed_user_content'] = 'not sent'
        return _mock_response(content='Done.', finish_reason='stop')

    result = _run(agent, model_call)
    assert result['pending_steer'] == '[Alex] first\nlegacy\n[Blair] second'
    assert result.get('pending_steer_entries') == expected
    assert all('steer_entries' not in row for row in result['messages'])
    assert agent._drain_pending_steer() is None
    untouched = _run(agent, lambda _: _mock_response(content='Hi.', finish_reason='stop'))
    assert not untouched.get('pending_steer_entries')
