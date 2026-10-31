"""Host-owned sender facts survive the same drains as the model's steer text."""
from copy import deepcopy

from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
from agent.prompt_builder import steer_user_row
from tests.agent.test_steer import _bare_agent


def test_post_batch_keeps_per_call_facts_and_owned_paths():
    agent = _bare_agent()
    agent._turn_recoverable_audio_paths = ['/audio/opener.ogg']
    facts = {'sender_id': 'sender-a', 'typed_user_content': 'The first announcement.',
             'recoverable_audio_paths': ['/audio/first.ogg']}
    original = deepcopy(facts)
    assert agent.steer('[Alex] rendered first', facts=facts)
    facts['recoverable_audio_paths'].append('/audio/not-sent.ogg')
    assert agent.steer('legacy text')
    other = {'sender_id': 'sender-b', 'typed_user_content': 'The second announcement.',
             'recoverable_audio_paths': ['/audio/second.ogg']}
    assert agent.steer('[Blair] rendered second', facts=other)
    messages = [{'role': 'tool', 'content': 'done', 'tool_call_id': 'tool-1'}]
    apply_pending_steer_to_tool_results(agent, messages, 1)
    row = messages[-1]
    assert row['content'] == steer_user_row(
        '[Alex] rendered first\nlegacy text\n[Blair] rendered second')['content']
    assert row['steer_entries'] == [
        {'text': '[Alex] rendered first', 'facts': original},
        {'text': 'legacy text'},
        {'text': '[Blair] rendered second', 'facts': other},
    ]
    assert agent._turn_recoverable_audio_paths == [
        '/audio/opener.ogg', '/audio/first.ogg', '/audio/second.ogg']
    assert agent._drain_pending_steer() is None


def test_redirect_during_tools_preserves_facts():
    agent = _bare_agent()
    agent._executing_tools = True
    facts = {'sender_id': 'sender-a', 'typed_user_content': 'Use the second file.',
             'recoverable_audio_paths': []}
    assert agent.redirect('[Alex] Use the second file.', facts=facts)
    messages = [{'role': 'tool', 'content': 'done', 'tool_call_id': 'tool-1'}]
    apply_pending_steer_to_tool_results(agent, messages, 1)
    assert messages[-1]['steer_entries'] == [
        {'text': '[Alex] Use the second file.', 'facts': facts}]


def test_pre_api_requeue_preserves_entries_until_a_tool_row_exists():
    from agent.turn_iteration_prep import _inject_steer_after_newest_tool_result
    agent = _bare_agent()
    agent._turn_recoverable_audio_paths = []
    facts = {'sender_id': 'sender-a', 'typed_user_content': '',
             'recoverable_audio_paths': ['/audio/current.ogg']}
    agent.steer('[host audio notice]', facts=facts)
    pending = agent._drain_pending_steer(with_entries=True)
    # An arriving second steer cannot replace the first one's facts when the drain requeues.
    agent.steer('legacy text')
    messages = [{'role': 'user', 'content': 'opening text'}]
    _inject_steer_after_newest_tool_result(agent, messages, pending, current_turn_user_idx=0)
    assert agent._turn_recoverable_audio_paths == []
    messages.append({'role': 'tool', 'tool_call_id': 'one', 'content': 'done'})
    pending = agent._drain_pending_steer(with_entries=True)
    _inject_steer_after_newest_tool_result(agent, messages, pending, current_turn_user_idx=0)
    assert {entry['text']: entry.get('facts') for entry in messages[-1]['steer_entries']} == {
        '[host audio notice]': facts, 'legacy text': None}
    assert agent._turn_recoverable_audio_paths == ['/audio/current.ogg']
