"""Current-turn delivery must keep acceptance order and the original sender facts."""
from copy import deepcopy

import pytest

from agent.turn_iteration_prep import prepare_iteration
from tests.agent.test_repair_reanchors_current_turn_user_idx import _agent


def tool_round(call_id):
    return [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "probe", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": call_id, "content": "synthetic result"},
    ]


def accept(agent, sender, words):
    facts = {"sender_id": sender, "typed_user_content": words, "recoverable_audio_paths": []}
    assert agent.steer(words, facts=facts)
    return {"text": words, "facts": facts}


@pytest.fixture
def agent(tmp_path, monkeypatch):
    instance = _agent(tmp_path, monkeypatch)
    try:
        yield instance
    finally:
        instance._session_db.close()


def test_repeated_pre_request_drains_preserve_delivery_order(agent):
    messages = [{"role": "user", "content": "current request"}, *tool_round("live")]
    original = deepcopy(messages)
    expected = []
    for sender, words in [("first", "use red"), ("second", "actually use blue")]:
        expected.append(accept(agent, sender, words))
        prepared = prepare_iteration(agent, messages=messages, api_call_count=1,
                                     user_message="current request", current_turn_user_idx=0)
        assert prepared.current_turn_user_idx == 0
    delivered = [entry for row in messages for entry in row.get("steer_entries", [])]
    assert delivered == expected
    assert messages[:len(original)] == original
    assert agent._drain_pending_steer(with_entries=True) is None


@pytest.mark.parametrize('stale_idx', [None, 0, 2, 99])
@pytest.mark.parametrize('live_tool', [False, True])
def test_rebuilt_compaction_list_reanchors_before_delivery(agent, stale_idx, live_tool):
    # Fresh copies mirror compaction rebuilding the transcript and invalidating
    # the old coordinate; historical tools are never eligible delivery points.
    messages = deepcopy([
        {"role": "user", "content": "old request"}, *tool_round("old"),
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current request"},
        *(tool_round("live") if live_tool else []),
    ])
    history = deepcopy(messages[:4])
    expected = [accept(agent, "first", "first correction"),
                accept(agent, "second", "second correction")]
    prepared = prepare_iteration(agent, messages=messages, api_call_count=1,
                                 user_message="current request", current_turn_user_idx=stale_idx)
    assert prepared.current_turn_user_idx == 4
    assert agent._persist_user_message_idx == 4
    assert messages[:4] == history
    delivered = [entry for row in messages for entry in row.get("steer_entries", [])]
    assert delivered == (expected if live_tool else [])
    assert agent._drain_pending_steer(with_entries=True) == (None if live_tool else expected)


@pytest.mark.parametrize('user_message', [None, 'missing current request'])
def test_unestablished_boundary_requeues_original_entries_before_newer_calls(agent, monkeypatch, user_message):
    messages = [{"role": "user", "content": "old request"}, *tool_round("old")]
    history = deepcopy(messages)
    expected = [accept(agent, "first", "first correction"),
                accept(agent, "second", "second correction")]
    original_drain = agent._drain_pending_steer

    def drain_with_new_arrival(**kwargs):
        entries = original_drain(**kwargs)
        expected.append(accept(agent, "third", "newer correction"))
        return entries

    monkeypatch.setattr(agent, '_drain_pending_steer', drain_with_new_arrival)
    prepare_iteration(agent, messages=messages, api_call_count=1,
                      user_message=user_message, current_turn_user_idx=2)
    assert messages == history
    assert original_drain(with_entries=True) == expected
