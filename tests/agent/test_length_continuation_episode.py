"""Length-continuation budget is per truncation episode, not per turn (aleph-vault #213).

On 19 Oct a kanban worker on claude-opus-5-5 exited with "Response still truncated after 4
continuation attempts". Its first thinking-only truncation (19:18) was recovered: the
continuation produced a tool call and the tools ran. Truncations two to four came almost an
hour later, with dozens of completed tool rounds in between. ``length_continue_retries`` was
reset only when a continuation ended in text (``finish_text_response``), never when it ended
in a tool call, so a worker's whole run counted toward one four-try ceiling. The fourth
truncation of the run, only the third in a row, ended the worker.

A tool round is completed work: it starts a new episode, the same way it already resets
``truncated_tool_call_retries``. Four truncations in a row still end the turn.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import FINISH_REASON_LENGTH
from tests.agent.test_run_agent import _make_tool_defs, _mock_response, _mock_tool_call


@pytest.fixture()
def loop_agent():
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.compression_enabled = False
        a.save_trajectories = False
        a.context_compressor.context_length = 1_000_000
        return a


def _thinking_only_truncation():
    """``finish_reason=length`` with no visible text and no tool call: the model's reasoning
    used the whole output cap (the 19 Oct shape)."""
    return _mock_response(
        content=None, finish_reason=FINISH_REASON_LENGTH,
        usage={"prompt_tokens": 4_000, "completion_tokens": 65_536, "total_tokens": 69_536},
    )


def _tool_round(call_id):
    return _mock_response(
        content="", finish_reason="tool_calls",
        tool_calls=[_mock_tool_call(name="web_search", arguments="{}", call_id=call_id)],
    )


def _run(agent, message):
    with (
        patch("model_tools.handle_function_call", return_value="search result"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(message)


def test_truncation_recovered_by_a_tool_call_starts_a_new_episode(loop_agent):
    # The 19 Oct run: one recovered truncation, more tool work, then three truncations in a
    # row, and the fourth try writes the answer. Before the fix the third truncation of that
    # second episode was the fourth of the turn, and the turn ended without the answer.
    loop_agent.client.chat.completions.create.side_effect = [
        _thinking_only_truncation(),
        _tool_round("c1"),
        _tool_round("c2"),
        _thinking_only_truncation(),
        _thinking_only_truncation(),
        _thinking_only_truncation(),
        _mock_response(content="Here is the result.", finish_reason="stop"),
    ]
    result = _run(loop_agent, "do the long task")

    assert loop_agent.client.chat.completions.create.call_count == 7
    assert result["final_response"] == "Here is the result."
    assert result["completed"] is True
    assert "continuation attempts" not in (result.get("error") or "")


def test_four_truncations_in_a_row_still_end_the_turn(loop_agent):
    loop_agent.client.chat.completions.create.side_effect = [
        _tool_round("c1"),
        _thinking_only_truncation(),
        _thinking_only_truncation(),
        _thinking_only_truncation(),
        _thinking_only_truncation(),
        _mock_response(content="never requested", finish_reason="stop"),
    ]
    result = _run(loop_agent, "do the long task")

    assert loop_agent.client.chat.completions.create.call_count == 5
    assert result["partial"] is True
    assert "truncated after 4 continuation attempts" in (result.get("error") or "")


def test_each_new_episode_starts_at_the_first_rung_of_the_output_budget(loop_agent):
    # The boost ladder is indexed by the episode's retry count: a fresh episode's first
    # continuation gets rung 1, not the rung where the previous episode left off.
    loop_agent.max_tokens = 4096
    loop_agent.client.chat.completions.create.side_effect = [
        _thinking_only_truncation(),
        _thinking_only_truncation(),
        _tool_round("c1"),
        _thinking_only_truncation(),
        _mock_response(content="done", finish_reason="stop"),
    ]
    result = _run(loop_agent, "do the long task")

    sent = [
        call.kwargs.get("max_tokens") or call.kwargs.get("max_completion_tokens")
        for call in loop_agent.client.chat.completions.create.call_args_list
    ]
    assert result["final_response"] == "done"
    # sent[0]: the configured 4096. sent[1]: episode 1, rung 1 (8192). sent[2] is the
    # request that returned the tool round (rung 2). sent[3]: the configured 4096 again.
    # sent[4]: episode 2 starts again at rung 1 (8192), not at rung 3 (32768).
    assert sent[0] == 4096
    assert sent[1] == 8192
    assert sent[3] == 4096
    assert sent[4] == 8192


def test_continuation_without_a_sent_cap_keeps_the_room_that_ran_out(loop_agent):
    # The 19 Oct route sends no max_tokens; the relay's default (65,536 after the cpa-gui
    # rule) is what each thinking-only response used up. Every continuation must again go
    # out with no cap, never with the 8,192 / 16,384 / 32,768 rungs, and never above the
    # default with a doubled number Hermes can't check against the model's limit.
    assert loop_agent.max_tokens is None
    loop_agent.client.chat.completions.create.side_effect = [
        _thinking_only_truncation(),
        _thinking_only_truncation(),
        _thinking_only_truncation(),
        _mock_response(content="answer", finish_reason="stop"),
    ]
    result = _run(loop_agent, "do the long task")

    calls = loop_agent.client.chat.completions.create.call_args_list
    assert result["final_response"] == "answer"
    assert len(calls) == 4
    for call in calls:
        assert "max_tokens" not in call.kwargs and "max_completion_tokens" not in call.kwargs


def test_truncation_writes_one_log_line_with_the_cap_and_tokens(loop_agent, caplog):
    import logging
    loop_agent.client.chat.completions.create.side_effect = [
        _thinking_only_truncation(),
        _mock_response(content="answer", finish_reason="stop"),
    ]
    with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
        _run(loop_agent, "do the long task")
    lines = [r.getMessage() for r in caplog.records if "output cap sent" in r.getMessage()]
    assert len(lines) == 1
    assert "none (provider default)" in lines[0] and "output 65536 tokens" in lines[0]
