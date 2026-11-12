from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.authored_turn_context import capture, snapshots, render
from agent.agent_runtime_helpers import _merge_consecutive_users
from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX, _SUMMARY_END_MARKER, _INFLIGHT_TASK_REPLAY_HEADER
from agent.micro_compaction import MicroCompactionMixin
from tests.agent.test_authored_admission import configure




def test_real_compactor_output_still_builds_verified_request(tmp_path, monkeypatch):
    from agent.authored_turn_context import check_request
    from agent.turn_context import build_api_messages
    from tests.agent.test_cron_inflight_prompt_reappend import _compress, _tool_pairs
    from tests.agent.test_authored_wire import make_agent
    agent = make_agent(tmp_path, monkeypatch)
    original = {"role": "user", "content": "Pending job", "message_uid": "one"}
    capture(agent, original)
    result = _compress([{"role": "system", "content": "Fixture"}, original, *_tool_pairs(40)])
    wire, _ = build_api_messages(agent, result, current_turn_user_idx=len(result)-1,
        ext_prefetch_cache="", plugin_user_context="", moa_config=None, active_system_prompt="SYSTEM")
    check_request(agent, {"messages": wire})
    assert any(snapshots(row) == snapshots(original) for row in result)


def rows(tmp_path):
    agent = configure(SimpleNamespace(), tmp_path)
    first = {"role": "user", "content": "first", "message_uid": "one"}
    second = {"role": "user", "content": "", "message_uid": "two"}
    for row in (first, second):
        capture(agent, row)
        row["_db_persisted"] = True
    return first, second


@pytest.mark.parametrize("route", ["sequence", "micro"])
def test_merge_preserves_snapshots_including_empty_occurrence(tmp_path, route):
    first, second = rows(tmp_path)
    expected = snapshots(first) + snapshots(second)
    if route == "sequence": result, _ = _merge_consecutive_users([first, second])
    else: result = MicroCompactionMixin()._merge_adjacent_user_turns([first, second])
    assert len(result) == 1
    assert snapshots(result[0]) == expected
    assert not result[0].get("_db_persisted")
    assert result[0]["message_uid"] == "one"
    assert result[0]["_absorbed_message_uids"] == ["two"]
    assert all(s["body"] in render(result[0], result[0]["content"]) for s in expected)


@pytest.mark.parametrize("role,already", [("user", False), ("assistant", False), ("user", True)])
def test_handoff_and_protected_head_restatement_keep_identity(tmp_path, role, already):
    first, _ = rows(tmp_path)
    carrier = {"role": role, "content": SUMMARY_PREFIX + "\nsummary\n" + _SUMMARY_END_MARKER,
               "_compressed_summary": True, "message_uid": "carrier"}
    if already: carrier["content"] += "\n" + _INFLIGHT_TASK_REPLAY_HEADER + "\nfirst"
    compressor = ContextCompressor.__new__(ContextCompressor)
    compressor.quiet_mode = True
    result = compressor._reappend_inflight_user_task([carrier], first)
    surviving = next(r for r in reversed(result) if r["role"] == "user")
    assert snapshots(surviving) == snapshots(first)
    assert first["message_uid"] == surviving["message_uid"] or first["message_uid"] in surviving.get("_absorbed_message_uids", [])
    again = compressor._reappend_inflight_user_task(deepcopy(result), first)
    surviving = next(r for r in reversed(again) if r["role"] == "user")
    assert snapshots(surviving) == snapshots(first)
