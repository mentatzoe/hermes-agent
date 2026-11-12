"""Authored context through real prologue, queue admissions and persisted replay."""
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.authored_turn_context import Source, snapshots, block
from agent.mid_turn_input import mid_turn_human_row
from agent.interrupt_control import InterruptControlMixin
from agent.turn_context import build_api_messages
from tests.agent.test_turn_context import _FakeAgent, _build
from tests.agent.test_authored_turn_context import PROFILE, NAMES


def configure(agent, tmp_path):
    path = tmp_path / "profile.md"
    path.write_bytes(PROFILE.encode())
    agent._authored_turn_source = Source.from_config(
        {"file": str(path), "sections": list(NAMES)}, tmp_path)
    agent._authored_turn_expected = {}
    return agent


@pytest.mark.parametrize("mode", ["none", "empty", "error"])
@pytest.mark.parametrize("staged", [False, True])
def test_prologue_admits_independently_of_recall(tmp_path, mode, staged):
    agent = configure(_FakeAgent(), tmp_path)
    if mode != "none":
        from unittest.mock import MagicMock
        agent._memory_manager = MagicMock()
        agent._memory_manager.has_providers = True
        agent._memory_manager.prefetch_all.return_value = ""
        if mode == "error": agent._memory_manager.prefetch_all.side_effect = RuntimeError("offline")
    query = "What did we decide about the fixture?"
    if staged:
        agent._pending_cli_user_message = {"role": "user", "content": query, "display_metadata": {"actor": "human"}}
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        ctx = _build(agent, user_message=query)
    if mode != "none":
        agent._memory_manager.prefetch_all.assert_called_once_with(query, session_id=agent.session_id)
    row = ctx.messages[ctx.current_turn_user_idx]
    assert row["content"] == query
    assert len(snapshots(row)) == 1
    assert block(snapshots(row)[0]) in row["api_content"]
    if staged: assert row["display_metadata"]["actor"] == "human"




@pytest.mark.parametrize("method", ["steer", "redirect"])
def test_native_codex_midturn_refuses_unverified_delivery(tmp_path, method):
    from unittest.mock import MagicMock
    agent = configure(QueueAgent(), tmp_path)
    agent.api_mode = "codex_app_server"
    agent._codex_session = SimpleNamespace(request_steer=MagicMock(return_value=True))
    assert getattr(agent, method)("correction") is False
    agent._codex_session.request_steer.assert_not_called()


class QueueAgent(InterruptControlMixin):
    def __init__(self):
        self._interrupt_requested = False
        self._model_request_active = threading.Event()
        self._model_request_active.set()


@pytest.mark.parametrize("method,slot", [("steer", "_pending_steer"), ("redirect", "_pending_redirect")])
def test_queue_snapshots_each_input_before_coalescing(tmp_path, method, slot):
    agent = configure(QueueAgent(), tmp_path)
    assert getattr(agent, method)("one", facts={"typed_user_message": "one"})
    agent._authored_turn_source.path.write_bytes(PROFILE.replace("Meaning.", "Revised.").encode())
    assert getattr(agent, method)("two", facts={"typed_user_message": "two"})
    assert not agent._authored_turn_expected  # queued != delivered to current request
    entries = getattr(agent, slot)
    row = mid_turn_human_row(agent, entries, content="framed one and two")
    assert len(snapshots(row)) == 2
    assert "Revised." not in snapshots(row)[0]["body"]
    assert "Revised." in snapshots(row)[1]["body"]
    assert [e["text"] for e in row["steer_entries"]] == ["one", "two"]
    assert len(agent._authored_turn_expected) == 2


def test_helper_inherits_binding_without_memory_or_parent_expectations(tmp_path):
    from agent.read_only_memory import inherit_read_only_memory
    parent = configure(_FakeAgent(), tmp_path)
    parent._authored_turn_expected = {"parent": "must not leak"}
    child = SimpleNamespace(_memory_manager=None)
    inherit_read_only_memory(parent, child)
    assert child._authored_turn_source is parent._authored_turn_source
    assert child._authored_turn_expected == {}
    assert child._memory_manager is None
