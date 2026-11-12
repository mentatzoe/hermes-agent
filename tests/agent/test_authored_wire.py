"""Final provider-shaped requests, not just source/render unit tests."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.authored_turn_context import AuthoredContextUnavailable, block, snapshots, text_of
from agent.turn_context import _stage_turn_user_message, _stamp_api_content_sidecar, build_api_messages, _PER_TURN_RESET_STATE
from agent.turn_api_request import build_api_request
from run_agent import AIAgent
from tests.agent.test_authored_turn_context import PROFILE, NAMES


def make_agent(tmp_path, monkeypatch, mode="chat_completions"):
    import json
    path = tmp_path / "profile.md"
    path.write_bytes(PROFILE.encode())
    (tmp_path / "config.yaml").write_text(json.dumps({"agent": {"authored_turn_context": {
        "file": str(path), "sections": list(NAMES)}}}))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "state.db")
    with patch("model_tools.get_tool_definitions", return_value=[]), patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(api_key="fixture", base_url="https://fixture.invalid/v1", model="test-model",
                        api_mode=mode, quiet_mode=True, skip_context_files=True, skip_memory=True,
                        skip_background_review=True, max_iterations=2, session_db=db)
    agent._cached_system_prompt = "SYSTEM"
    agent.compression_enabled = False
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.save_trajectories = False
    for name, value in _PER_TURN_RESET_STATE: setattr(agent, name, value)
    agent._authored_turn_expected = {}
    agent._current_turn_timestamp = 1.0
    return agent


def request(agent):
    row, _ = _stage_turn_user_message(agent, "hello", None, None, None, None, None)
    _stamp_api_content_sidecar(agent, [row], 0, "RECALL", "HOOK", preflight_compressed=False)
    wire, system = build_api_messages(agent, [row], current_turn_user_idx=0,
        ext_prefetch_cache="RECALL", plugin_user_context="HOOK", moa_config=None, active_system_prompt="SYSTEM")
    return row, dict(api_messages=wire, _moa_prepared_request=None, tools_for_api=agent.tools,
        system_message=system, messages=[row], original_user_message="hello", approx_tokens=0,
        total_chars=0, retry_count=0, api_call_count=1, api_request_id="req", api_start_time=0.,
        effective_task_id="task", turn_id="turn")


@pytest.mark.parametrize("mode", ["chat_completions", "codex_responses", "anthropic_messages"])
@pytest.mark.parametrize("drop", [False, True])
def test_final_request_after_middleware_and_conversion(tmp_path, monkeypatch, mode, drop):
    agent = make_agent(tmp_path, monkeypatch, mode)
    row, kwargs = request(agent)
    from hermes_cli.middleware import apply_llm_request_middleware
    def transform(payload, **kw):
        result = apply_llm_request_middleware(payload, **kw)
        if drop:
            for message in result.payload.get("messages", result.payload.get("input", [])):
                if message.get("role") == "user": message["content"] = "dropped"
        return result
    with patch("hermes_cli.middleware.apply_llm_request_middleware", side_effect=transform):
        if drop:
            with pytest.raises(AuthoredContextUnavailable, match="lost"):
                build_api_request(agent, **kwargs)
        else:
            sent = build_api_request(agent, **kwargs).api_kwargs
            visible = "\n".join(text_of(m.get("content")) for m in sent.get("messages", sent.get("input", [])) if m.get("role") == "user")
            assert block(snapshots(row)[0]) in visible
            assert visible.count("RECALL") == visible.count("HOOK") == 1
            assert row["content"] == "hello"
            assert "authored_turn_context" not in str(sent.keys())


def test_real_chat_turn_and_resume_persist_exact_request(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, monkeypatch)
    calls = []
    def reply(**kw):
        calls.append(deepcopy(kw))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=None,
            reasoning=None, reasoning_content=None), finish_reason="stop")], usage=None, model="test-model")
    from unittest.mock import MagicMock
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = reply
    result = agent.run_conversation("hello")
    assert result["final_response"] == "done"
    rows = agent._session_db.get_messages(agent.session_id)
    user = next(row for row in rows if row["role"] == "user")
    sent = next(row for row in calls[0]["messages"] if row["role"] == "user")
    assert user["content"] == "hello"
    assert user["api_content"] == sent["content"]
    assert block(snapshots(user)[0]) in sent["content"]
    history = agent._session_db.get_messages_as_conversation(agent.session_id)
    result = agent.run_conversation("again", conversation_history=history)
    assert result["final_response"] == "done"
    assert next(row for row in calls[-1]["messages"] if row["role"] == "user")["content"] == sent["content"]


def test_codex_opener_reaches_session_input(tmp_path, monkeypatch):
    from agent.transports.codex_app_server_session import CodexAppServerSession, TurnResult
    captured = []
    monkeypatch.setattr(CodexAppServerSession, "ensure_started", lambda self: "thread-fixture")
    def turn(self, user_input, **kw):
        captured.append(user_input)
        return TurnResult(final_text="done", projected_messages=[{"role": "assistant", "content": "done"}])
    monkeypatch.setattr(CodexAppServerSession, "run_turn", turn)
    agent = make_agent(tmp_path, monkeypatch, "codex_app_server")
    result = agent.run_conversation("hello")
    assert result["final_response"] == "done"
    user = next(row for row in result["messages"] if row["role"] == "user")
    assert block(snapshots(user)[0]) in captured[0]


def test_moa_holds_before_preparer_or_inner_model(tmp_path, monkeypatch):
    from tests.agent.test_turn_context import _build
    agent = make_agent(tmp_path, monkeypatch)
    agent.provider = "moa"
    with pytest.raises(AuthoredContextUnavailable, match="MoA"):
        _build(agent, moa_active=True)
