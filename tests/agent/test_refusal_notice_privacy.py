"""Refusals expose visible text once, never private reasoning (30 Sep incident)."""
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

PRIVATE = "PRIVATE-REASONING-7f3a"


def response(content=None, reasoning=PRIVATE, finish_reason="content_filter", refusal=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=content, reasoning=reasoning, reasoning_content=None,
        reasoning_details=None, tool_calls=None, refusal=refusal,
    ), finish_reason=finish_reason)], usage=None, model="test-model")


@pytest.fixture
def agent(tmp_path, monkeypatch):
    from run_agent import AIAgent
    from hermes_state import SessionDB
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("")
    (tmp_path / "config.yaml").write_text("{}")
    db = SessionDB(db_path=tmp_path / "state.db")
    a = AIAgent(model="test-model", provider="openrouter", api_key="test-only", session_db=db,
                base_url="https://example.invalid/v1", platform="discord",
                quiet_mode=True, skip_context_files=True, skip_memory=True)
    a._disable_streaming = True
    a.compression_enabled = False
    a._try_activate_fallback = lambda **kwargs: False
    a._has_pending_fallback = lambda: False
    monkeypatch.setattr("agent.turn_context._maybe_title_session_at_turn_start", lambda *args, **kwargs: None)
    yield a
    db.close()


def run(agent, monkeypatch, reply):
    statuses = []
    agent.status_callback = lambda *args: statuses.append(str(args))
    call = Mock(side_effect=reply if isinstance(reply, Exception) else None,
                return_value=reply)
    monkeypatch.setattr(agent, "_interruptible_api_call", call)
    result = agent.run_conversation("hello")
    return result, statuses, call


def no_leak(result, statuses):
    assert PRIVATE not in (result.get("final_response") or "")
    assert PRIVATE not in (result.get("error") or "")
    assert all(PRIVATE not in text for text in statuses)


@pytest.mark.parametrize("content", [None, f"<think>{PRIVATE}</think>",
    f"<thinking>{PRIVATE}</thinking>I cannot help with that."])
def test_refusal_quotes_only_visible_text(agent, monkeypatch, content):
    result, statuses, call = run(agent, monkeypatch, response(content))
    no_leak(result, statuses)
    assert result["failed"]
    assert "content_policy_blocked" in result["error"]
    assert call.call_count == 1
    assert not [s for s in statuses if "safety" in s.lower() or "declined" in s.lower()]
    if content and "I cannot" in content:
        assert "I cannot help with that." in result["final_response"]
    else:
        assert "no explanation" in result["final_response"]


def test_refusal_field_remains_visible(agent, monkeypatch):
    result, statuses, _ = run(agent, monkeypatch, response(finish_reason="stop", refusal="I cannot help."))
    no_leak(result, statuses)
    assert "I cannot help." in result["final_response"]


def test_refusal_reasoning_is_bounded_in_local_log(agent, monkeypatch, caplog):
    with caplog.at_level(logging.INFO):
        result, statuses, _ = run(agent, monkeypatch, response(reasoning=PRIVATE + "x" * 900))
    no_leak(result, statuses)
    records = [r.getMessage() for r in caplog.records if "Refusal reasoning" in r.getMessage()]
    assert len(records) == 1
    assert PRIVATE in records[0]
    assert len(records[0]) < 700


class PolicyError(Exception):
    status_code = 400


@pytest.mark.parametrize("suffix", [f"<thinking>{PRIVATE}</thinking>", f"<think>{PRIVATE}", ""])
def test_provider_error_is_sanitized_once(agent, monkeypatch, suffix):
    result, statuses, _ = run(agent, monkeypatch,
        PolicyError("Output blocked by content filtering policy (content_filter) " + suffix))
    no_leak(result, statuses)
    assert result["failed"]
    assert "content_policy_blocked" in result["error"]
    assert not [s for s in statuses if "safety" in s.lower() or "refused" in s.lower()]


def test_sanitizer_failure_cannot_publish_raw_refusal(agent, monkeypatch):
    monkeypatch.setattr(agent, "_strip_think_blocks", Mock(side_effect=ValueError("unparseable")))
    result, statuses, _ = run(agent, monkeypatch, response(f"<think>{PRIVATE}</think>"))
    no_leak(result, statuses)
    assert "no explanation" in result["final_response"]


@pytest.mark.parametrize("platform", ["discord", "cli"])
def test_empty_terminal_keeps_upstream_privacy(agent, monkeypatch, platform):
    agent.platform = platform
    result, statuses, _ = run(agent, monkeypatch, response(content="", finish_reason="stop"))
    no_leak(result, statuses)


def test_normal_reply_still_stores_reasoning(agent, monkeypatch):
    result, statuses, _ = run(agent, monkeypatch, response(content="Visible answer.", finish_reason="stop"))
    assert result["final_response"] == "Visible answer."
    assert result["messages"][-1]["reasoning"] == PRIVATE
    stored = agent._session_db.get_messages(agent.session_id)
    assert any(row.get("reasoning") == PRIVATE for row in stored)
