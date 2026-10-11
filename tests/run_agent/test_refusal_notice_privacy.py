"""A refusal notice quotes only what the model said out loud.

Incident (2026-09-30, shared Discord room): the provider refused a turn with
``finish_reason=content_filter`` and no visible text. The refusal handler fell
back to the model's *reasoning* and posted it to the room as "Model's
explanation:" — a bot's private working notes, which also misgendered the
person in the room. Each refusal also produced two posts: a lifecycle status
line and then the full notice.

Contracts pinned here, through the real ``run_conversation`` loop:

* finish_reason route and content_policy_blocked exception route: reasoning
  never reaches ``final_response``, ``error``, or any status callback; with no
  visible text the notice says "The model returned no explanation."
* visible refusal text is still quoted (inline ``<think>`` parts stripped).
* one refusal → one user-visible notice (no extra lifecycle status).
* the reasoning is kept in agent.log at INFO, truncated.
* the empty-response terminal only shows a reasoning excerpt on operator
  surfaces (CLI/TUI/desktop/ACP), never on chat platforms or cron.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

PRIVATE = "PRIVATE-REASONING-7f3a: I'm reviewing the room; she said"


def _response(content=None, finish_reason="content_filter", reasoning=None,
              reasoning_content=None, refusal=None):
    msg = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning=reasoning,
        reasoning_content=reasoning_content,
        refusal=refusal,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)],
        model="test/model",
        usage=None,
        id="resp_1",
    )


class _ProviderError(Exception):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


def _make_agent(platform=None):
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        kwargs = dict(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        if platform is not None:
            kwargs["platform"] = platform
        a = AIAgent(**kwargs)
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a.compression_enabled = False
    a.save_trajectories = False
    return a


def _run(agent, statuses):
    agent.status_callback = lambda kind, text=None: statuses.append((kind, str(text)))
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation("hello")


def _assert_no_leak(result, statuses, secret=PRIVATE):
    assert secret not in (result.get("final_response") or "")
    assert secret not in (result.get("error") or "")
    for _kind, text in statuses:
        assert secret not in text


# ── finish_reason=content_filter route ─────────────────────────────────────

@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_finish_reason_refusal_with_only_reasoning_quotes_no_explanation(field):
    agent = _make_agent(platform="discord")
    agent.client.chat.completions.create.return_value = _response(**{field: PRIVATE})
    statuses = []

    result = _run(agent, statuses)

    assert result["failed"] is True
    assert "content_policy_blocked" in result["error"]
    assert "declined to respond" in result["final_response"]
    assert "The model returned no explanation." in result["final_response"]
    assert "Model's explanation" not in result["final_response"]
    _assert_no_leak(result, statuses)
    assert agent.client.chat.completions.create.call_count == 1


def test_finish_reason_refusal_quotes_visible_text_only():
    agent = _make_agent(platform="discord")
    agent.client.chat.completions.create.return_value = _response(
        content=f"<think>{PRIVATE}</think>I can't help with that request.",
        reasoning=PRIVATE,
    )
    statuses = []

    result = _run(agent, statuses)

    assert "Model's explanation: I can't help with that request." in result["final_response"]
    _assert_no_leak(result, statuses)


def test_finish_reason_refusal_field_still_surfaced():
    """``message.refusal`` is visible text the provider meant for the user."""
    agent = _make_agent(platform="discord")
    agent.client.chat.completions.create.return_value = _response(
        finish_reason="stop", refusal="I won't help with that.", reasoning=PRIVATE,
    )
    statuses = []

    result = _run(agent, statuses)

    assert "I won't help with that." in result["final_response"]
    _assert_no_leak(result, statuses)


def test_finish_reason_refusal_is_one_notice():
    agent = _make_agent(platform="discord")
    agent.client.chat.completions.create.return_value = _response(reasoning=PRIVATE)
    statuses = []

    result = _run(agent, statuses)

    assert result["final_response"].count("declined to respond") == 1
    refusal_statuses = [
        text for _kind, text in statuses
        if "declined" in text.lower() or "safety" in text.lower()
    ]
    assert refusal_statuses == []


def test_finish_reason_refusal_logs_reasoning_at_info_truncated(caplog):
    agent = _make_agent(platform="discord")
    long_reasoning = PRIVATE + " " + ("x" * 900)
    agent.client.chat.completions.create.return_value = _response(reasoning=long_reasoning)
    statuses = []

    with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
        result = _run(agent, statuses)

    _assert_no_leak(result, statuses)
    records = [r for r in caplog.records if "Refusal reasoning" in r.getMessage()]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    logged = records[0].getMessage()
    assert PRIVATE in logged
    assert logged.endswith("...")
    assert len(logged) < len(long_reasoning)


def test_fallback_attempt_status_still_shown_before_refusal():
    """A fallback-attempt status says something different, so it may stay."""
    agent = _make_agent(platform="discord")
    agent.client.chat.completions.create.return_value = _response(reasoning=PRIVATE)
    statuses = []
    with (
        patch.object(agent, "_has_pending_fallback", return_value=True),
        patch.object(agent, "_try_activate_fallback", return_value=False),
    ):
        result = _run(agent, statuses)

    assert any("trying fallback" in text for _k, text in statuses)
    assert result["final_response"].count("declined to respond") == 1
    _assert_no_leak(result, statuses)


# ── content_policy_blocked exception route ─────────────────────────────────

def test_exception_route_strips_reasoning_echoed_in_provider_error():
    agent = _make_agent(platform="discord")
    agent.client.chat.completions.create.side_effect = _ProviderError(
        "Output blocked by content filtering policy (content_filter) "
        f"<thinking>{PRIVATE}</thinking>"
    )
    statuses = []

    result = _run(agent, statuses)

    assert result["failed"] is True
    assert "content_policy_blocked" in result["error"]
    assert "safety filter blocked this request" in result["final_response"]
    assert "content filtering policy" in result["final_response"]
    _assert_no_leak(result, statuses)


def test_exception_route_is_one_notice():
    agent = _make_agent(platform="discord")
    agent.client.chat.completions.create.side_effect = _ProviderError(
        "Output blocked by content filtering policy (content_filter)"
    )
    statuses = []

    result = _run(agent, statuses)

    assert result["final_response"].count("safety filter blocked") == 1
    assert [t for _k, t in statuses if "safety filter" in t.lower()] == []


def _stream_chunk(content=None, reasoning=None):
    delta = SimpleNamespace(
        content=content, tool_calls=None,
        reasoning_content=reasoning, reasoning=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, delta=delta, finish_reason=None)],
        model=None,
        usage=None,
    )


def test_exception_route_streamed_reasoning_then_filter_error(monkeypatch):
    """Streaming: the model thinks, then the provider kills the stream with
    a content-filter error before any visible text. The streamed reasoning
    must not appear in the notice or any status."""
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    agent = _make_agent(platform="discord")
    agent.stream_delta_callback = lambda _text: None  # a chat display consumer

    def _stream(*_a, **_kw):
        yield _stream_chunk(reasoning=PRIVATE)
        raise _ProviderError(
            "Output blocked by content filtering policy (content_filter)"
        )

    request_client = MagicMock()
    request_client.chat.completions.create.side_effect = _stream
    statuses = []
    with (
        patch.object(agent, "_create_request_openai_client", return_value=request_client),
        patch.object(agent, "_close_request_openai_client"),
    ):
        result = _run(agent, statuses)

    assert result["failed"] is True
    assert "content_policy_blocked" in result["error"]
    assert result["final_response"].count("safety filter blocked") == 1
    _assert_no_leak(result, statuses)


# ── empty-response terminal: reasoning excerpt only for the operator ───────

def _reasoning_only():
    return _response(content="", finish_reason="stop", reasoning=PRIVATE)


@pytest.mark.parametrize("platform", ["discord", "telegram", "cron", "api_server"])
def test_empty_terminal_never_shows_reasoning_on_shared_surfaces(platform, tmp_path, monkeypatch):
    agent = _make_agent(platform=platform)
    agent._disable_streaming = True
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda _kw: _reasoning_only())
    statuses = []

    result = _run(agent, statuses)

    _assert_no_leak(result, statuses)
    assert "only internal reasoning" not in (result.get("final_response") or "")


@pytest.mark.parametrize("platform", ["cli", "tui", "desktop", "acp", "subagent"])
def test_reasoning_excerpt_allowed_on_operator_surfaces(platform):
    from agent.conversation_loop import _reasoning_excerpt_allowed

    assert _reasoning_excerpt_allowed(SimpleNamespace(platform=platform)) is True


@pytest.mark.parametrize("platform", ["discord", "telegram", "slack", "cron", "api_server", "webhook"])
def test_reasoning_excerpt_refused_on_audience_surfaces(platform):
    from agent.conversation_loop import _reasoning_excerpt_allowed

    assert _reasoning_excerpt_allowed(SimpleNamespace(platform=platform)) is False


def test_empty_terminal_keeps_reasoning_excerpt_for_cli(monkeypatch):
    """The operator-surface feature (PR #48795 idea) still works."""
    agent = _make_agent(platform="cli")
    agent._disable_streaming = True
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda _kw: _reasoning_only())

    result = _run(agent, [])

    assert "only internal reasoning" in result["final_response"]
    assert PRIVATE in result["final_response"]
