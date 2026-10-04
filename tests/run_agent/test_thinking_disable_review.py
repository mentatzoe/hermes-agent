"""Offline integration regressions for thinking-disable request repair.

Synthetic SDK responses exercise the real continuation producer, custom
profile/ChatCompletionsTransport consumer, dispatch, and exception loop. They
are not a replay or direct capture of the historical unlogged truncation.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import anthropic_adapter as aa
from agent.error_classifier import FailoverReason, classify_api_error
from tests.run_agent.test_thinking_disable_rejected_20261001 import (
    OPUS_55_400, _FakeApiError, _make_agent,
)

BASE = "https://relay.invalid/v1"


@pytest.fixture(autouse=True)
def clear_learning():
    aa._RUNTIME_DISABLE_REJECTED_MODELS.clear()
    yield
    aa._RUNTIME_DISABLE_REJECTED_MODELS.clear()


def response(content="done", finish="stop"):
    return SimpleNamespace(
        id="chatcmpl-synthetic", model="synthetic",
        choices=[SimpleNamespace(index=0, finish_reason=finish, message=SimpleNamespace(
            content=content, tool_calls=None, reasoning_content="synthetic reasoning",
        ))], usage=None,
    )


def run_sdk(agent, outcomes, seen=None, history=None):
    """Keep real request construction/dispatch; substitute only SDK create."""
    pending = iter(outcomes)

    def create(**kwargs):
        if seen is not None:
            seen.append(copy.deepcopy(kwargs))
        outcome = next(pending)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    agent.client.chat.completions.create.side_effect = create
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
    ):
        return agent.run_conversation("write an answer", conversation_history=history)


def make(model="claude-opus-5-5", config=None):
    agent = _make_agent(model)
    agent.base_url = BASE
    agent._api_max_retries = 1  # deterministic repair must not use backoff budget
    agent.reasoning_config = config or {"enabled": True, "effort": "max"}
    agent._cached_system_prompt = "Offline synthetic test."
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


@pytest.mark.parametrize("message", [
    "Audit logging cannot be disabled for this account",
    "Safety filtering cannot be disabled",
    "Tool validation cannot be disabled while reasoning is enabled",
    "Unknown parameter: thinking.type.disabled",
    "thinking.type.disabled must be a boolean",
    "thinking.type.disabled is allowed; tool streaming is unsupported",
])
def test_unrelated_disable_or_schema_errors_are_not_thinking_rejections(message):
    assert not aa.is_thinking_disable_rejection(message)
    assert classify_api_error(_FakeApiError(400, message), provider="custom").reason != (
        FailoverReason.thinking_disable_rejected
    )


@pytest.mark.parametrize("message", [
    "Thinking cannot be disabled for this model",
    "Reasoning cannot be disabled",
    OPUS_55_400,
])
def test_subject_specific_rejections(message):
    assert aa.is_thinking_disable_rejection(message)


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_rejection_wording_requires_http_400(status):
    assert classify_api_error(_FakeApiError(status, OPUS_55_400)).reason != (
        FailoverReason.thinking_disable_rejected
    )


def test_runtime_learning_is_exact_model_and_route_scoped():
    model = "tenant-a/claude-future.1"
    aa.remember_thinking_disable_rejected(model, provider="custom", base_url=BASE)
    assert aa.thinking_disable_is_rejected(model, provider="custom", base_url=BASE)
    for other_model, provider, base in [
        (model, "openrouter", BASE),
        (model, "custom", "https://different.invalid/v1"),
        ("tenant-b/claude-future.1", "custom", BASE),
        ("claude-future.1", "custom", BASE),
        ("tenant-a/claude-future-1", "custom", BASE),
    ]:
        assert not aa.thinking_disable_is_rejected(other_model, provider=provider, base_url=base)


def test_real_truncation_to_custom_wire_and_later_turn_restores_max(caplog):
    caplog.set_level("INFO", logger="agent.conversation_loop")
    agent = make()
    original = copy.deepcopy(agent.reasoning_config)
    seen = []
    result = run_sdk(agent, [response("", "length"), response()], seen)
    assert result["completed"] is True, result.get("error")
    assert [k.get("reasoning_effort") for k in seen] == ["max", "low"]
    assert all("reasoning" not in (k.get("extra_body") or {}) for k in seen)
    assert all((k.get("extra_body") or {}).get("thinking") != {"type": "disabled"} for k in seen)
    assert not any(m["role"] == "assistant" and not m.get("content") for m in result["messages"])
    assert agent.reasoning_config == original
    next_seen = []
    later = run_sdk(agent, [response("later answer")], next_seen, result["messages"])
    assert later["completed"] is True
    assert [k.get("reasoning_effort") for k in next_seen] == ["max"]
    assert agent.reasoning_config == original


def test_unknown_model_repair_precedes_credential_recovery_and_eager_fallback():
    agent = make("claude-future-7")
    seen = []
    with (
        patch.object(agent, "_recover_with_credential_pool", return_value=(True, False)) as pool,
        patch.object(agent, "_try_activate_fallback", return_value=True) as fallback,
    ):
        result = run_sdk(agent, [
            response("", "length"), _FakeApiError(400, OPUS_55_400), response(),
        ], seen)
    assert result["completed"] is True, result.get("error")
    assert [k.get("reasoning_effort") for k in seen] == ["max", "none", "low"]
    pool.assert_not_called()
    fallback.assert_not_called()
    assert agent.reasoning_config == {"enabled": True, "effort": "max"}
    assert aa.thinking_disable_is_rejected(agent.model, provider="custom", base_url=BASE)
    assert not aa.thinking_disable_is_rejected(agent.model, provider="openrouter", base_url=BASE)
    later = []
    assert run_sdk(agent, [response()], later, result["messages"])["completed"] is True
    assert [k.get("reasoning_effort") for k in later] == ["max"]


def test_configured_off_repair_does_not_mutate_config():
    original = {"enabled": False, "effort": "none"}
    agent = make("claude-future-8", copy.deepcopy(original))
    seen = []
    result = run_sdk(agent, [_FakeApiError(400, OPUS_55_400), response()], seen)
    assert result["completed"] is True, result.get("error")
    assert [k.get("reasoning_effort") for k in seen] == ["none", "low"]
    assert agent.reasoning_config == original
    # Switch to a distinct model: user-configured off must still apply there.
    agent.model = "glm-unrelated"
    later = []
    assert run_sdk(agent, [response()], later)["completed"] is True
    assert [k.get("reasoning_effort") for k in later] == ["none"]


@pytest.mark.parametrize("fallback_available", [False, True])
def test_second_rejection_aborts_or_falls_back_without_loop(fallback_available):
    agent = make("claude-future-9", {"enabled": False, "effort": "none"})
    seen = []
    agent._fallback_chain = [{"provider": "custom", "model": "glm-unrelated"}] if fallback_available else []
    agent._fallback_index = 0

    def activate():
        if agent._fallback_index >= len(agent._fallback_chain):
            return False
        agent._fallback_index += 1
        agent.model = "glm-unrelated"
        return True

    outcomes = [_FakeApiError(400, OPUS_55_400), _FakeApiError(400, OPUS_55_400)]
    if fallback_available:
        outcomes.append(response("fallback answer"))
    with patch.object(agent, "_try_activate_fallback", side_effect=activate) as fallback:
        result = run_sdk(agent, outcomes, seen)
    if fallback_available:
        assert result["completed"] is True, result.get("error")
        assert [k.get("reasoning_effort") for k in seen] == ["none", "low", "none"]
        assert fallback.call_count == 1
    else:
        assert result.get("completed") is not True
        assert len(seen) == 2


def test_learned_custom_and_auxiliary_consumer_builders_share_route_scope():
    from agent.auxiliary_client import _build_call_kwargs
    from agent.transports.chat_completions import ChatCompletionsTransport
    import model_tools  # noqa: F401
    from providers import get_provider_profile

    model = "claude-future-10"
    aa.remember_thinking_disable_rejected(model, provider="custom", base_url=BASE)
    cfg = {"enabled": False, "effort": "none"}
    for base, expected in [(BASE, None), ("https://other.invalid/v1", "none")]:
        kwargs = ChatCompletionsTransport().build_kwargs(
            model=model, messages=[{"role": "user", "content": "hi"}],
            reasoning_config=cfg, provider_profile=get_provider_profile("custom"), base_url=base,
        )
        assert kwargs.get("reasoning_effort") == expected
        aux = _build_call_kwargs(
            model=model, messages=[{"role": "user", "content": "hi"}],
            provider="custom", base_url=base, reasoning_config=cfg,
        )
        assert aux.get("reasoning_effort") == expected


@pytest.mark.parametrize("config", [
    {"enabled": False, "effort": "none"},
    {"enabled": True, "effort": "none"},
])
def test_auxiliary_generic_fallback_omits_both_disable_spellings(config):
    from agent.auxiliary_client import _build_call_kwargs
    kwargs = _build_call_kwargs(
        model="claude-opus-5-5", provider="unregistered-synthetic",
        base_url=BASE, messages=[{"role": "user", "content": "hi"}],
        reasoning_config=config,
    )
    assert "reasoning" not in (kwargs.get("extra_body") or {})


def test_native_transport_learning_keeps_original_catalog_identity():
    from agent.transports.anthropic import AnthropicTransport
    model = "anthropic/claude-opus-5.7"
    aa.remember_thinking_disable_rejected(model, provider="native-synthetic", base_url=BASE)
    for provider, expected_disable in [("native-synthetic", False), ("different", True)]:
        kwargs = AnthropicTransport().build_kwargs(
            model=model, messages=[{"role": "user", "content": "hi"}],
            reasoning_config={"enabled": False, "effort": "none"},
            provider_name=provider, base_url=BASE,
        )
        assert (kwargs.get("thinking") == {"type": "disabled"}) is expected_disable


def test_custom_named_route_context_is_forwarded_to_profile():
    from agent.transports.chat_completions import ChatCompletionsTransport
    import model_tools  # noqa: F401
    from providers import get_provider_profile
    model = "claude-opus-5.7"
    aa.remember_thinking_disable_rejected(model, provider="custom:synthetic", base_url=BASE)
    for provider, expected in [("custom:synthetic", None), ("custom", "none")]:
        kwargs = ChatCompletionsTransport().build_kwargs(
            model=model, messages=[{"role": "user", "content": "hi"}],
            reasoning_config={"enabled": False, "effort": "none"},
            provider_profile=get_provider_profile("custom"),
            provider_name=provider, base_url=BASE,
        )
        assert kwargs.get("reasoning_effort") == expected


def test_native_auxiliary_projection_keeps_learning_scoped_and_config_intact():
    from agent.auxiliary_client import _build_call_kwargs
    model = "claude-opus-5.7"
    original = {"enabled": False, "effort": "none"}
    aa.remember_thinking_disable_rejected(model, provider="anthropic", base_url=BASE)
    for base, expected in [
        (BASE, {"enabled": True, "effort": "low"}),
        ("https://different.invalid/v1", original),
    ]:
        kwargs = _build_call_kwargs(
            model=model, provider="anthropic", base_url=base,
            messages=[{"role": "user", "content": "hi"}], reasoning_config=original,
        )
        assert kwargs["_reasoning_config"] == expected
        assert original == {"enabled": False, "effort": "none"}


def test_opus_55_static_match_does_not_include_a_different_point_release():
    # Static knowledge also prevents future-looking prefix collisions.
    assert not aa.thinking_disable_is_rejected("claude-opus-5-50")
