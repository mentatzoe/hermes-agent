"""Thinking disable must never reach a model that rejects it (2026-10-01).

Incident: cron 089c7665efc9 (claude-opus-5-5 via cpa-gui, an OpenAI-compatible
CLIProxyAPI relay at 127.0.0.1:8317, logged as provider=custom) died at 01:46
with HTTP 400 '"thinking.type.disabled" is not supported for this model.
Use "thinking.type.adaptive" and "output_config.effort"'.

Observed code mechanism: thinking-only finish_reason=length arms
``_ephemeral_reasoning_off``. The custom profile translates the resulting
``{"enabled": False, "effort": "none"}`` into ``reasoning_effort="none"``;
CLIProxyAPI translates it into ``thinking.type="disabled"`` and Opus 5.5
rejects that shape. The historical HTTP 400 was captured; its preceding
thinking-only truncation is a timing-based inference, not a logged event.
These tests exercise synthetic responses, not a replay of that missing event.

``_accepts_thinking_disable`` only guarded the native Messages wire, which
this route never used.  Fix: one shared verdict consulted by every builder of
a disable, the continuation override degrades to low effort for those models,
and the 400 itself is recoverable (remember + resend with thinking on).
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import anthropic_adapter
from agent.error_classifier import FailoverReason, classify_api_error

OPUS_55_400 = (
    '"thinking.type.disabled" is not supported for this model. Use '
    '"thinking.type.adaptive" and "output_config.effort" to control '
    "thinking behavior."
)


@pytest.fixture(autouse=True)
def _clear_runtime_verdicts():
    anthropic_adapter._RUNTIME_DISABLE_REJECTED_MODELS.clear()
    yield
    anthropic_adapter._RUNTIME_DISABLE_REJECTED_MODELS.clear()


class _FakeApiError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(f"Error code: {status_code} - {message}")
        self.status_code = status_code
        self.body = {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": message},
        }
        self.response = None


# ── The shared verdict ───────────────────────────────────────────────────


class TestSharedVerdict:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-5-5",
            "claude-opus-5.5",
            "anthropic/claude-opus-5.5",
            "claude-opus-5-5-20260901",
            "claude-fable-5-1",
        ],
    )
    def test_reasoning_mandatory_models_reject_a_disable(self, model):
        assert anthropic_adapter.thinking_disable_is_rejected(model) is True

    @pytest.mark.parametrize(
        "model", ["claude-opus-5", "claude-sonnet-5", "claude-opus-4-8", "glm-5.3", None, ""]
    )
    def test_other_models_are_unaffected(self, model):
        assert anthropic_adapter.thinking_disable_is_rejected(model) is False

    def test_runtime_learning_covers_unlisted_models(self):
        assert anthropic_adapter.thinking_disable_is_rejected("claude-opus-6") is False
        anthropic_adapter.remember_thinking_disable_rejected("claude-opus-6")
        assert anthropic_adapter.thinking_disable_is_rejected("claude-opus-6") is True

    def test_native_messages_wire_keeps_the_omission_for_opus_55(self):
        kwargs = anthropic_adapter.build_anthropic_kwargs(
            model="claude-opus-5-5",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": False, "effort": "none"},
        )
        assert "thinking" not in kwargs

    def test_native_messages_wire_still_disables_opus_5(self):
        kwargs = anthropic_adapter.build_anthropic_kwargs(
            model="claude-opus-5",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": False, "effort": "none"},
        )
        assert kwargs["thinking"] == {"type": "disabled"}


# ── Every builder of a disable consults it ───────────────────────────────


class TestWireBuilders:
    def _custom_profile(self):
        import model_tools  # noqa: F401  (registers bundled provider plugins)
        import providers

        return providers.get_provider_profile("custom")

    def test_custom_profile_omits_reasoning_effort_none_for_opus_55(self):
        profile = self._custom_profile()
        _, top = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "none"},
            model="claude-opus-5-5",
            base_url="http://127.0.0.1:8317/v1",
        )
        assert "reasoning_effort" not in top

    def test_custom_profile_still_disables_other_models(self):
        profile = self._custom_profile()
        _, top = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "none"},
            model="glm-5.3",
            base_url="http://127.0.0.1:8317/v1",
        )
        assert top["reasoning_effort"] == "none"

    def test_custom_profile_enable_path_unchanged_for_opus_55(self):
        profile = self._custom_profile()
        _, top = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "max"},
            model="claude-opus-5-5",
            base_url="http://127.0.0.1:8317/v1",
        )
        assert top["reasoning_effort"] == "max"

    def test_legacy_chat_completions_path_omits_the_disable(self):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="claude-opus-5-5",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_config={"enabled": False, "effort": "none"},
            supports_reasoning=True,
        )
        assert "reasoning" not in (kwargs.get("extra_body") or {})

    def test_legacy_chat_completions_path_still_disables_others(self):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="glm-5.3",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_config={"enabled": False, "effort": "none"},
            supports_reasoning=True,
        )
        assert kwargs["extra_body"]["reasoning"] == {"enabled": False, "effort": "none"}


# ── The one-shot continuation override ───────────────────────────────────


class TestContinuationOverride:
    def test_override_degrades_to_low_effort_for_opus_55(self):
        from agent.chat_completion_helpers import _reasoning_config_for_wire

        agent = SimpleNamespace(
            model="claude-opus-5-5",
            reasoning_config={"enabled": True, "effort": "max"},
            _ephemeral_reasoning_off=True,
        )
        cfg = _reasoning_config_for_wire(agent)
        assert cfg == {"enabled": True, "effort": "low"}
        assert agent._ephemeral_reasoning_off is False
        assert _reasoning_config_for_wire(agent) == {"enabled": True, "effort": "max"}

    def test_override_still_disables_models_that_accept_it(self):
        from agent.chat_completion_helpers import _reasoning_config_for_wire

        agent = SimpleNamespace(
            model="glm-5.3",
            reasoning_config={"enabled": True, "effort": "high"},
            _ephemeral_reasoning_off=True,
        )
        assert _reasoning_config_for_wire(agent) == {"enabled": False, "effort": "none"}


# ── The 400 itself degrades gracefully ───────────────────────────────────


class TestClassification:
    def test_incident_400_is_recoverable(self):
        result = classify_api_error(
            _FakeApiError(400, OPUS_55_400), provider="custom", model="claude-opus-5-5"
        )
        assert result.reason == FailoverReason.thinking_disable_rejected
        # Deterministic: recovered once by the loop's branch, never by backoff.
        assert result.retryable is False
        assert result.should_compress is False

    @pytest.mark.parametrize(
        "message",
        ["Thinking is mandatory for this model", "Reasoning is mandatory for this model"],
    )
    def test_sibling_wordings(self, message):
        result = classify_api_error(_FakeApiError(400, message), provider="nous", model="x")
        assert result.reason == FailoverReason.thinking_disable_rejected

    def test_unrelated_400_is_not_swallowed(self):
        result = classify_api_error(
            _FakeApiError(400, "Unknown parameter: 'foo'"), provider="custom", model="m"
        )
        assert result.reason != FailoverReason.thinking_disable_rejected


def _mock_response(content: str):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="claude-opus-5-5", usage=None)


def _make_agent(model: str):
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=MagicMock()),
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
    ):
        agent = AIAgent(
            api_key="fx",
            base_url="https://relay.invalid/v1",
            provider="custom",
            model=model,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        return agent


def _run(agent, fake_api_call):
    with (
        patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("run_agent.OpenAI", return_value=MagicMock()),
        patch("agent.agent_runtime_helpers.time.sleep"),
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
    ):
        return agent.run_conversation("write the proposal")


class TestLoopRecovery:
    def test_disable_rejection_recovers_and_completes_the_turn(self):
        """Simulates a model the static list does not know yet: the first
        request carries a disable, the provider 400s, the loop remembers the
        verdict and resends with thinking on, and the turn completes."""
        agent = _make_agent("claude-opus-7")  # unlisted: only runtime learning saves it
        agent._api_max_retries = 3
        agent.reasoning_config = {"enabled": True, "effort": "max"}
        agent._ephemeral_reasoning_off = True
        seen = []

        def fake_api_call(api_kwargs):
            seen.append(copy.deepcopy({k: v for k, v in api_kwargs.items() if k != "messages"}))
            if len(seen) == 1:
                raise _FakeApiError(400, OPUS_55_400)
            return _mock_response("Here is the proposal.")

        # The stale-flag guard clears a flag armed before the turn; re-arm it
        # inside the turn by patching the reset point's consumer instead.
        from agent import chat_completion_helpers as cch

        original = cch._consume_ephemeral_reasoning_off
        armed = {"once": True}

        def consume(a):
            if armed.pop("once", False):
                return True
            return original(a)

        with patch.object(cch, "_consume_ephemeral_reasoning_off", side_effect=consume):
            result = _run(agent, fake_api_call)

        assert result["completed"] is True, result.get("error")
        assert result["final_response"] == "Here is the proposal."
        assert len(seen) == 2
        assert seen[0].get("reasoning_effort") == "none"
        assert seen[1].get("reasoning_effort") == "low", seen[1]
        assert agent.reasoning_config == {"enabled": True, "effort": "max"}
        assert anthropic_adapter.thinking_disable_is_rejected(
            "claude-opus-7", provider="custom", base_url=agent.base_url,
        ) is True

    def test_known_mandatory_model_never_sends_the_disable(self):
        """claude-opus-5-5 is listed: the override never reaches the wire as a
        disable, so there is no 400 to recover from."""
        agent = _make_agent("claude-opus-5-5")
        agent._api_max_retries = 3
        agent.reasoning_config = {"enabled": True, "effort": "max"}
        seen = []

        def fake_api_call(api_kwargs):
            seen.append(api_kwargs.get("reasoning_effort"))
            return _mock_response("ok")

        from agent import chat_completion_helpers as cch

        original = cch._consume_ephemeral_reasoning_off
        armed = {"once": True}

        def consume(a):
            if armed.pop("once", False):
                return True
            return original(a)

        with patch.object(cch, "_consume_ephemeral_reasoning_off", side_effect=consume):
            result = _run(agent, fake_api_call)

        assert result["completed"] is True
        assert seen == ["low"], seen

    def test_second_rejection_still_surfaces(self):
        """One-shot guard: a model that keeps 400ing after the resend must not
        loop forever."""
        agent = _make_agent("claude-opus-7")
        agent._api_max_retries = 3
        agent.reasoning_config = {"enabled": False, "effort": "none"}
        calls = []

        def fake_api_call(api_kwargs):
            calls.append(api_kwargs.get("reasoning_effort"))
            raise _FakeApiError(400, OPUS_55_400)

        result = _run(agent, fake_api_call)
        assert result.get("completed") is not True
        assert len(calls) == 2, calls
        # Configured "off" was converted to the lowest effort for the resend.
        assert calls[1] == "low"
