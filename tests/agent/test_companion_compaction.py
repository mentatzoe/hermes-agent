"""Row 7: real config -> compressor -> bounded, identity-aware checkpoint.

Only the auxiliary model is fake. No assertion treats its prose as model evidence.
"""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from run_agent import AIAgent


@pytest.fixture
def companion_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    identity = tmp_path / "identity.md"
    identity.write_text("Aleph is the companion. Zoe uses they/them.\n")
    config = {"model": {"context_length": 200000}, "compression": {
        "threshold": 0.55, "protect_first": 2, "protect_last": 8,
        "companion": {"enabled": True, "agent_name": "Aleph", "user_name": "Zoe",
                      "people": ["Zoe (they/them)", "Aleph"],
                      "identity_files": [str(identity)], "letter_budget_tokens": 2100,
                      "appendix_budget_tokens": 1300,
                      "lanes": [{"name": "test lane", "match": {"chat_ids": ["room"]},
                                 "note": "Keep the open promise."}]}}}
    (tmp_path / "config.yaml").write_text(json.dumps(config))
    with (patch("model_tools.get_tool_definitions", return_value=[]),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("agent.process_bootstrap.OpenAI"),
          patch("hermes_cli.plugins.discover_plugins")):
        agent = AIAgent(model="test-model", provider="openai", api_key="offline-test",
                        base_url="http://localhost:1/v1", skip_memory=True,
                        skip_context_files=True, quiet_mode=True)
        yield agent, identity
        agent.close()


@pytest.mark.parametrize("tail_mode", ["lean", "legacy"])
def test_compaction_preserves_identity_letter_and_names(companion_agent, monkeypatch, tail_mode):
    agent, identity = companion_agent
    c = agent.context_compressor
    c.tail_mode = tail_mode
    turns = []
    for i in range(35):
        turns.extend([{"role": "user", "content": f"Remember our promise {i}: keep /tmp/example-{i}.md."},
                      {"role": "assistant", "content": "I will carry that forward. " * 70}])
    captured = []

    def summary_call(**kwargs):
        captured.append(kwargs["messages"][0]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
            content="## Letter To My Next Self\nI promised to carry this forward.\n\n## In Flight\nAleph owes the follow-up.\n\n## Factual Appendix (for retrieval)\n/tmp/example-34.md"))])

    monkeypatch.setattr("agent.context_compressor.call_llm", summary_call)
    result = c._generate_summary(turns, memory_context="Fresh peer card: Zoe uses they/them.")
    assert "## Who Is Here" in result
    assert "Zoe uses they/them" in result
    assert "Fresh peer card" in result
    assert "## Letter To My Next Self" in captured[0]
    assert "WHO IS IN THIS CONVERSATION" in captured[0]
    assert "Zoe (they/them)" in captured[0]
    assert "[ZOE]: Remember our promise" in captured[0]
    assert "[ALEPH]: I will carry" in captured[0]
    assert "~2100 tokens" in captured[0]
    assert "~1300 tokens" in captured[0]
    assert "characters\n  are not the people" in captured[0]
    assert "## Who Is Here" not in c._previous_summary
    if tail_mode == "lean":
        assert "## Zoe's Messages (verbatim, newest first)" in result
        assert "> Remember our promise 34: keep /tmp/example-34.md." in result
    assert len(captured) == 1


def test_rehydrated_identity_is_re_sourced_not_summarized(companion_agent, monkeypatch):
    agent, identity = companion_agent
    c = agent.context_compressor
    assert getattr(c, "_companion", None) is not None
    from agent.companion_compaction import IDENTITY_HEADING, IDENTITY_END
    c._previous_summary = f"{IDENTITY_HEADING}\nSTALE IDENTITY\n{IDENTITY_END}\n## Letter To My Next Self\nEarlier promise."
    identity.write_text("FRESH IDENTITY: Zoe uses they/them; Aleph remembers.\n")
    prompts = []
    def summarize(prompt, started):
        prompts.append(prompt)
        return "## Letter To My Next Self\nThe promise is still open."
    monkeypatch.setattr(c, "_call_summary_llm", summarize)
    result = c._generate_summary([{"role": "user", "content": "Keep that promise."}])
    assert "STALE IDENTITY" not in prompts[0]
    assert "Earlier promise." in prompts[0]
    assert "FRESH IDENTITY" in result
    assert result.count(IDENTITY_HEADING) == 1
    assert IDENTITY_HEADING not in c._previous_summary


def test_archived_user_message_is_quoted_once():
    from agent.context_compressor import _build_verbatim_user_section
    row = {"role": "user", "content": "Keep my exact words.", "platform_message_id": "msg-1"}
    distinct = {**row, "platform_message_id": "msg-2"}
    result = _build_verbatim_user_section([row, row.copy(), distinct])
    assert result.count("> Keep my exact words.") == 2
