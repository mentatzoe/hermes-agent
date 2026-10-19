"""Companion compaction stands in a recent saved copy of the memory provider's context.

Regression for aleph-vault#202: three paths compact with no provider context, so the summary
writer never saw the Honcho peer cards. The gateway's hygiene helper is built without a memory
provider, Honcho turns itself off for cron, Kanban and webhook sessions, and its card fetch gives
up after two seconds. Only the summary model is fake here.
"""
import json
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.companion_compaction import IDENTITY_END, IDENTITY_HEADING

CARDS = "User peer card:\nZoe uses they/them, always.\n\nAI peer card:\nAleph is the companion."
SAVED = os.path.join("state", "companion-compaction-provider-context.json")


def _config(**companion):
    return {"model": {"context_length": 200000}, "compression": {
        "threshold": 0.55, "protect_first": 2, "protect_last": 4,
        "companion": {"enabled": True, "agent_name": "Aleph", "user_name": "Zoe",
                      "people": ["Zoe (they/them, always)", "Aleph"], **companion}}}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(json.dumps(_config()))
    return tmp_path


def _agent(memory_manager):
    from run_agent import AIAgent

    with (patch("model_tools.get_tool_definitions", return_value=[]),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("agent.process_bootstrap.OpenAI"),
          patch("hermes_cli.plugins.discover_plugins")):
        # skip_memory=True is how the gateway builds its hygiene helper.
        agent = AIAgent(model="test-model", provider="openai", api_key="offline-test",
                        base_url="http://localhost:1/v1", skip_memory=True,
                        skip_context_files=True, quiet_mode=True, session_db=None)
    agent._memory_manager = memory_manager
    agent._invalidate_system_prompt = lambda: None
    agent._build_system_prompt = lambda _message: "new-system-prompt"
    return agent


def _compact(agent, monkeypatch):
    """Run the host compaction entry point; return (summarizer prompt, summary message text)."""
    prompts = []

    def summary_call(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
            content="## Letter To My Next Self\nThey asked me to keep the promise.\n\n## In Flight\nNone."))])

    monkeypatch.setattr("agent.context_compressor.call_llm", summary_call)
    messages = [{"role": "system", "content": "sys"}]
    for i in range(30):
        messages += [{"role": "user", "content": f"Promise {i}: keep /tmp/example-{i}.md."},
                     {"role": "assistant", "content": "I will carry that forward. " * 90}]
    compressed = agent._compress_context(messages, "sys", approx_tokens=150_000, force=True)[0]
    summary = next(m["content"] for m in compressed if IDENTITY_HEADING in str(m.get("content")))
    assert len(prompts) == 1
    return prompts[0], summary


def _header(summary):
    return summary[summary.index(IDENTITY_HEADING):summary.index(IDENTITY_END)]


def _provider(text):
    manager = MagicMock()
    manager.on_pre_compress.return_value = text
    return manager


@pytest.mark.parametrize("missing", ["no provider (gateway hygiene)", "provider returned nothing"])
def test_saved_cards_reach_a_compaction_without_provider_context(home, monkeypatch, missing):
    live_prompt, _ = _compact(_agent(_provider(CARDS)), monkeypatch)
    assert "Zoe uses they/them, always." in live_prompt
    assert "Saved copy" not in live_prompt
    saved = home / SAVED
    assert saved.is_file() and saved.stat().st_mode & 0o077 == 0

    later = _agent(None if missing.startswith("no provider") else _provider(""))
    prompt, summary = _compact(later, monkeypatch)
    assert "MEMORY PROVIDER CONTEXT" in prompt
    assert "Zoe uses they/them, always." in prompt and "Saved copy from" in prompt
    assert "Zoe uses they/them, always." in _header(summary)
    assert "Saved copy from" in _header(summary)


@pytest.mark.parametrize("change", ["too old", "other people", "turned off"])
def test_saved_cards_are_not_used_when_stale_for_other_people_or_off(home, monkeypatch, change):
    _compact(_agent(_provider(CARDS)), monkeypatch)
    if change == "too old":
        record = json.loads((home / SAVED).read_text())
        record["saved_at"] = time.time() - 2 * 3600
        (home / SAVED).write_text(json.dumps(record))
        (home / "config.yaml").write_text(json.dumps(_config(saved_context_max_age_hours=1)))
    elif change == "other people":
        (home / "config.yaml").write_text(json.dumps(_config(user_name="Sam", people=["Sam (she/her)"])))
    else:
        (home / "config.yaml").write_text(json.dumps(_config(saved_context_max_age_hours=0)))
    prompt, summary = _compact(_agent(None), monkeypatch)
    assert "MEMORY PROVIDER CONTEXT" not in prompt
    assert "Zoe uses they/them, always." not in _header(summary)


def test_no_copy_when_companion_names_no_user(home, monkeypatch):
    # Without a user name the copy can't tell one person's cards from another's.
    (home / "config.yaml").write_text(json.dumps(_config(user_name="")))
    _compact(_agent(_provider(CARDS)), monkeypatch)
    assert not (home / SAVED).exists()
    prompt, _ = _compact(_agent(None), monkeypatch)
    assert "MEMORY PROVIDER CONTEXT" not in prompt


def test_unreadable_saved_copy_does_not_block_compaction(home, monkeypatch):
    (home / "state").mkdir()
    (home / SAVED).write_text("{not json")
    prompt, summary = _compact(_agent(None), monkeypatch)
    assert "MEMORY PROVIDER CONTEXT" not in prompt
    assert "## Letter To My Next Self" in summary


def test_each_profile_reads_only_its_own_saved_cards(home, tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    _compact(_agent(_provider(CARDS)), monkeypatch)
    other = tmp_path / "profiles" / "other"
    other.mkdir(parents=True)
    (other / "config.yaml").write_text(json.dumps(_config()))
    token = set_hermes_home_override(other)
    try:
        prompt, _ = _compact(_agent(None), monkeypatch)
    finally:
        reset_hermes_home_override(token)
    assert "MEMORY PROVIDER CONTEXT" not in prompt
    assert not (other / SAVED).exists()
