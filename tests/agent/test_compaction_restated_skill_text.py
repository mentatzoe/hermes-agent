"""A cron request restated after compaction points at its skills instead of repeating their text.

Regression for aleph-vault#203: a cron job's prompt opens with the full text of every skill the job loads.
When compaction fired mid-run, the restatement after the summary's end marker carried all of it again
(five skills, about 152,000 characters, behind a 16,000-character summary). Only the summary model is fake.
"""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import _SUMMARY_END_MARKER

JOB_SENTINEL = "CRON_JOB_PROMPT_triage_the_inbox_and_report"
BODIES = {"restate-alpha": "ALPHA_SKILL_BODY_SENTINEL", "restate-beta": "BETA_SKILL_BODY_SENTINEL"}


@pytest.fixture
def cron_prompt(tmp_path, monkeypatch):
    """The real cron prompt for a job with two skills; restate-beta declares a config value."""
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLES_DIR", str(home / "skill-bundles"))
    import tools.skills_tool as skills_tool
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", home / "skills")
    monkeypatch.setattr(skills_tool, "HERMES_HOME", home)
    config = "metadata:\n  hermes:\n    config:\n      - key: restate.data_dir\n        description: Data folder\n"
    for name, body in BODIES.items():
        (home / "skills" / name).mkdir()
        (home / "skills" / name / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: test\n{config if name == 'restate-beta' else ''}---\n\n"
            + f"{body}\n" * 400, encoding="utf-8")
    (home / "config.yaml").write_text(json.dumps({
        "model": {"context_length": 200000},
        "compression": {"threshold": 0.55, "protect_first_n": 2, "protect_last_n": 4},
        "skills": {"config": {"restate": {"data_dir": "/srv/restate-data"}}}}), encoding="utf-8")
    from cron import scheduler
    return scheduler._build_job_prompt({"id": "job-203", "skills": list(BODIES), "prompt": JOB_SENTINEL})


def _compact(prompt, monkeypatch, *, can_reload_skills):
    """Compact a mid-run cron transcript through the host entry point; return the compacted messages."""
    from run_agent import AIAgent

    with (patch("model_tools.get_tool_definitions", return_value=[]),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("agent.process_bootstrap.OpenAI"),
          patch("hermes_cli.plugins.discover_plugins")):
        agent = AIAgent(model="test-model", provider="openai", api_key="offline-test",
                        base_url="http://localhost:1/v1", skip_memory=True,
                        skip_context_files=True, quiet_mode=True, session_db=None)
    agent._invalidate_system_prompt = lambda: None
    agent._build_system_prompt = lambda _message: "sys"
    # A cron agent has skill_view whenever its toolsets include skills; the facade reads this set.
    agent.valid_tool_names = {"skill_view"} if can_reload_skills else set()
    monkeypatch.setattr("agent.context_compressor.call_llm", lambda **_kw: SimpleNamespace(choices=[
        SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="## Summary\nRan triage steps."))]))
    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": prompt, "message_uid": "job-prompt-uid"}]
    for i in range(40):
        messages += [{"role": "assistant", "content": f"step {i}", "tool_calls": [
                         {"id": f"c{i}", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
                     {"role": "tool", "tool_call_id": f"c{i}", "content": "tool output " * 200}]
    return agent._compress_context(messages, "sys", approx_tokens=150_000, force=True)[0]


def _text(messages):
    return "\n".join(m["content"] if isinstance(m.get("content"), str) else str(m.get("content")) for m in messages)


def _restatement(messages):
    """The restated request: the text after the replay header, on whichever row carries it."""
    from agent.context_compressor import _INFLIGHT_TASK_REPLAY_HEADER

    carrier = next(t for t in (_text([m]) for m in messages) if _INFLIGHT_TASK_REPLAY_HEADER in t)
    return carrier.rsplit(_INFLIGHT_TASK_REPLAY_HEADER, 1)[1]


def test_restated_cron_request_points_at_its_skills_instead_of_repeating_them(cron_prompt, monkeypatch):
    compacted = _compact(cron_prompt, monkeypatch, can_reload_skills=True)
    restated = _restatement(compacted)
    assert _text(compacted).rsplit(_SUMMARY_END_MARKER, 1)[1].count(JOB_SENTINEL) == 1
    assert JOB_SENTINEL in restated
    for name, body in BODIES.items():
        assert f"reload with skill_view(name='{name}')" in restated
        assert body not in _text(compacted)
    # skill_view does not return the resolved config block, so the restatement keeps it.
    assert "restate.data_dir = /srv/restate-data" in restated
    assert len(restated) < 3_000


def test_request_is_restated_whole_when_the_agent_cannot_reload_skills(cron_prompt, monkeypatch):
    restated = _restatement(_compact(cron_prompt, monkeypatch, can_reload_skills=False))
    assert JOB_SENTINEL in restated
    assert all(body in restated for body in BODIES.values())
