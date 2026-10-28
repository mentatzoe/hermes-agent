"""Background reviews with an external memory provider keep procedures in skills and personal
material in memory.

With an external provider (Honcho) registered, every review kind gets the provider's tools and the
procedural skill-review prompt; a skill review never turns a personal preference into a skill, even
when the provider's save tools are unavailable. Without an external provider nothing changes:
native-only reviews keep upstream's prompts.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import background_review as bg
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider, MemoryReviewBinding
from run_agent import AIAgent

PROVIDER_PROMPT = "Maintain attributed external memory using review_write."
SCOPES = [(True, False), (False, True), (True, True)]
# The preference-into-skill routing the procedural prompt replaces: the first three are the skill and
# combined prompts' own signals, the last two hide in the shared lesson-layer block.
OLD_ROUTING = ("FIRST-CLASS", "Be ACTIVE", "User-preference embedding", "embed the preference",
               "standing user preferences", "preferences for how the result should look",
               "THIS user's specifications")


def _schema(name):
    return {"name": name, "description": name, "parameters": {"type": "object", "properties": {}}}


def _tool(name, args=None):
    return SimpleNamespace(id="call-" + name, type="function", function=SimpleNamespace(
        name=name, arguments=json.dumps(args or {})))


class Provider(MemoryProvider):
    name = "external-test"

    def __init__(self):
        self.calls = []
        self.opt_in = True

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return [_schema("review_write")]

    def handle_tool_call(self, tool_name, args, **kwargs):
        raise AssertionError("a review must use the captured capability")

    def memory_review_binding(self, *, session_id=""):
        if not self.opt_in:
            return None

        def dispatch(name, args):
            self.calls.append((session_id, name, args))
            return json.dumps({"success": True})

        return MemoryReviewBinding(PROVIDER_PROMPT, ("review_write",), dispatch)


@pytest.fixture
def make_agent(monkeypatch):
    config = {"memory": {"memory_enabled": False, "user_profile_enabled": False, "nudge_interval": 2},
              "skills": {"creation_nudge_interval": 0}}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", MagicMock())
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kwargs: [
        {"type": "function", "function": _schema(n)} for n in ("skill_view", "skills_list", "skill_manage")])
    built = []

    def build(provider=None):
        manager = MemoryManager()
        if provider is not None:
            manager.add_provider(provider)
        agent = AIAgent(model="test/model", provider="openai", api_key="test-key",
                        base_url="https://example.invalid/v1", quiet_mode=True, skip_context_files=True,
                        memory_manager=manager, enabled_toolsets=["memory", "skills"], max_iterations=3)
        agent._cached_system_prompt = "Stable test prefix"
        agent._emit_auxiliary_failure = lambda label, error: pytest.fail(f"{label}: {error}")
        built.append((agent, manager))
        return agent

    yield build
    for agent, manager in built:
        manager.shutdown_all()
        agent.release_clients()


def _run_review(agent, monkeypatch, *, review_memory, review_skills, tool_calls):
    """Run one review through the real fork and return (prompt, fork, tool-result messages)."""
    seen = {}

    def review_turn(fork, **kwargs):
        seen["fork"] = fork
        seen["messages"] = []
        fork._execute_tool_calls_sequential(
            SimpleNamespace(tool_calls=[_tool(n) for n in tool_calls]), seen["messages"], "routing-test")

    monkeypatch.setattr(AIAgent, "run_conversation", review_turn)
    target, prompt = bg.spawn_background_review_thread(
        agent, [], review_memory=review_memory, review_skills=review_skills, task_cfg={})
    target()
    return prompt, seen["fork"], seen["messages"]


@pytest.mark.parametrize("review_memory, review_skills", SCOPES)
def test_every_review_kind_gets_the_provider_tools(make_agent, monkeypatch, review_memory, review_skills):
    provider = Provider()
    agent = make_agent(provider)
    prompt, fork, messages = _run_review(
        agent, monkeypatch, review_memory=review_memory, review_skills=review_skills,
        tool_calls=["review_write"])
    assert PROVIDER_PROMPT in prompt
    assert json.loads(messages[0]["content"])["success"] is True
    assert [c[1] for c in provider.calls] == ["review_write"]


@pytest.mark.parametrize("review_memory, review_skills", [(False, True), (True, True)])
def test_skill_reviews_with_a_provider_use_the_procedural_prompt(make_agent, review_memory, review_skills):
    agent = make_agent(Provider())
    _target, prompt = bg.spawn_background_review_thread(
        agent, [], review_memory=review_memory, review_skills=review_skills, task_cfg={})
    assert bg._PROCEDURAL_SKILL_REVIEW_PROMPT in prompt
    assert PROVIDER_PROMPT in prompt
    assert not any(phrase in prompt for phrase in OLD_ROUTING)
    assert "USER.md" not in prompt and "MEMORY.md" not in prompt


@pytest.mark.parametrize("review_memory, review_skills", [(False, True), (True, True)])
def test_skill_review_stays_procedural_when_the_provider_cannot_be_captured(
        make_agent, monkeypatch, review_memory, review_skills):
    provider = Provider()
    provider.opt_in = False
    agent = make_agent(provider)
    prompt, fork, _messages = _run_review(
        agent, monkeypatch, review_memory=review_memory, review_skills=review_skills, tool_calls=[])
    assert prompt == bg._PROCEDURAL_SKILL_REVIEW_PROMPT
    assert getattr(fork, "_memory_review_dispatch", None) is None
    assert "unsaved" in prompt and "into a skill" in prompt
    assert not provider.calls


def test_procedural_prompt_drops_preference_embedding_and_keeps_the_skill_writing_rules():
    prompt = bg._PROCEDURAL_SKILL_REVIEW_PROMPT
    assert not any(phrase in prompt for phrase in OLD_ROUTING)
    assert "skill_view" in prompt  # read-before-write survives
    assert "Nothing to save" in prompt  # a normal result, not a missed learning


def test_procedural_lesson_block_stays_in_step_with_the_upstream_block_it_mirrors():
    """The mirrored block may drop only the lines that send user preferences into SKILL.md. Every
    other line must still be the upstream line, so an upstream edit shows up here when merged."""
    def lines(block):
        return {line.strip() for line in block.splitlines() if line.strip().startswith("•")}

    mirrored, upstream = lines(bg._PROCEDURAL_LESSON_LAYER_BLOCK), lines(bg._LESSON_LAYER_BLOCK)
    changed = mirrored - upstream
    assert len(changed) == 1 and next(iter(changed)).startswith("• Always-on rules")
    dropped = upstream - mirrored
    assert len(dropped) == 1 and "standing user preferences" in next(iter(dropped))


@pytest.mark.parametrize("scope, attr", [
    ((True, False), "_MEMORY_REVIEW_PROMPT"), ((False, True), "_SKILL_REVIEW_PROMPT"),
    ((True, True), "_COMBINED_REVIEW_PROMPT")])
def test_without_a_provider_reviews_keep_the_upstream_prompts_and_no_provider_tools(
        make_agent, monkeypatch, scope, attr):
    agent = make_agent(None)
    prompt, fork, _messages = _run_review(
        agent, monkeypatch, review_memory=scope[0], review_skills=scope[1], tool_calls=[])
    assert prompt == getattr(agent, attr)
    assert getattr(fork, "_memory_review_dispatch", None) is None


def test_native_store_beside_a_provider_keeps_the_native_memory_prompt_and_swaps_only_the_skill_half(make_agent):
    agent = make_agent(Provider())
    agent._memory_store = object()
    agent._memory_enabled = True
    _target, prompt = bg.spawn_background_review_thread(
        agent, [], review_memory=True, review_skills=True, task_cfg={})
    assert agent._MEMORY_REVIEW_PROMPT in prompt
    assert bg._PROCEDURAL_SKILL_REVIEW_PROMPT in prompt
    assert PROVIDER_PROMPT in prompt
    assert not any(phrase in prompt for phrase in OLD_ROUTING)


@pytest.mark.parametrize("review_memory, review_skills", SCOPES)
def test_spawn_captures_the_binding_for_every_review_kind(make_agent, monkeypatch, review_memory, review_skills):
    agent = make_agent(Provider())
    spawn = MagicMock()
    monkeypatch.setattr(agent, "_spawn_background_review_now", spawn)
    agent._spawn_background_review([{"role": "user", "content": "I prefer written updates."}],
                                   review_memory=review_memory, review_skills=review_skills)
    binding = spawn.call_args.kwargs["memory_review_binding"]
    assert isinstance(binding, MemoryReviewBinding) and binding.tool_names == ("review_write",)


def test_spawn_captures_no_binding_without_a_provider(make_agent, monkeypatch):
    agent = make_agent(None)
    spawn = MagicMock()
    monkeypatch.setattr(agent, "_spawn_background_review_now", spawn)
    agent._spawn_background_review([{"role": "user", "content": "hello"}], review_skills=True)
    assert spawn.call_args.kwargs.get("memory_review_binding") is None


def test_deferred_skill_review_keeps_the_binding_captured_at_enqueue(make_agent, monkeypatch):
    import run_agent
    from agent.review_idle_queue import QUEUE
    provider = Provider()
    agent = make_agent(provider)
    queued = []
    monkeypatch.setattr(bg, "load_background_review_settings", lambda: (True, {}))
    monkeypatch.setattr(run_agent, "_review_should_defer", lambda *args: True)
    monkeypatch.setattr(run_agent, "_review_queue_key", lambda *args: "key")
    monkeypatch.setattr(QUEUE, "enqueue", lambda owner, key, kwargs: queued.append(kwargs))
    original_id = agent.session_id
    agent._spawn_background_review([{"role": "user", "content": "Source."}], review_skills=True)
    agent.session_id = "later-session"
    binding = queued[0]["memory_review_binding"]
    binding.handle_tool_call("review_write", {"text": "evidence"})
    assert provider.calls == [(original_id, "review_write", {"text": "evidence"})]


def test_a_stated_preference_reaches_the_provider_through_a_real_skill_review_loop(make_agent):
    """Scripted model, real fork: the skill review is offered the memory tool and the prompt that
    sends the preference there; the call lands on the captured provider, not in a skill."""
    provider = Provider()
    agent = make_agent(provider)
    transcript = [{"role": "user", "content": "For design reviews use one table, please."},
                  {"role": "assistant", "content": "Done."}]
    responses = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="", tool_calls=[_tool("review_write", {"text": "Prefers one table in design reviews"})]),
            finish_reason="tool_calls")], usage=None, model="test/model"),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="Saved the preference to memory.", tool_calls=None), finish_reason="stop")],
            usage=None, model="test/model"),
    ]
    agent.client.chat.completions.create.side_effect = responses
    target, _prompt = bg.spawn_background_review_thread(
        agent, transcript, review_memory=False, review_skills=True, task_cfg={})
    target()
    assert provider.calls == [(agent.session_id, "review_write", {"text": "Prefers one table in design reviews"})]
    sent = agent.client.chat.completions.create.call_args_list[0].kwargs["messages"]
    review_prompt = sent[-1]["content"]
    assert bg._PROCEDURAL_SKILL_REVIEW_PROMPT in review_prompt
    assert not any(phrase in review_prompt for phrase in OLD_ROUTING)
