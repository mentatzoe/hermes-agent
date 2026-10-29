"""Background reviews with an external memory provider keep procedures in skills and personal
material in memory.

With an external provider (Honcho) registered, every review kind gets the provider's tools and the
procedural skill-review prompt; a skill review never turns a personal preference into a skill, even
when the provider's save tools are unavailable. Without an external provider nothing changes:
native-only reviews keep upstream's prompts.
"""

import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import background_review as bg
from agent.memory_manager import MemoryManager, external_memory_active, memory_review_binding_for_agent
from agent.memory_provider import MemoryProvider, MemoryReviewBinding
from agent.prompt_builder import SKILLS_GUIDANCE, SKILLS_GUIDANCE_EXTERNAL_MEMORY, clear_skills_system_prompt_cache
from agent.system_prompt import build_system_prompt
from run_agent import AIAgent

PROVIDER_PROMPT = "Maintain attributed external memory using review_write."
SCOPES = [(True, False), (False, True), (True, True)]
# What the main prompt says about skills when no external provider owns personal memory.
LEGACY_SKILL_LINES = ("the user's preferred approach", "offer to save as a skill")
TRANSCRIPT = [{"role": "user", "content": "For this report, use one table, please."},
              {"role": "assistant", "content": "Done."}]
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
        self.lifecycle = []
        self.opt_in = True

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.lifecycle.append(("initialize", session_id))

    def prefetch(self, query, *, session_id=""):
        self.lifecycle.append(("prefetch", query))
        return ""

    def sync_turn(self, user_content, assistant_content, **kwargs):
        self.lifecycle.append(("sync", user_content))

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

    def build(provider=None, **agent_kwargs):
        manager = MemoryManager()
        if provider is not None:
            manager.add_provider(provider)
        kwargs = dict(model="test/model", provider="openai", api_key="test-key",
                      base_url="https://example.invalid/v1", quiet_mode=True, skip_context_files=True,
                      memory_manager=manager, enabled_toolsets=["memory", "skills"], max_iterations=3)
        agent = AIAgent(**{**kwargs, **agent_kwargs})
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


def test_procedural_prompt_does_not_stop_a_preference_only_review_from_saving_to_memory():
    """'Nothing to save' must not read as 'skip the memory save': a review with no procedure makes no
    skill change, and ends only when there is nothing for the skills or for the provider either."""
    _head, marker, tail = bg._PROCEDURAL_SKILL_REVIEW_PROMPT.partition("'Nothing to save.' is an ordinary result")
    assert marker, "the procedural prompt lost its closing paragraph"
    assert "make no skill change" in tail
    assert "only when there is nothing for either" in tail
    assert "never a reason to touch a skill" in tail


# The one sentence of upstream's lesson block that sends standing user preferences into SKILL.md.
PREFERENCE_EDIT = ("(standing user preferences, gates", "(gates")
# The block's opening paragraph is rewritten on purpose (a skill is the procedure, not the user's
# preferences); only its first clause is shared, so pin that to notice an upstream rewrite.
SHARED_OPENING = "What a skill IS: the instructions for doing a class of task the most efficient and correct way"


def _bullets(block):
    return block[block.index("  • "):]


def _assert_procedural_block_follows(upstream):
    """The procedural block keeps every upstream bullet word for word except the one preference edit,
    and has its own opening paragraph, which says personal preferences live in memory."""
    old, new = PREFERENCE_EDIT
    procedural = bg._PROCEDURAL_LESSON_LAYER_BLOCK
    assert old in upstream, "upstream reworded the line the procedural block edits; decide that edit again"
    assert upstream.startswith(SHARED_OPENING) and procedural.startswith(SHARED_OPENING)
    assert _bullets(procedural) == _bullets(upstream).replace(old, new)
    assert "personal preferences are not part of it" in procedural[:procedural.index("  • ")]
    assert not any(phrase in procedural for phrase in OLD_ROUTING)


def test_procedural_lesson_block_is_upstreams_bullets_with_only_the_preference_edit():
    _assert_procedural_block_follows(bg._LESSON_LAYER_BLOCK)


@pytest.mark.parametrize("old, new", [
    ("references/ is for depth that is only needed sometimes:", "references/ is forbidden:"),  # in the edited bullet
    ("Procedure first:", "Commands first:"),  # in an untouched bullet
    ("What a skill IS: the instructions", "What a skill IS: the guidance"),  # in the rewritten opening
])
def test_a_change_to_upstreams_lesson_block_is_caught_wherever_it_lands(monkeypatch, old, new):
    assert old in bg._LESSON_LAYER_BLOCK
    monkeypatch.setattr(bg, "_LESSON_LAYER_BLOCK", bg._LESSON_LAYER_BLOCK.replace(old, new))
    with pytest.raises(AssertionError):
        _assert_procedural_block_follows(bg._LESSON_LAYER_BLOCK)


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


@pytest.fixture
def skills_home(tmp_path, monkeypatch):
    """A HERMES_HOME holding one skill, so the system prompt renders its skills index."""
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    home = tmp_path / "skills-home"
    skill = home / "skills" / "demo" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: demo\ndescription: Fixture method.\n---\n# Fixture method\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    clear_skills_system_prompt_cache(clear_snapshot=True)
    yield home
    clear_skills_system_prompt_cache(clear_snapshot=True)


def _parent_with_real_prompt(make_agent, provider, **agent_kwargs):
    """A parent whose cached system prompt is what a live conversation would have built."""
    agent = make_agent(provider, **agent_kwargs)
    agent._cached_system_prompt = build_system_prompt(agent)
    return agent


def _first_request_system_text(agent, monkeypatch, *, routed, scope, **review_kwargs):
    """Run one review through the real fork with a scripted model; return the system text of its
    first outbound request, which is what the reviewing model is actually told."""
    runtime = bg._resolve_review_runtime(agent, {})
    monkeypatch.setattr(bg, "_resolve_review_runtime", lambda *a, **k: {
        **runtime, "model": "test/reviewer" if routed else agent.model, "routed": routed})
    sent = []

    def answer(**kwargs):
        sent.append(copy.deepcopy(kwargs["messages"]))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="Nothing to save.", tool_calls=None), finish_reason="stop")], usage=None, model="test/reviewer")

    agent.client.chat.completions.create.side_effect = answer
    target, _prompt = bg.spawn_background_review_thread(
        agent, TRANSCRIPT, review_memory=scope[0], review_skills=scope[1], task_cfg={}, **review_kwargs)
    target()
    assert sent, "the review never reached the model"
    return "\n".join(m["content"] for m in sent[0] if m["role"] in {"system", "developer"})


def _assert_external_skills_routing(text):
    assert SKILLS_GUIDANCE_EXTERNAL_MEMORY in text
    assert not any(line in text for line in LEGACY_SKILL_LINES)


def test_the_parent_prompt_fixture_really_carries_the_external_routing(make_agent, skills_home):
    """Guards the tests below: if the parent itself did not render the external variant, they would
    prove nothing."""
    agent = _parent_with_real_prompt(make_agent, Provider())
    _assert_external_skills_routing(agent._cached_system_prompt)


@pytest.mark.parametrize("routed", [True, False], ids=["routed-model", "same-model"])
@pytest.mark.parametrize("opt_in", [True, False], ids=["binding-offered", "no-binding"])
@pytest.mark.parametrize("review_memory, review_skills", SCOPES)
def test_every_review_request_keeps_the_external_routing_in_its_system_prompt(
        make_agent, skills_home, monkeypatch, review_memory, review_skills, opt_in, routed):
    """The procedural review prompt arrives as a user message; the system prompt above it must not
    contradict it. A routed review builds its own system prompt, and a fork has no memory manager."""
    provider = Provider()
    provider.opt_in = opt_in
    agent = _parent_with_real_prompt(make_agent, provider)
    system = _first_request_system_text(
        agent, monkeypatch, routed=routed, scope=(review_memory, review_skills))
    _assert_external_skills_routing(system)
    if not routed:
        assert system == agent._cached_system_prompt  # cache parity: the inherited prefix is untouched
    assert provider.lifecycle == []  # prompt routing only: the fork never ran the provider's lifecycle


@pytest.mark.parametrize("opt_in", [True, False], ids=["binding-offered", "no-binding"])
def test_a_same_model_fork_keeps_the_external_routing_when_it_compacts(make_agent, skills_home, opt_in):
    """A same-model fork starts on the parent's cached prefix, but its own first compaction rebuilds
    the system prompt from scratch at the commit boundary."""
    from agent.conversation_compression import _rebuild_system_prompt_at_boundary

    provider = Provider()
    provider.opt_in = opt_in
    agent = _parent_with_real_prompt(make_agent, provider)
    fork, _rt, routed = bg.build_cache_parity_fork(agent, {}, max_iterations=2)
    try:
        assert not routed and fork._memory_manager is None
        _assert_external_skills_routing(fork._cached_system_prompt)
        _assert_external_skills_routing(_rebuild_system_prompt_at_boundary(fork, None))
    finally:
        fork.release_clients()


def test_a_side_question_fork_carries_the_parents_prompt_mode_without_a_provider(make_agent, skills_home):
    agent = _parent_with_real_prompt(make_agent, Provider())
    fork, _rt, _routed = bg.build_cache_parity_fork(agent, {}, max_iterations=1, write_origin="side_question")
    try:
        assert external_memory_active(fork) and fork._memory_manager is None
    finally:
        fork.release_clients()


def test_a_provider_with_its_tools_gated_off_keeps_the_upstream_routing_in_forks_too(
        make_agent, skills_home, monkeypatch):
    """Same gate as the provider's own system-prompt block: with the memory toolset disabled there is
    nowhere to send personal material, so neither the parent nor its forks are told to."""
    from agent.conversation_compression import _rebuild_system_prompt_at_boundary

    agent = _parent_with_real_prompt(make_agent, Provider(), disabled_toolsets=["memory"])
    assert not external_memory_active(agent) and LEGACY_SKILL_LINES[0] in agent._cached_system_prompt
    fork, _rt, _routed = bg.build_cache_parity_fork(agent, {}, max_iterations=2)
    try:
        assert not external_memory_active(fork)
        rebuilt = _rebuild_system_prompt_at_boundary(fork, None)
        assert SKILLS_GUIDANCE in rebuilt and SKILLS_GUIDANCE_EXTERNAL_MEMORY not in rebuilt
        assert all(line in rebuilt for line in LEGACY_SKILL_LINES)
    finally:
        fork.release_clients()
    system = _first_request_system_text(agent, monkeypatch, routed=True, scope=(False, True))
    assert SKILLS_GUIDANCE in system and SKILLS_GUIDANCE_EXTERNAL_MEMORY not in system


def test_a_captured_binding_keeps_its_review_in_external_mode_after_the_parents_gate_closes(
        make_agent, skills_home, monkeypatch):
    """A deferred review keeps the capability captured when it was queued. The prompt mode goes with
    it, so the system prompt never contradicts the procedural review request it sits under."""
    agent = _parent_with_real_prompt(make_agent, Provider())
    binding = memory_review_binding_for_agent(agent)
    assert binding is not None
    agent.disabled_toolsets = ["memory"]
    assert not external_memory_active(agent)
    system = _first_request_system_text(
        agent, monkeypatch, routed=True, scope=(False, True), memory_review_binding=binding)
    _assert_external_skills_routing(system)


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
