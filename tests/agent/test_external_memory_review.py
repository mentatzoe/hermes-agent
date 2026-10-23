"""External review capabilities must not attach provider lifecycle to detached forks."""

import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import background_review as bg
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from agent.turn_context import _tick_memory_nudge
from run_agent import AIAgent


def _schema(name):
    return {"name": name, "description": name,
            "parameters": {"type": "object", "properties": {}}}


class ReviewProvider(MemoryProvider):
    name = "external-test"

    def __init__(self):
        self.lifecycle = []
        self.calls = []
        self.peer = "original-peer"
        self.opt_in = True

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.lifecycle.append(("initialize", session_id))

    def get_tool_schemas(self):
        return [_schema(n) for n in ("review_write", "provider_other", "unexposed")]

    def handle_tool_call(self, tool_name, args, **kwargs):
        raise AssertionError("review must use the captured capability, not mutable provider state")

    def memory_review_binding(self, *, session_id=""):
        from agent.memory_provider import MemoryReviewBinding
        if not self.opt_in:
            return None
        peer = self.peer

        def dispatch(name, args):
            self.calls.append((session_id, peer, name, args))
            return json.dumps({"success": True})

        return MemoryReviewBinding("Maintain attributed external memory using review_write.",
                                   ("review_write", "unexposed"), dispatch)

    def prefetch(self, query, *, session_id=""):
        self.lifecycle.append(("prefetch", query))
        return ""

    def sync_turn(self, user_content, assistant_content, **kwargs):
        self.lifecycle.append(("sync", user_content))

    def shutdown(self):
        self.lifecycle.append(("shutdown",))


@pytest.fixture
def parent(monkeypatch):
    config = {"memory": {"memory_enabled": False, "user_profile_enabled": False,
                         "nudge_interval": 2}, "skills": {"creation_nudge_interval": 0}}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", MagicMock())
    base = ["skill_view", "skills_list", "skill_manage"]
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kwargs: [
        {"type": "function", "function": _schema(n)} for n in base])
    provider = ReviewProvider()
    manager = MemoryManager()
    manager.add_provider(provider)
    agent = AIAgent(model="test/model", provider="openai", api_key="test-key",
                    base_url="https://example.invalid/v1", quiet_mode=True,
                    skip_context_files=True, memory_manager=manager,
                    enabled_toolsets=["memory", "skills"], max_iterations=3)
    agent._cached_system_prompt = "Stable test prefix"
    agent._emit_auxiliary_failure = lambda label, error: pytest.fail(f"{label}: {error}")
    agent.valid_tool_names.discard("unexposed")
    agent.tools = [t for t in agent.tools if t["function"]["name"] != "unexposed"]
    yield agent, provider
    manager.shutdown_all()
    agent.release_clients()


def test_native_off_nudge_uses_exposed_opt_in_and_interval(parent):
    agent, provider = parent
    assert agent._memory_store is None
    assert agent._memory_nudge_interval == 2
    assert not _tick_memory_nudge(agent)
    assert _tick_memory_nudge(agent)
    provider.opt_in = False
    assert not _tick_memory_nudge(agent)
    provider.opt_in = True
    agent.valid_tool_names.clear()
    assert not _tick_memory_nudge(agent)
    agent.valid_tool_names.add("review_write")
    agent._memory_nudge_interval = 0
    assert not _tick_memory_nudge(agent)


def test_actual_completed_turn_spawns_memory_review_native_off(parent, monkeypatch):
    agent, _provider = parent
    agent._turns_since_memory = 1
    response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="A complete answer.", tool_calls=None),
        finish_reason="stop")], model="test/model", usage=None)
    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = response
    spawn = MagicMock()
    monkeypatch.setattr(agent, "_spawn_background_review", spawn)
    result = agent.run_conversation("Remember that I prefer written updates.")
    assert result["final_response"] == "A complete answer."
    spawn.assert_called_once()
    assert spawn.call_args.kwargs["review_memory"] is True


def test_binding_is_frozen_captured_and_lifecycle_free(parent):
    agent, provider = parent
    binding = agent._memory_manager.memory_review_binding(session_id=agent.session_id)
    with pytest.raises(FrozenInstanceError):
        binding.prompt = "changed"
    provider.peer = "new-peer"
    binding.handle_tool_call("review_write", {"text": "evidence"})
    assert provider.calls[0][:2] == (agent.session_id, "original-peer")
    assert provider.lifecycle == []
    assert "error" in json.loads(binding.handle_tool_call("provider_other", {}))
    assert len(provider.calls) == 1


@pytest.mark.parametrize("combined", [False, True])
def test_native_off_prompt_routes_to_provider_without_native_store(parent, combined):
    agent, _provider = parent
    _target, prompt = bg.spawn_background_review_thread(
        agent, [], review_memory=True, review_skills=combined, task_cfg={})
    assert "Maintain attributed external memory" in prompt
    assert "USER.md" not in prompt and "MEMORY.md" not in prompt
    if combined:
        assert "skill" in prompt.lower()


def _tool(name, args=None):
    return SimpleNamespace(id="call-" + name, type="function", function=SimpleNamespace(
        name=name, arguments=json.dumps(args or {})))


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("concurrent", [False, True])
def test_detached_fork_executes_captured_tools_through_executor(parent, monkeypatch, routed, concurrent):
    agent, provider = parent
    from hermes_cli.observability import shared_metrics_loop
    records = []
    monkeypatch.setattr(shared_metrics_loop, "record_provider_memory_call",
                        lambda *args, **kwargs: records.append((args, kwargs)))
    # A deterministic scrub seam proves the callback still goes through manager egress.
    monkeypatch.setattr("agent.memory_manager._scrub", lambda value: value.replace("private-token", "[redacted]"))
    if routed:
        rt = bg._resolve_review_runtime(agent, {})
        monkeypatch.setattr(bg, "_resolve_review_runtime", lambda *args: {**rt, "routed": True})
    forks = []

    def review_turn(fork, **kwargs):
        forks.append(fork)
        assert fork._memory_manager is None
        assert fork._session_db is None and fork._persist_disabled
        assert fork._memory_store is None
        assert fork._memory_nudge_interval == 0
        assert not hasattr(fork._memory_review_dispatch, "shutdown_all")
        messages = []
        calls = [_tool("review_write", {"text": "private-token"}),
                 _tool("provider_other"), _tool("unexposed")]
        execute = fork._execute_tool_calls_concurrent if concurrent else fork._execute_tool_calls_sequential
        execute(SimpleNamespace(tool_calls=calls), messages, "review-test")
        assert json.loads(messages[0]["content"])["success"] is True
        assert "error" in json.loads(messages[1]["content"])
        assert "error" in json.loads(messages[2]["content"])
        assert "Background review denied" in messages[1]["content"]
        assert "Background review denied" in messages[2]["content"]
        assert "error" in json.loads(fork._memory_review_dispatch.handle_tool_call("unexposed", {}))
        assert "error" in json.loads(fork._memory_review_dispatch.handle_tool_call("provider_other", {}))
        # Mutating the provider after spawn cannot retarget the captured peer capability.
        assert "Maintain attributed external memory" in kwargs["user_message"]

    monkeypatch.setattr(AIAgent, "run_conversation", review_turn)
    target, _prompt = bg.spawn_background_review_thread(
        agent, [], review_memory=True, task_cfg={"extra_tools": ["provider_other", "unexposed"]})
    provider.peer = "changed-after-spawn"
    target()
    assert len(forks) == 1
    assert provider.calls == [(agent.session_id, "original-peer", "review_write", {"text": "[redacted]"})]
    assert provider.lifecycle == []
    assert len(records) == 1
    assert records[0][0][:3] == (provider.name, "review_write", {"text": "[redacted]"})


def test_skills_only_and_btw_forks_have_no_review_dispatch(parent, monkeypatch):
    agent, provider = parent
    forks = []

    def review_turn(fork, **kwargs):
        forks.append(fork)
        assert getattr(fork, "_memory_review_dispatch", None) is None
        messages = []
        fork._execute_tool_calls_sequential(SimpleNamespace(tool_calls=[_tool("review_write")]),
                                            messages, "skills-test")
        assert "error" in json.loads(messages[0]["content"])

    monkeypatch.setattr(AIAgent, "run_conversation", review_turn)
    target, prompt = bg.spawn_background_review_thread(
        agent, [], review_skills=True, task_cfg={"extra_tools": ["review_write"]})
    target()
    assert len(forks) == 1
    assert "Maintain attributed external memory" not in prompt
    fork, _rt, _routed = bg.build_cache_parity_fork(agent, {}, max_iterations=1, write_origin="btw")
    assert getattr(fork, "_memory_review_dispatch", None) is None
    fork.release_clients()
    assert not provider.calls and not provider.lifecycle


def test_cancelled_review_cannot_dispatch(parent, monkeypatch):
    agent, provider = parent
    run = bg.prepare_background_review_run(agent)
    run.cancel()
    called = MagicMock()
    monkeypatch.setattr(AIAgent, "run_conversation", called)
    target, _prompt = bg.spawn_background_review_thread(
        agent, [], review_memory=True, task_cfg={}, review_run=run)
    target()
    called.assert_not_called()
    assert run.request_done.is_set()
    assert not provider.calls and not provider.lifecycle


def test_real_review_loop_never_ingests_harness_or_initializes_provider(parent, monkeypatch):
    agent, provider = parent
    load = MagicMock(side_effect=AssertionError("fork must not instantiate a memory provider"))
    monkeypatch.setattr("plugins.memory.load_memory_provider", load)
    responses = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="", tool_calls=[_tool("review_write", {"text": "attributed evidence"})]),
            finish_reason="tool_calls")], usage=None, model="test/model"),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="Review complete.", tool_calls=None), finish_reason="stop")],
            usage=None, model="test/model"),
    ]
    agent.client.chat.completions.create.side_effect = responses
    target, _prompt = bg.spawn_background_review_thread(
        agent, [{"role": "user", "content": "Source evidence."},
                {"role": "assistant", "content": "Acknowledged."}], review_memory=True, task_cfg={})
    target()
    assert provider.calls == [(agent.session_id, "original-peer", "review_write",
                               {"text": "attributed evidence"})]
    assert provider.lifecycle == []
    load.assert_not_called()


@pytest.mark.parametrize("concurrent", [False, True])
def test_review_callback_stays_behind_approval_and_write_guards(parent, monkeypatch, concurrent):
    agent, provider = parent
    approvals = []

    def hooks(name, **kwargs):
        if name == "pre_tool_call":
            return [{"action": "approve", "message": "Review write needs approval"}]
        return []

    def deny(name, description, **kwargs):
        approvals.append(name)
        return {"approved": False, "message": "approval denied"}

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", hooks)
    monkeypatch.setattr("tools.approval.request_tool_approval", deny)

    def review_turn(fork, **kwargs):
        messages = []
        execute = fork._execute_tool_calls_concurrent if concurrent else fork._execute_tool_calls_sequential
        execute(SimpleNamespace(tool_calls=[_tool("review_write")]), messages, "guard-test")
        assert "approval denied" in messages[0]["content"]

    monkeypatch.setattr(AIAgent, "run_conversation", review_turn)
    target, _prompt = bg.spawn_background_review_thread(agent, [], review_memory=True, task_cfg={})
    target()
    assert approvals == ["review_write"]
    assert not provider.calls and not provider.lifecycle


def test_native_review_prompt_and_nudge_unchanged_without_provider_opt_in(parent):
    agent, provider = parent
    provider.opt_in = False
    agent._memory_store = object()
    agent._memory_enabled = True
    agent.valid_tool_names.add("memory")
    assert not _tick_memory_nudge(agent)
    assert _tick_memory_nudge(agent)
    _target, prompt = bg.spawn_background_review_thread(agent, [], review_memory=True, task_cfg={})
    assert prompt == agent._MEMORY_REVIEW_PROMPT


def test_manager_binding_is_opt_in_external_owned_and_revoked_on_shutdown():
    from agent.memory_provider import MemoryReviewBinding
    provider = ReviewProvider()
    assert MemoryProvider.memory_review_binding(provider) is None
    provider.name = "builtin"
    manager = MemoryManager()
    manager.add_provider(provider)
    assert manager.memory_review_binding() is None
    provider.name = "external-test"
    # The advertised capability cannot claim unrelated or reserved core tools.
    provider.memory_review_binding = lambda **kwargs: MemoryReviewBinding(
        "review", ("review_write", "memory", "terminal", "unregistered"),
        lambda name, args: provider.calls.append(name) or '{"success":true}')
    binding = manager.memory_review_binding()
    assert binding.tool_names == ("review_write",)
    assert "error" in json.loads(binding.handle_tool_call("terminal", {}))
    manager.shutdown_all()
    assert "error" in json.loads(binding.handle_tool_call("review_write", {}))
    assert not provider.calls


def test_deferred_review_captures_at_enqueue_not_later_dispatch(parent, monkeypatch):
    import run_agent
    from agent.review_idle_queue import QUEUE
    agent, provider = parent
    queued = []
    monkeypatch.setattr(bg, 'load_background_review_settings', lambda: (True, {}))
    monkeypatch.setattr(run_agent, '_review_should_defer', lambda *args: True)
    monkeypatch.setattr(run_agent, '_review_queue_key', lambda *args: 'original-key')
    monkeypatch.setattr(QUEUE, 'enqueue', lambda owner, key, kwargs: queued.append(kwargs))
    original_id = agent.session_id
    agent._spawn_background_review([{'role':'user','content':'Original evidence.'}], review_memory=True)
    assert len(queued) == 1
    provider.peer = 'later-peer'
    agent.session_id = 'later-session'
    calls = []
    monkeypatch.setattr(bg, '_run_review_in_thread', lambda *args, **kw: calls.append(kw))
    class InlineThread:
        def __init__(self, target, **kwargs): self.target = target
        def start(self): self.target()
    monkeypatch.setattr(run_agent.threading, 'Thread', InlineThread)
    agent._spawn_background_review_now(**queued[0])
    binding = calls[0]['memory_review_binding']
    binding.handle_tool_call('review_write', {'text':'evidence'})
    assert provider.calls == [(original_id, 'original-peer', 'review_write', {'text':'evidence'})]


def test_cancelled_or_finished_fork_revokes_borrowed_dispatch(parent, monkeypatch):
    agent, provider = parent
    run = bg.prepare_background_review_run(agent)
    saved = []
    def review_turn(fork, **kwargs):
        saved.append(fork._memory_review_dispatch)
        run.cancel()
        assert 'error' in json.loads(saved[0].handle_tool_call('review_write', {}))
    monkeypatch.setattr(AIAgent, 'run_conversation', review_turn)
    target, _ = bg.spawn_background_review_thread(agent, [], review_memory=True, task_cfg={}, review_run=run)
    target()
    assert 'error' in json.loads(saved[0].handle_tool_call('review_write', {}))
    assert not provider.calls


def test_provider_without_opt_in_or_with_failed_capture_is_not_eligible(parent):
    agent, provider = parent
    provider.memory_review_binding = None  # legacy duck-typed provider
    assert agent._memory_manager.memory_review_binding() is None

    def fail(**kwargs):
        raise RuntimeError("capture unavailable")

    provider.memory_review_binding = fail
    assert not _tick_memory_nudge(agent)
    assert agent._turns_since_memory == 0
