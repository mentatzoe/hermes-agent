"""Sequential tool execution must abandon the wait when the user interrupts.

Regression tests for the "interrupt doesn't end a running tool" class:
the sequential executor path previously ran the tool inline (when the
deadline was disabled) or waited in 5s slices without checking
``agent._interrupt_requested`` — a non-cooperative tool (e.g. a blocking
FAL ``handler.get()``) held the whole turn hostage until it returned.

Now the wait loop polls the interrupt flag every
``_SEQUENTIAL_INTERRUPT_POLL_SECONDS`` and, after a 3s cooperative grace,
synthesizes a cancelled tool result and abandons the worker.
"""

import threading
import time

import pytest

import agent.tool_executor as tool_executor
from agent.tool_executor import (
    _ManagedToolResult,
    _ToolCancelledResult,
    _run_sequential_tool_execution_middleware,
)

class _FakeAgent:
    def __init__(self):
        self._tool_worker_threads = set()
        self._tool_worker_threads_lock = threading.Lock()
        self._interrupt_requested = False
        self.activity = []

    def _touch_activity(self, msg):
        self.activity.append(msg)

@pytest.fixture()
def fake_agent():
    return _FakeAgent()

@pytest.fixture(autouse=True)
def _fast_polls(monkeypatch):
    # Keep the test fast: short poll slice, no config lookups.
    monkeypatch.setattr(tool_executor, "_SEQUENTIAL_INTERRUPT_POLL_SECONDS", 0.05)
    emitted = []
    monkeypatch.setattr(
        tool_executor,
        "_emit_terminal_post_tool_call",
        lambda agent, **kw: emitted.append(kw),
    )
    yield emitted

def test_interrupt_abandons_noncooperative_tool(monkeypatch, fake_agent, _fast_polls):
    """A blocking tool is abandoned within ~poll+grace once interrupted."""

    started = threading.Event()

    def _fake_middleware(agent_arg, **kwargs):
        started.set()
        time.sleep(30)  # non-cooperative: never checks is_interrupted()
        return _ManagedToolResult(
            result="late result", args={}, middleware_trace=[],
            blocked=False, dispatched=True,
        )

    monkeypatch.setattr(
        tool_executor, "_run_agent_tool_execution_middleware", _fake_middleware
    )
    monkeypatch.setattr(
        tool_executor, "_resolve_sequential_tool_timeout", lambda: None
    )

    def _interrupt_soon():
        started.wait(5)
        time.sleep(0.1)
        fake_agent._interrupt_requested = True

    threading.Thread(target=_interrupt_soon, daemon=True).start()

    t0 = time.monotonic()
    managed = _run_sequential_tool_execution_middleware(
        fake_agent,
        function_name="image_generate",
        function_args={"prompt": "x"},
        effective_task_id="t",
        tool_call_id="call_1",
        execute=lambda a: "unused",
    )
    elapsed = time.monotonic() - t0

    assert isinstance(managed.result, _ToolCancelledResult)
    assert "cancelled" in str(managed.result)
    # poll (0.05s) + interrupt delay (0.1s) + grace (3s) + slack — nowhere
    # near the 30s tool runtime.
    assert elapsed < 10.0
    # The executor emitted the terminal post_tool_call itself.
    assert any(kw.get("status") == "cancelled" for kw in _fast_polls)

def test_interrupt_prefers_real_result_from_cooperative_tool(
    monkeypatch, fake_agent, _fast_polls
):
    """A tool that finishes within the grace window returns its real result."""

    def _fake_middleware(agent_arg, **kwargs):
        # Cooperative-ish: returns quickly once running (well inside grace).
        time.sleep(0.3)
        return _ManagedToolResult(
            result="real result", args={}, middleware_trace=[],
            blocked=False, dispatched=True,
        )

    monkeypatch.setattr(
        tool_executor, "_run_agent_tool_execution_middleware", _fake_middleware
    )
    monkeypatch.setattr(
        tool_executor, "_resolve_sequential_tool_timeout", lambda: None
    )
    fake_agent._interrupt_requested = True  # interrupted before first poll

    managed = _run_sequential_tool_execution_middleware(
        fake_agent,
        function_name="web_search",
        function_args={},
        effective_task_id="t",
        tool_call_id="call_2",
        execute=lambda a: "unused",
    )

    assert managed.result == "real result"
    assert not isinstance(managed.result, _ToolCancelledResult)


@pytest.mark.parametrize("tool_name", ["write_file", "patch"])
def test_interrupt_abandonment_fences_file_dispatch_after_finalization(
    tmp_path, monkeypatch, _fast_polls, tool_name
):
    """Preflight returning during abandonment logging must not admit a late edit."""
    import json
    from contextlib import contextmanager
    from types import SimpleNamespace

    import model_tools
    from tests.agent.test_sequential_tool_timeout import _make_agent
    from tools.file_tools import read_file_tool

    agent = _make_agent(tmp_path)
    agent.valid_tool_names = {"read_file", "write_file", "patch"}
    agent.enabled_toolsets = ["file"]
    agent._checkpoint_mgr.enabled = True
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    target = tmp_path / "target.txt"
    target.write_text("before\n")
    task_id, call_id = "interrupt-preflight", "interrupted-file-call"
    assert "error" not in json.loads(read_file_tool(str(target), task_id=task_id))
    args = {"path": str(target)}
    if tool_name == "write_file":
        args["content"] = "after\n"
    else:
        args.update(old_string="before", new_string="after")
    call = SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(
        name=tool_name, arguments=json.dumps(args)))

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    boundary, allow_mutation = threading.Event(), threading.Event()
    real_worker = tool_executor._registered_tool_worker
    real_handler = model_tools.handle_function_call
    admitted, outcomes = [], []

    @contextmanager
    def tracked_worker(owner):
        try:
            with real_worker(owner) as tid:
                yield tid
        finally:
            finished.set()
            boundary.set()

    def checkpoint(*_args):
        entered.set()
        assert release.wait(10), "checkpoint preflight was not released"

    def interrupt_after_preflight_started(*_args):
        assert entered.wait(10), "worker never reached checkpoint preflight"
        assert agent.interrupt(tool_reason="user interrupt")
        return "interrupted", None

    def held_handler(*args, **kwargs):
        admitted.append(kwargs["tool_call_id"])
        boundary.set()
        assert allow_mutation.wait(10), "handler was not released after finalization"
        result = real_handler(*args, **kwargs)
        outcomes.append(result)
        return result

    info = tool_executor.logger.info

    def release_in_abandonment_log(message, *args, **kwargs):
        if "abandoned due to" in message:
            # The actual three-second grace and completion check have ended.
            # Let preflight finish before logging returns, then wait for either
            # a rejected dispatch or the held real handler to reach the boundary.
            release.set()
            assert boundary.wait(10), "worker did not reach the dispatch boundary"
        info(message, *args, **kwargs)

    monkeypatch.setattr(tool_executor, "_ensure_file_checkpoint", checkpoint)
    monkeypatch.setattr(tool_executor, "_registered_tool_worker", tracked_worker)
    monkeypatch.setattr(tool_executor, "_poll_sequential_future", interrupt_after_preflight_started)
    monkeypatch.setattr(tool_executor.logger, "info", release_in_abandonment_log)
    monkeypatch.setattr(model_tools, "handle_function_call", held_handler)
    messages = []
    try:
        tool_executor.execute_tool_calls_sequential(
            agent, SimpleNamespace(tool_calls=[call]), messages, task_id)
        assert len(messages) == 1 and "cancelled" in messages[0]["content"]
        assert target.read_text() == "before\n"
        # Turn finalization clears the worker's cooperative interrupt as well.
        # A revoked call must stay revoked even after that shared signal clears.
        assert agent.clear_interrupt()
    finally:
        release.set()
        allow_mutation.set()
        assert finished.wait(10), "abandoned worker did not finish"
    assert target.read_text() == "before\n", ("edit landed after cancellation", admitted, outcomes)
    assert admitted == [], ("body admitted during abandonment logging", outcomes)
    assert [(e["tool_call_id"], e.get("error_type")) for e in _fast_polls] == [
        (call_id, "tool_interrupted")]
