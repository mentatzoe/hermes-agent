"""Tests for the in-flight tool activity heartbeat (#84491).

The gateway's turn-inactivity watchdog
(``gateway/run.py::_watch_gateway_turn_inactivity``) abandons a turn once
``seconds_since_activity`` exceeds the inactivity timeout (default 30 min).
Activity was only stamped when a tool *started* and when it *completed*, so
a tool call that ran silently for 30+ minutes looked idle to the watchdog
and the turn was hard-abandoned mid-execution (processes reaped). The
heartbeat in ``_run_agent_tool_execution_middleware`` stamps activity
periodically while a tool call is in flight. Integration tests coordinate
on those stamps, not elapsed wall-clock time (fork #40).
"""

import json
import threading
import time
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate_hermes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(exist_ok=True)


def _make_agent(monkeypatch):
    """Minimal AIAgent-like stub, mirroring test_start_order_gate.py."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "")
    import run_agent as _ra

    class _Stub:
        _interrupt_requested = False
        _interrupt_message = None
        log_prefix = ""
        quiet_mode = True
        verbose_logging = False
        log_prefix_chars = 200
        _checkpoint_mgr = MagicMock(enabled=False)
        tool_progress_callback = None
        tool_start_callback = None
        tool_complete_callback = None
        tool_progress_mode = "off"
        _todo_store = MagicMock()
        _session_db = None
        valid_tool_names = set()
        _turns_since_memory = 0
        _iters_since_skill = 0
        _current_tool = None
        _last_activity = 0.0
        session_id = ""
        _current_turn_id = ""
        _current_api_request_id = ""

        def __init__(self):
            self._tool_worker_threads: set = set()
            self._tool_worker_threads_lock = threading.Lock()
            self._active_children_lock = threading.Lock()

        def _touch_activity(self, desc):
            self._last_activity = time.time()

        def _vprint(self, msg, force=False):
            pass

        def _safe_print(self, msg):
            pass

        def _should_emit_quiet_tool_messages(self):
            return False

        def _should_start_quiet_spinner(self):
            return False

        def _has_stream_consumers(self):
            return False

        def _tool_result_content_for_active_model(self, name, result):
            return result

        def _record_file_mutation_result(self, *a, **kw):
            pass

        def _apply_pending_steer_to_tool_results(self, *a, **kw):
            pass

    stub = _Stub()
    stub._subdirectory_hints = MagicMock()
    stub._subdirectory_hints.check_tool_call = lambda *a, **kw: None
    stub._flush_messages_to_session_db = lambda *a, **kw: None
    stub._append_guardrail_observation = lambda name, args, result, **kw: result
    stub.interrupt = _ra.AIAgent.interrupt.__get__(stub)
    stub.clear_interrupt = _ra.AIAgent.clear_interrupt.__get__(stub)
    stub._guardrail_block_result = lambda d: json.dumps({"error": "blocked"})
    return stub


@pytest.fixture
def heartbeat_threads(monkeypatch):
    """Observe the real middleware-owned heartbeat, without replacing its loop."""
    import agent.tool_executor as te

    heartbeat = te._run_tool_activity_heartbeat
    threads = []

    def observe(agent, stop, label, **kwargs):
        threads.append((threading.current_thread(), stop))
        return heartbeat(agent, stop, label, **kwargs)

    monkeypatch.setattr(te, "_run_tool_activity_heartbeat", observe)
    monkeypatch.setattr(te, "_TOOL_ACTIVITY_HEARTBEAT_INTERVAL_S", 0.05)
    try:
        yield threads
    finally:
        # Clean up even when a shutdown regression makes an assertion fail.
        for thread, stop in threads:
            stop.set()
            thread.join(timeout=5)


def _assert_heartbeat_stopped(threads):
    assert len(threads) == 1, "middleware did not start exactly one heartbeat"
    thread, stop = threads[0]
    assert stop.is_set(), "middleware did not signal heartbeat shutdown"
    assert not thread.is_alive(), "heartbeat thread outlived tool execution"


class _InFlightTool:
    """Keep execute open until the real heartbeat has refreshed activity twice."""

    def __init__(self):
        self.in_flight = threading.Event()
        self.repeated = threading.Event()
        self.touches = []

    def touch(self, desc):
        if desc.startswith("tool running:") and self.in_flight.is_set():
            self.touches.append(desc)
            if len(self.touches) >= 2:
                self.repeated.set()

    def execute(self, next_args):
        self.in_flight.set()
        try:
            assert self.repeated.wait(5), "no repeated heartbeat while execute was in flight"
            return json.dumps({"ok": True})
        finally:
            self.in_flight.clear()


def test_heartbeat_touches_periodically_and_stops():
    """The heartbeat thread touches activity on cadence, then exits on stop."""
    import agent.tool_executor as te

    touches: list = []
    stop = threading.Event()
    repeated = threading.Event()

    class _Agent:
        def _touch_activity(self, desc):
            touches.append(desc)
            if len(touches) >= 2:
                repeated.set()

    thread = threading.Thread(
        target=te._run_tool_activity_heartbeat,
        args=(_Agent(), stop, "tool running: terminal"),
        kwargs={"interval": 0.05},
        daemon=True,
    )
    thread.start()
    try:
        assert repeated.wait(5), "heartbeat did not repeat"
    finally:
        stop.set()
        thread.join(timeout=5)

    assert not thread.is_alive(), "heartbeat thread did not exit on stop"
    assert len(touches) >= 2, f"expected periodic touches, got {len(touches)}"
    assert all(touch == "tool running: terminal" for touch in touches)


def test_slow_tool_call_refreshes_activity_during_execution(monkeypatch, heartbeat_threads):
    """A tool call running longer than one interval gets activity stamps.

    Before the fix, only the start stamp ("executing tool: X") and the
    completion stamp existed; a silent 30+ minute call left the clock
    frozen and the gateway watchdog abandoned the turn.
    """
    import agent.tool_executor as te

    agent = _make_agent(monkeypatch)
    agent._tool_guardrails = MagicMock(
        before_call=lambda name, args: MagicMock(allows_execution=True)
    )
    tool = _InFlightTool()
    agent._touch_activity = tool.touch

    result = te._run_agent_tool_execution_middleware(
        agent,
        function_name="terminal",
        function_args={"command": "true"},
        effective_task_id="task",
        tool_call_id="tc1",
        execute=tool.execute,
        display_index=1,
    )

    assert json.loads(result.result) == {"ok": True}

    assert len(tool.touches) >= 2
    assert all(desc == "tool running: terminal" for desc in tool.touches)
    _assert_heartbeat_stopped(heartbeat_threads)


def test_fast_tool_call_does_not_leave_stray_heartbeat(monkeypatch, heartbeat_threads):
    """A quick tool exits the heartbeat thread; no touches after return."""
    import agent.tool_executor as te

    agent = _make_agent(monkeypatch)
    agent._tool_guardrails = MagicMock(
        before_call=lambda name, args: MagicMock(allows_execution=True)
    )
    te._run_agent_tool_execution_middleware(
        agent,
        function_name="terminal",
        function_args={"command": "true"},
        effective_task_id="task",
        tool_call_id="tc1",
        execute=lambda args: json.dumps({"ok": True}),
        display_index=1,
    )

    _assert_heartbeat_stopped(heartbeat_threads)


def test_heartbeat_stops_when_execute_raises(monkeypatch, heartbeat_threads):
    """If the tool call raises, the heartbeat thread still stops (no leak)."""

    import agent.tool_executor as te

    agent = _make_agent(monkeypatch)
    agent._tool_guardrails = MagicMock(
        before_call=lambda name, args: MagicMock(allows_execution=True)
    )
    def _boom(next_args):
        raise RuntimeError("tool exploded")

    with pytest.raises(RuntimeError, match="tool exploded"):
        te._run_agent_tool_execution_middleware(
            agent,
            function_name="terminal",
            function_args={"command": "true"},
            effective_task_id="task",
            tool_call_id="tc1",
            execute=_boom,
            display_index=1,
        )

    _assert_heartbeat_stopped(heartbeat_threads)


def test_concurrent_tool_call_heartbeat(monkeypatch, heartbeat_threads):
    """Concurrent execution also stamps activity via the shared chokepoint."""
    import agent.tool_executor as te

    agent = _make_agent(monkeypatch)
    agent._tool_guardrails = MagicMock(
        before_call=lambda name, args: MagicMock(allows_execution=True)
    )
    tool = _InFlightTool()
    agent._touch_activity = tool.touch

    agent._execute_tool_calls_concurrent = (
        __import__("run_agent").AIAgent._execute_tool_calls_concurrent.__get__(agent)
    )

    class _FakeToolCall:
        def __init__(self, name, call_id):
            self.function = MagicMock(name=name, arguments="{}")
            self.function.name = name
            self.id = call_id

    class _FakeAssistantMsg:
        def __init__(self, tool_calls):
            self.tool_calls = tool_calls

    def _invoke(name, *a, **kw):
        return tool.execute({})

    agent._invoke_tool = MagicMock(side_effect=_invoke)

    msg = _FakeAssistantMsg([_FakeToolCall("tool_a", "tc_a")])
    messages: list = []
    agent._execute_tool_calls_concurrent(msg, messages, "task")

    assert len(tool.touches) >= 2
    assert all(desc == "tool running: tool_a" for desc in tool.touches)
    assert json.loads(messages[-1]["content"]) == {"ok": True}
    _assert_heartbeat_stopped(heartbeat_threads)


def test_heartbeat_exits_once_worker_tid_is_interrupted():
    """The heartbeat must not outlive the worker it speaks for (#111922).

    The executor abandons a wedged worker by raising its interrupt bit
    (``_interrupt_worker_tids``); that worker never reaches ``stop_event.set()``,
    so the heartbeat has to stop on the bit itself, or the abandoned tool keeps
    the inactivity watchdog pinned for the rest of the run.
    """
    import agent.tool_executor as te
    from tools.interrupt import set_interrupt

    touches: list = []
    touched = threading.Event()
    stop = threading.Event()
    fake_worker_tid = 10**9 + 111922  # not a live thread; only the bit matters

    class _Agent:
        def _touch_activity(self, desc):
            touches.append(desc)
            touched.set()

    thread = threading.Thread(
        target=te._run_tool_activity_heartbeat,
        args=(_Agent(), stop, "tool running: terminal"),
        kwargs={"interval": 0.05, "worker_tid": fake_worker_tid},
        daemon=True,
    )
    thread.start()
    try:
        assert touched.wait(5), "heartbeat never stamped while the worker was live"
        set_interrupt(True, fake_worker_tid)
        thread.join(timeout=5)
        assert not thread.is_alive(), "heartbeat kept running after its worker was abandoned"
        assert not stop.is_set(), "heartbeat must exit on the worker interrupt alone"
    finally:
        stop.set()
        thread.join(timeout=5)
        set_interrupt(False, fake_worker_tid)


def test_sequential_timeout_stops_abandoned_workers_heartbeat(monkeypatch, heartbeat_threads):
    """After the sequential deadline abandons a non-cooperative tool, no more
    ``tool running:`` stamps arrive — the timed-out tool no longer fakes liveness (#111922)."""
    import agent.tool_executor as te

    monkeypatch.setattr(te, "_SEQUENTIAL_INTERRUPT_POLL_SECONDS", 0.05)
    monkeypatch.setattr(te, "_resolve_sequential_tool_timeout", lambda: 0.3)
    monkeypatch.setattr(te, "_emit_terminal_post_tool_call", lambda agent, **kw: None)
    # The middleware's execution seam: ``_run_with_activity_heartbeat`` around ``execute`` on the worker.
    monkeypatch.setattr(
        te, "_run_agent_tool_execution_middleware",
        lambda agent, execute, function_name="terminal", **kw: te._ManagedToolResult(
            result=te._run_with_activity_heartbeat(agent, function_name, lambda: execute({})),
            args={}, middleware_trace=[], blocked=False, dispatched=True,
        ),
    )

    class _Agent:
        _interrupt_requested = False
        _tool_interrupt_reason = None

        def __init__(self):
            self._tool_worker_threads = set()
            self._tool_worker_threads_lock = threading.Lock()
            self.stamps: list = []

        def _touch_activity(self, desc):
            self.stamps.append((time.monotonic(), desc))

        def interrupt(self, *a, **k):
            pass

    agent = _Agent()
    release = threading.Event()
    try:
        managed = te._run_sequential_tool_execution_middleware(
            agent, function_name="terminal", function_args={}, effective_task_id="t",
            tool_call_id="tc1", execute=lambda args: release.wait() or "late",
        )
        assert isinstance(managed.result, te._ToolTimeoutResult)
        assert len(heartbeat_threads) == 1
        thread, stop = heartbeat_threads[0]
        thread.join(timeout=5)
        assert not thread.is_alive(), "abandoned worker's heartbeat did not exit"
        assert not stop.is_set(), "worker must still be blocked, without normal shutdown"
    finally:
        release.set()
