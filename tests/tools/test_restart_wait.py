"""Pure observation waits and their turn-scoped restart fence (Aleph Vault #242)."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

from tools.process_registry import _handle_process
from tools.restart_wait import bind_wait_owner, claim_waits, wait_for_condition, wait_for_dispatch


def _agent():
    return SimpleNamespace(
        _current_turn_id="first", _tool_worker_threads=set(),
        _tool_worker_threads_lock=threading.Lock(),
    )


def _start_wait(agent, path):
    result, done = [], threading.Event()

    def run():
        from agent.tool_executor import _registered_tool_worker
        with _registered_tool_worker(agent), bind_wait_owner(agent):
            result.append(wait_for_condition("file_exists", str(path), 10, task_id="owner"))
        done.set()

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        claim = claim_waits(agent)
        if claim is not None:
            return result, done, thread, claim
        done.wait(0.01)
    path.touch()
    thread.join(5)
    pytest.fail("native wait did not register")


def test_handler_observes_timer_without_a_process():
    result = json.loads(_handle_process({"action": "wait_for", "condition": "timer", "timeout": 1}))
    assert result == {"status": "ready", "condition": "timer"}


def test_file_wait_never_creates_or_consumes_the_receipt(tmp_path):
    path = tmp_path / "exit.json"
    assert wait_for_condition("file_exists", str(path), 0.01)["status"] == "timeout"
    assert not path.exists()
    path.write_text('{"returncode":0}')
    assert wait_for_condition("file_exists", str(path), 1)["status"] == "ready"
    assert path.read_text() == '{"returncode":0}'
    assert wait_for_condition("file_missing", str(path), 0.01)["status"] == "timeout"


@pytest.mark.parametrize("args", [
    {"condition": "command", "path": "/unused"},
    {"condition": "file_exists", "path": "relative"},
    {"condition": "timer", "path": "/unused"},
    {"condition": "timer", "timeout": True},
    {"condition": "timer", "timeout": float("inf")},
    {"condition": "timer", "timeout": -1},
])
def test_invalid_waits_fail_without_execution(args):
    assert wait_for_condition(**args)["status"] == "error"


@pytest.mark.platforms("posix")
def test_lock_observation_is_read_only_and_does_not_take_ownership(tmp_path):
    import fcntl
    path = tmp_path / "controller.lock"
    assert wait_for_condition("lock_free", str(path), 1)["status"] == "ready"
    assert not path.exists()
    path.write_text("unchanged")
    with path.open("r") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert wait_for_condition("lock_free", str(path), 0.01)["status"] == "timeout"
    assert wait_for_condition("lock_free", str(path), 1)["status"] == "ready"
    assert path.read_text() == "unchanged"
    assert wait_for_condition("lock_free", str(tmp_path), 1)["status"] == "error"


def test_claim_fences_readiness_and_new_dispatch_until_persistence_finishes(tmp_path):
    agent = _agent()
    path = tmp_path / "exit"
    result, done, thread, claim = _start_wait(agent, path)
    attempted, admitted = threading.Event(), []

    def next_dispatch():
        attempted.set()
        admitted.append(wait_for_dispatch(agent))

    next_thread = threading.Thread(target=next_dispatch)
    next_thread.start()
    try:
        assert attempted.wait(5)
        path.touch()
        assert not done.is_set()
        claim.release(interrupted=True)
        assert done.wait(5)
        next_thread.join(5)
        assert admitted == [False]
        assert result == [{"status": "interrupted", "condition": "file_exists"}]
        agent._current_turn_id = "successor"
        assert wait_for_dispatch(agent)
        assert claim_waits(agent) is None
    finally:
        claim.release()
        path.touch()
        thread.join(5)
        next_thread.join(5)


def test_failed_claim_release_allows_the_same_wait_to_finish(tmp_path):
    agent = _agent()
    path = tmp_path / "exit"
    result, done, thread, claim = _start_wait(agent, path)
    try:
        path.touch()
        claim.release()
        assert done.wait(5)
        assert result == [{"status": "ready", "condition": "file_exists"}]
        assert wait_for_dispatch(agent)
    finally:
        claim.release()
        path.touch()
        thread.join(5)


def test_concurrent_mutator_and_other_turns_are_not_claimed(tmp_path):
    agent, other = _agent(), _agent()
    path = tmp_path / "exit"
    result, done, thread, claim = _start_wait(agent, path)
    try:
        claim.release()
        with agent._tool_worker_threads_lock:
            agent._tool_worker_threads.add(-1)  # an executing non-wait tool
        assert claim_waits(agent) is None
        assert claim_waits(other) is None
        with agent._tool_worker_threads_lock:
            agent._tool_worker_threads.remove(-1)
        claim = claim_waits(agent)
        assert claim is not None
        assert claim_waits(agent) is None
        claim.release(interrupted=True)
        assert done.wait(5)
        assert result[0]["status"] == "interrupted"
    finally:
        claim.release()
        path.touch()
        thread.join(5)


def test_delayed_worker_cannot_register_a_wait_for_a_successor_turn(tmp_path):
    from agent.tool_executor import _registered_tool_worker
    from tools.thread_context import propagate_context_to_thread
    agent = _agent()
    path, done, results = tmp_path / "exit", threading.Event(), []

    def old_worker():
        with _registered_tool_worker(agent):
            results.append(wait_for_condition("file_exists", str(path), 10, task_id="old"))
        done.set()

    with bind_wait_owner(agent):
        submitted = propagate_context_to_thread(old_worker)
    agent._current_turn_id = "new-model-turn"
    thread = threading.Thread(target=submitted)
    thread.start()
    try:
        assert done.wait(5), "an abandoned worker was attributed to the successor turn"
        assert results[0]["status"] == "interrupted"
        assert claim_waits(agent) is None
    finally:
        path.touch()
        thread.join(5)


def test_restart_admission_refuses_shell_and_python_before_any_environment_work(monkeypatch):
    import gateway.run as gateway_run
    import tools.terminal_tool as terminal
    import tools.code_execution_tool as code
    runner = SimpleNamespace(_restart_requested=True)
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)

    def unexpected(*args, **kwargs):
        pytest.fail("execution or environment acquisition started during restart")

    monkeypatch.setattr(terminal, "_plan_execution", unexpected)
    monkeypatch.setattr(code, "_load_config", unexpected)
    assert "restart requested" in terminal.terminal_tool("unexecuted command")
    assert "restart requested" in code.execute_code("raise AssertionError('must not run')")
    runner._restart_requested = False
    from tools.restart_wait import launch_block_reason
    assert launch_block_reason() is None
