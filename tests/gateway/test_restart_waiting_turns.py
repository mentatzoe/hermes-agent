"""Waiting-only turns release both restart drains and retain recovery (#242)."""
import asyncio
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.interrupt_control import InterruptControlMixin
from agent.tool_executor import _registered_tool_worker
from gateway.run_restart_wait import yield_restart_waiters
from gateway.session import SessionStore
from tools.process_registry import _handle_process
from tools.restart_wait import claim_waits
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


class WaitAgent(InterruptControlMixin):
    def __init__(self, sid):
        self.session_id = sid
        self._current_turn_id = "first"
        self._tool_worker_threads = set()
        self._tool_worker_threads_lock = threading.Lock()
        self._active_children = []
        self._active_children_lock = threading.Lock()
        self._pending_redirect_lock = threading.Lock()
        self._pending_redirect = None
        self._interrupt_requested = False
        self._execution_thread_id = None
        self.quiet_mode = True


def start_wait(runner, key, agent, path, unwind_gate=None):
    done, results = threading.Event(), []
    runner._running_agents[key] = agent

    def run():
        try:
            with _registered_tool_worker(agent):
                results.append(json.loads(_handle_process(
                    {"action": "wait_for", "condition": "file_exists", "path": str(path), "timeout": 20},
                    task_id=agent.session_id,
                )))
                if unwind_gate is not None:
                    unwind_gate.wait(10)
        finally:
            if runner._running_agents.get(key) is agent:
                runner._running_agents.pop(key)
            done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        claim = claim_waits(agent)
        if claim is not None:
            claim.release()
            return done, results, thread
        done.wait(0.01)
    path.touch()
    thread.join(5)
    pytest.fail("wait did not register")


def make_wait_runner(tmp_path):
    runner, adapter = make_restart_runner()
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    entry = runner.session_store.get_or_create_session(make_restart_source(chat_id="observer"))
    runner._scale_to_zero_status = lambda *args: None
    return runner, adapter, entry


@pytest.mark.asyncio
async def test_waiting_only_restart_drains_promptly_and_boot_resumes_the_saved_turn(tmp_path):
    runner, _, entry = make_wait_runner(tmp_path)
    agent = WaitAgent(entry.session_id)
    receipt = tmp_path / "exit.json"
    done, results, thread = start_wait(runner, entry.session_key, agent, receipt)
    drains = []

    async def stop(**kwargs):
        drains.append(await runner._drain_active_agents(1800, cron_timeout=0))

    runner.stop = stop
    try:
        assert runner.request_restart(via_service=True)
        # This is a real 1800s after-turn budget and a second 1800s drain,
        # bounded by the test only to catch either retaining the observer.
        await asyncio.wait_for(runner._restart_task, 5)
        assert done.is_set() and results[0]["status"] == "interrupted"
        assert drains and drains[0][1] is False
        assert not receipt.exists()  # transport/pass completion was not invented
        runner.session_store.close_all_db_handles()
        restored = SessionStore(tmp_path / "sessions", runner.config)
        saved = restored.lookup_by_session_key(entry.session_key)
        assert saved.session_id == entry.session_id
        assert saved.resume_pending and saved.resume_reason == "restart_wait"
        next_runner, adapter = make_restart_runner()
        next_runner.session_store = restored
        resumed = []

        async def handle(event):
            resumed.append(event.source.chat_id)

        adapter.handle_message = handle
        assert next_runner._schedule_resume_pending_sessions() == 1
        assert next_runner._schedule_resume_pending_sessions() == 0
        await asyncio.gather(*next_runner._background_tasks)
        assert resumed == ["observer"]
        restored.close_all_db_handles()
    finally:
        receipt.touch()
        thread.join(5)
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_zero_after_turn_budget_keeps_wait_recovery_through_the_real_stop_drain(tmp_path):
    runner, _, entry = make_wait_runner(tmp_path)
    agent = WaitAgent(entry.session_id)
    receipt, unwind = tmp_path / "exit", threading.Event()
    done, results, thread = start_wait(runner, entry.session_key, agent, receipt, unwind)
    actual_drain = runner._drain_active_agents
    runner._restart_after_turn_timeout = 0
    runner._cron_drain_timeout = 0
    ctx = SimpleNamespace(elapsed=lambda: 0, deferred_count=lambda: 0)

    async def drain(timeout, cron_timeout):
        saved = runner.session_store.lookup_by_session_key(entry.session_key)
        assert saved.resume_reason == "restart_wait"
        unwind.set()
        return await actual_drain(timeout, cron_timeout)

    async def stop(**kwargs):
        await runner._stop_drain_active_work(5, ctx)

    runner._drain_active_agents, runner.stop = drain, stop
    try:
        runner.request_restart(via_service=True)
        await asyncio.wait_for(runner._restart_task, 5)
        assert done.is_set() and results[0]["status"] == "interrupted"
        assert not ctx.timed_out
        runner.session_store.close_all_db_handles()
        restored = SessionStore(tmp_path / "sessions", runner.config)
        saved = restored.lookup_by_session_key(entry.session_key)
        assert saved.resume_pending and saved.resume_reason == "restart_wait"
        restored.close_all_db_handles()
    finally:
        unwind.set()
        receipt.touch()
        thread.join(5)
        runner.session_store.close_all_db_handles()


def test_failed_save_does_not_publish_a_resume_marker(tmp_path, monkeypatch):
    runner, _, entry = make_wait_runner(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(runner.session_store, "_save_entry", fail)
    try:
        with pytest.raises(OSError, match="disk unavailable"):
            runner.session_store.mark_resume_pending(entry.session_key, "restart_wait", expected_session_id=entry.session_id)
        assert not entry.resume_pending and entry.resume_reason is None
    finally:
        runner.session_store.close_all_db_handles()


def test_saved_wait_continues_without_asking_the_user_again():
    from gateway.run import build_resume_recovery_note
    note = build_resume_recovery_note("restart_wait", interactive=True)
    assert "CONTINUE the interrupted task" in note
    assert "ask what they would like" not in note
    note = build_resume_recovery_note("restart_wait", "new instruction", interactive=True)
    assert "NEW message" in note and note.endswith("new instruction")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["refused", "exception"])
async def test_no_wait_is_interrupted_without_a_durable_resume_marker(tmp_path, failure):
    runner, _, entry = make_wait_runner(tmp_path)
    agent = WaitAgent(entry.session_id)
    receipt = tmp_path / "exit"
    done, results, thread = start_wait(runner, entry.session_key, agent, receipt)
    mark = AsyncMock(return_value=False, side_effect=OSError("disk unavailable") if failure == "exception" else None)
    runner._async_session_store = SimpleNamespace(_store=runner.session_store, mark_resume_pending=mark)
    try:
        await yield_restart_waiters(runner)
        assert not agent._interrupt_requested
        assert not done.is_set()
        assert runner._awaitable_work_count() == 1
        receipt.touch()
        assert await asyncio.to_thread(done.wait, 5)
        assert results[0]["status"] == "ready"
    finally:
        receipt.touch()
        thread.join(5)
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_receipt_resolution_during_save_cannot_launch_the_next_tool(tmp_path):
    runner, _, entry = make_wait_runner(tmp_path)
    agent = WaitAgent(entry.session_id)
    receipt = tmp_path / "exit"
    done, results, thread = start_wait(runner, entry.session_key, agent, receipt)
    entered, finish_save = asyncio.Event(), asyncio.Event()
    actual = runner.async_session_store.mark_resume_pending

    async def mark(key, reason, **kwargs):
        entered.set()
        await finish_save.wait()
        return await actual(key, reason, **kwargs)

    runner._async_session_store = SimpleNamespace(_store=runner.session_store, mark_resume_pending=mark)
    task = asyncio.create_task(yield_restart_waiters(runner))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        receipt.touch()
        assert not done.is_set()
        finish_save.set()
        await asyncio.wait_for(task, 5)
        assert await asyncio.to_thread(done.wait, 5)
        assert results[0]["status"] == "interrupted"
    finally:
        finish_save.set()
        await task
        receipt.touch()
        thread.join(5)
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["reset", "suspend"])
async def test_replaced_or_suspended_conversation_is_not_marked_for_wait_recovery(tmp_path, change):
    runner, _, entry = make_wait_runner(tmp_path)
    agent = WaitAgent(entry.session_id)
    receipt = tmp_path / "exit"
    done, results, thread = start_wait(runner, entry.session_key, agent, receipt)
    try:
        if change == "reset":
            runner.session_store.reset_session(entry.session_key)
        else:
            runner.session_store.suspend_session(entry.session_key)
        await yield_restart_waiters(runner)
        current = runner.session_store.lookup_by_session_key(entry.session_key)
        assert not current.resume_pending
        assert not agent._interrupt_requested
        receipt.touch()
        assert await asyncio.to_thread(done.wait, 5)
        assert results[0]["status"] == "ready"
    finally:
        receipt.touch()
        thread.join(5)
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_releasing_observer_does_not_shortcut_a_working_turn(tmp_path):
    runner, _, entry = make_wait_runner(tmp_path)
    agent = WaitAgent(entry.session_id)
    receipt = tmp_path / "exit"
    done, results, thread = start_wait(runner, entry.session_key, agent, receipt)
    busy = WaitAgent("busy")
    runner._running_agents["busy"] = busy
    stopped = asyncio.Event()

    async def stop(**kwargs):
        stopped.set()

    runner.stop = stop
    try:
        runner.request_restart(via_service=True)
        assert await asyncio.to_thread(done.wait, 5)
        assert results[0]["status"] == "interrupted"
        assert not stopped.is_set() and not busy._interrupt_requested
        runner._running_agents.pop("busy")
        await asyncio.wait_for(runner._restart_task, 5)
        assert stopped.is_set()
    finally:
        runner._running_agents.pop("busy", None)
        receipt.touch()
        thread.join(5)
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize("owned", ["process", "delegation", "child", "mutator"])
async def test_wait_does_not_hide_owned_work_or_a_parallel_mutator(tmp_path, monkeypatch, owned):
    runner, _, entry = make_wait_runner(tmp_path)
    agent = WaitAgent(entry.session_id)
    receipt = tmp_path / "exit"
    done, results, thread = start_wait(runner, entry.session_key, agent, receipt)
    from tools.process_registry import process_registry
    if owned == "process":
        monkeypatch.setitem(process_registry._running, "test-owned", SimpleNamespace(
            exited=False, owner_task_id=agent.session_id, session_key=entry.session_key,
        ))
    elif owned == "delegation":
        monkeypatch.setattr("tools.async_delegation.has_live_for_session", lambda **kwargs: True)
    elif owned == "child":
        agent._active_children.append(object())
    else:
        with agent._tool_worker_threads_lock:
            agent._tool_worker_threads.add(-1)
    try:
        await yield_restart_waiters(runner)
        assert not agent._interrupt_requested
        assert runner._awaitable_work_count() == 1
        receipt.touch()
        assert await asyncio.to_thread(done.wait, 5)
        assert results[0]["status"] == "ready"
    finally:
        receipt.touch()
        thread.join(5)
        runner.session_store.close_all_db_handles()
