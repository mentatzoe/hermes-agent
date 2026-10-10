"""A PTY reader's bounded reap handoff must not lose a later real exit."""
import threading

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


pytestmark = pytest.mark.platforms("posix")


def test_poll_reports_late_pty_signal(handed_off):
    registry, session, pty = handed_off
    pty.signalstatus = 15

    result = registry.poll(session.id)

    assert result["status"] == "exited"
    assert result["exit_code"] == -15
    assert registry.completion_queue.get_nowait()["exit_code"] == -15
    assert pty.closes == 1
    assert pty.waits == 1


class HandedOffPty:
    """No OS PTY required: EOF followed by ECHILD leaves status unresolved."""

    def __init__(self):
        self.exitstatus: int | None = None
        self.signalstatus: int | None = None
        self.flag_eof = True
        self.waits = 0
        self.closes = 0
        self.checks = 0

    def isalive(self):
        self.checks += 1
        return True

    def read(self, size):
        raise EOFError

    def wait(self):
        self.waits += 1
        raise ChildProcessError("ECHILD")

    def close(self):
        self.closes += 1


@pytest.fixture
def handed_off(monkeypatch):
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    pty = HandedOffPty()
    session = ProcessSession(id="proc_pty_handoff", command="test", notify_on_complete=True)
    session._pty = pty
    registry._running[session.id] = session
    reader = threading.Thread(target=registry._pty_reader_loop, args=(session,), daemon=True)
    session._reader_thread = reader
    reader.start()
    reader.join(timeout=3)
    assert not reader.is_alive(), "reader must hand off without a client poll"
    assert pty.waits == 1
    assert not session.exited
    assert not session._completion_event.is_set()
    assert registry.completion_queue.empty()
    assert session.id in registry._running
    return registry, session, pty


def test_stored_pty_status_without_reader_is_reconciled(monkeypatch):
    """Also runs on plain 0.21.6, independently of the F5 reader handoff fix."""
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    pty = HandedOffPty()
    pty.exitstatus = 9
    session = ProcessSession(id="proc_pty_stored", command="test", notify_on_complete=True)
    session._pty = pty
    registry._running[session.id] = session
    result = registry.poll(session.id)
    assert result["status"] == "exited"
    assert result["exit_code"] == 9
    assert registry.completion_queue.get_nowait()["exit_code"] == 9
    assert pty.waits == 0


def test_poll_publishes_late_pty_exit_once(handed_off):
    registry, session, pty = handed_off
    pty.exitstatus = 9

    result = registry.poll(session.id)

    assert result["status"] == "exited"
    assert result["exit_code"] == 9
    assert session._completion_event.is_set()
    assert session.id not in registry._running
    assert registry._finished[session.id] is session
    notice = registry.completion_queue.get_nowait()
    assert notice["session_id"] == session.id
    assert notice["exit_code"] == 9
    assert pty.closes == 1
    registry.poll(session.id)
    assert registry.completion_queue.empty()
    assert pty.closes == 1
    assert pty.waits == 1, "reconciliation must not call blocking wait"


@pytest.mark.parametrize("eof", [False, True])
def test_unknown_status_stays_tracked_without_touching_waiters(handed_off, monkeypatch, eof):
    from ptyprocess import PtyProcessError

    registry, session, pty = handed_off
    pty.flag_eof = eof

    def unavailable_status():
        raise PtyProcessError("status consumed elsewhere")

    monkeypatch.setattr(pty, "isalive", unavailable_status)
    for _ in range(2):
        assert registry.poll(session.id)["status"] == "running"
    assert session.exit_code is None
    assert not session.exited
    assert not session._completion_event.is_set()
    assert registry.completion_queue.empty()
    assert registry._running[session.id] is session
    assert session._pty is pty
    assert pty.closes == 0
    assert pty.waits == 1


def test_live_reader_keeps_exclusive_handle_ownership(handed_off):
    registry, session, _ = handed_off
    release = threading.Event()
    reader = threading.Thread(target=release.wait, daemon=True)

    class ReaderOwnedPty:
        def __getattr__(self, name):
            pytest.fail(f"reconciliation touched live reader's PTY: {name}")

    session._pty = ReaderOwnedPty()
    session._reader_thread = reader
    reader.start()
    try:
        assert registry.poll(session.id)["status"] == "running"
        assert not session.exited
        assert session.exit_code is None
        assert not session._completion_event.is_set()
        assert registry.completion_queue.empty()
        assert registry._running[session.id] is session
    finally:
        release.set()
        reader.join(timeout=3)
    assert not reader.is_alive()


@pytest.mark.parametrize("query", ["list", "wait", "completion_wait"])
def test_other_status_queries_reconcile_pty_handoff(handed_off, query):
    registry, session, pty = handed_off
    pty.exitstatus = 9
    if query == "list":
        result, = registry.list_sessions()
    elif query == "wait":
        result = registry.wait(session.id, timeout=1)
    else:
        linger = registry.wait_for_pending_completions(timeout=1)
        assert linger["completed"] == [session.id]
        assert linger["timed_out"] == []
        result = registry.poll(session.id)
    assert result["status"] == "exited"
    assert result["exit_code"] == 9
    assert session._completion_event.is_set()
    assert registry.completion_queue.get_nowait()["exit_code"] == 9
    assert registry.completion_queue.empty()
    assert pty.closes == 1
    assert pty.waits == 1


def test_exit_status_takes_precedence_over_signal(handed_off):
    registry, session, pty = handed_off
    pty.exitstatus = 0
    pty.signalstatus = 15
    assert registry.poll(session.id)["exit_code"] == 0
    assert registry.completion_queue.get_nowait()["exit_code"] == 0


def test_kill_between_status_read_and_finish_keeps_killed_result(handed_off, monkeypatch):
    registry, session, pty = handed_off
    pty.exitstatus = 9
    finish = registry._finish_exited
    # Only OS signaling is replaced; kill commits, persistence, completion queue
    # and reconciliation all use the real production paths.
    monkeypatch.setattr(registry, "_signal_kill", lambda *args: None)
    monkeypatch.setattr(registry, "_post_kill_survivors", lambda *args: [])

    def kill_then_finish(observed, code):
        assert code == 9
        assert registry.kill_process(session.id)["status"] == "killed"
        finish(observed, code)

    monkeypatch.setattr(registry, "_finish_exited", kill_then_finish)
    result = registry.poll(session.id)
    assert result["status"] == "exited"
    assert result["exit_code"] == -15
    assert result["completion_reason"] == "killed"
    assert result["termination_source"] == "process.kill"
    assert session._completion_event.is_set()
    notice = registry.completion_queue.get_nowait()
    assert notice["exit_code"] == -15
    assert notice["completion_reason"] == "killed"
    assert registry.completion_queue.empty()
