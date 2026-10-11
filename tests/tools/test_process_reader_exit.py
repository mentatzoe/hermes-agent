"""Output EOF/failure is not process completion (2026-10-07 regression).

All registries/queues are private to the foreground test process. No gateway
watcher is started, and real subprocesses are never notification-enabled.
"""
import io
import os
import shlex
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture
def registry(monkeypatch):
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    return registry


def track(registry, proc):
    session = ProcessSession(id="proc_exit_regression", command="test", process=proc)
    registry._running[session.id] = session
    return session


@pytest.mark.parametrize("stream", ["eof", "read_error", "missing"])
@pytest.mark.parametrize("wait_error", ["timeout", "oserror"])
def test_pipe_reader_keeps_waiting_after_output_ends(registry, stream, wait_error):
    proc = Mock(returncode=None)
    proc.stdout = io.StringIO("") if stream == "eof" else None
    if stream == "read_error":
        proc.stdout = Mock()
        proc.stdout.buffer.read1.side_effect = OSError("capture pipe failed")
    session = track(registry, proc)
    # Synthetic process only: inspect the notification contract in a private queue.
    session.notify_on_complete = True
    calls = []

    def wait(timeout=None):
        calls.append(timeout)
        assert not session.exited
        assert session.id in registry._running
        assert not session._completion_event.is_set()
        assert registry.completion_queue.empty()
        if len(calls) == 1:
            if wait_error == "timeout":
                raise subprocess.TimeoutExpired("test", timeout or 0)
            raise OSError("transient wait failure")
        proc.returncode = 7
        return 7

    proc.wait.side_effect = wait
    proc.poll.side_effect = lambda: proc.returncode
    registry._reader_loop(session)
    assert len(calls) >= 2
    assert session.exited and session.exit_code == 7
    assert session._completion_event.is_set()
    item = registry.completion_queue.get_nowait()
    assert item["exit_code"] == 7
    registry._move_to_finished(session)
    assert registry.completion_queue.empty()


@pytest.mark.parametrize("wait_error", [True, False])
def test_pty_read_end_does_not_imply_exit(registry, wait_error):
    pty = Mock(exitstatus=None, signalstatus=None)
    pty.read.side_effect = EOFError
    pty.isalive.side_effect = lambda: pty.exitstatus is None
    session = track(registry, None)
    session._pty = pty
    calls = []

    def wait():
        calls.append(1)
        assert not session.exited
        assert not session._completion_event.is_set()
        if len(calls) == 1:
            if wait_error:
                raise OSError("transient PTY wait failure")
            return None
        pty.exitstatus = 9
        return 9

    pty.wait.side_effect = wait
    registry._pty_reader_loop(session)
    assert len(calls) >= 2
    assert session.exited and session.exit_code == 9


def test_pty_signal_exit_has_concrete_status(registry):
    pty = Mock(exitstatus=None, signalstatus=15)
    pty.isalive.return_value = False
    pty.wait.return_value = None
    session = track(registry, None)
    session._pty = pty
    registry._pty_reader_loop(session)
    assert session.exited and session.exit_code == -15


def test_missing_process_handle_is_not_successful_exit(registry):
    session = track(registry, None)
    registry._reader_loop(session)
    assert not session.exited
    assert not session._completion_event.is_set()
    assert session.id in registry._running


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell/descriptor integration")
@pytest.mark.parametrize("output_mode", ["quiet", "redirected", "closed", "broken_reader"])
def test_real_local_process_waits_for_exit(registry, monkeypatch, tmp_path, output_mode):
    # File handshake keeps the real child alive until after our assertions;
    # timeout bounds the child even if the test runner itself fails.
    ready, release = tmp_path / "ready", tmp_path / "release"
    code = (
        "import os,time,pathlib,sys; "
        + ("os.close(1); os.close(2); " if output_mode == "closed" else "")
        + f"pathlib.Path({str(ready)!r}).touch(); "
        + "deadline=time.monotonic()+20\n"
        + f"while not pathlib.Path({str(release)!r}).exists() and time.monotonic()<deadline: time.sleep(.01)\n"
        + "sys.exit(7)\n"
    )
    command = f"exec {shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    if output_mode == "redirected":
        command += f" > {shlex.quote(str(tmp_path / 'output'))} 2>&1"
    # Bypass user shell rc files, not the production local spawn/reader path.
    monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/sh")
    monkeypatch.setattr("tools.process_registry._SYSTEMD_SCOPE_AVAILABLE", False)
    if output_mode == "broken_reader":
        import select
        monkeypatch.setattr(select, "select", Mock(side_effect=ValueError("closed fd")))
    session = registry.spawn_local(command, cwd=str(tmp_path), env_vars={"HOME": str(tmp_path)})
    proc = session.process
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists(), session.output_buffer
        # Cross the old hard-coded five-second reap timeout when output ended.
        hold = 5.5 if output_mode in {"redirected", "closed", "broken_reader"} else .3
        completed_early = session._completion_event.wait(hold)
        assert proc.poll() is None, "fixture child must still be alive"
        assert not completed_early, "output ended but the OS process is still alive"
        assert not session.exited
        assert registry.poll(session.id)["status"] == "running"
        assert session.id in registry._running
        release.touch()
        assert session._completion_event.wait(5), "exit must complete without client polling"
        assert proc.poll() == 7
        assert session.exited and session.exit_code == 7
        assert session.id in registry._finished
        assert registry.completion_queue.empty()
    finally:
        release.touch()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
        session._reader_thread.join(timeout=3)
        if proc.stdout:
            proc.stdout.close()
