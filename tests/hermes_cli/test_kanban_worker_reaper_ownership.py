"""The gateway dispatcher must not consume another subprocess owner's exit status."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd


pytestmark = pytest.mark.platforms("posix")


@pytest.fixture(autouse=True)
def isolated_worker_registries(monkeypatch):
    monkeypatch.setattr(kbd, "_live_worker_procs", {})
    monkeypatch.setattr(kbd, "_recent_worker_exits", {})


def _wait_for_zombie(pid):
    """Observe exit without collecting the status that the reaper must leave alone."""
    process = psutil.Process(pid)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.status() == psutil.STATUS_ZOMBIE:
            return
        time.sleep(0.01)
    pytest.fail(f"child {pid} did not exit")


def test_non_worker_keeps_its_real_exit_status():
    with subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"]) as proc:
        _wait_for_zombie(proc.pid)
        reaped = kbd.reap_worker_zombies()
        assert proc.wait(timeout=10) == 3
        assert proc.pid not in reaped
        assert kbd._classify_worker_exit(proc.pid) == ("unknown", None)


@pytest.fixture
def spawn_worker(tmp_path, monkeypatch):
    # Keep profile resolution and every filesystem side effect in the test home.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    children = []

    def spawn(code):
        # Only replace the agent payload; spawning, tracking and waitpid are real.
        monkeypatch.setattr(
            kbd, "_worker_argv",
            lambda *args: [sys.executable, "-c", code],
        )
        task = kb.Task(
            id="t_reaper", title="reaper test", body=None, assignee="default",
            status="running", priority=0, created_by=None, created_at=0,
            started_at=None, completed_at=None, workspace_kind="dir",
            workspace_path=str(tmp_path), claim_lock=None, claim_expires=None,
            tenant=None,
        )
        pid = kbd._default_spawn(task, str(tmp_path))
        children.append(pid)
        return pid

    yield spawn

    for pid in children:
        try:
            waited_pid, status = os.waitpid(pid, os.WNOHANG)
            if waited_pid == 0:
                os.kill(pid, signal.SIGKILL)
                _, status = os.waitpid(pid, 0)
            proc = kbd._live_worker_procs.pop(pid, None)
            if proc is not None:
                proc.returncode = os.waitstatus_to_exitcode(status)
        except ChildProcessError:
            pass


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("raise SystemExit(3)", ("nonzero_exit", 3)),
        ("import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
         ("signaled", signal.SIGTERM)),
        (f"raise SystemExit({kb.KANBAN_RATE_LIMIT_EXIT_CODE})",
         ("rate_limited", kb.KANBAN_RATE_LIMIT_EXIT_CODE)),
    ],
)
def test_spawned_worker_is_reaped_and_classified(spawn_worker, code, expected):
    pid = spawn_worker(code)
    proc = kbd._live_worker_procs[pid]
    _wait_for_zombie(pid)
    assert kbd.reap_worker_zombies() == [pid]
    assert kbd._classify_worker_exit(pid) == expected
    assert proc.wait(timeout=10) == (-expected[1] if expected[0] == "signaled" else expected[1])
    assert pid not in kbd._live_worker_procs
    assert kbd.reap_worker_zombies() == []


def test_already_reaped_worker_is_dropped_without_inventing_a_status():
    with subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"]) as proc:
        kbd._live_worker_procs[proc.pid] = proc
        assert proc.wait(timeout=10) == 3
        assert kbd.reap_worker_zombies() == []
        assert proc.pid not in kbd._live_worker_procs
        assert kbd._classify_worker_exit(proc.pid) == ("unknown", None)


def test_running_worker_does_not_prevent_reaping_an_exited_worker(spawn_worker):
    running_pid = spawn_worker("import time; time.sleep(30)")
    exited_pid = spawn_worker("raise SystemExit(3)")
    _wait_for_zombie(exited_pid)
    assert kbd.reap_worker_zombies() == [exited_pid]
    assert running_pid in kbd._live_worker_procs
    assert kbd._live_worker_procs[running_pid].poll() is None
    assert kbd._classify_worker_exit(running_pid) == ("unknown", None)
    assert kbd._classify_worker_exit(exited_pid) == ("nonzero_exit", 3)


def test_worker_reaping_leaves_unrelated_child_status_untouched(spawn_worker):
    worker_pid = spawn_worker("raise SystemExit(75)")
    with subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"]) as other:
        _wait_for_zombie(worker_pid)
        _wait_for_zombie(other.pid)
        assert kbd.reap_worker_zombies() == [worker_pid]
        assert other.wait(timeout=10) == 3
        assert kbd._classify_worker_exit(other.pid) == ("unknown", None)
