"""A live worker stays alive to a dispatcher in another process, while a recycled PID stays foreign.

``worker_started_at`` is written by whichever process spawned the worker (a one-shot
``hermes kanban dispatch`` or the gateway's dispatcher) and read later by whichever process runs the
next sweep. On macOS psutil shifts ``create_time()`` by however far ``kern.boottime`` has moved since
psutil was imported, so a gateway that has been up a day and a fresh CLI read the same worker's start
about a second apart (aleph-vault #215: ``|179246502392`` recorded, ``|179246502492`` read). The exact
comparison read that as a recycled PID, reclaimed the card and started a second worker beside the
first. The composed fingerprint's start half now agrees within ``START_TIME_DRIFT_TOLERANCE``; the
epoch half, a malformed reading and an integer (pre-epoch) fingerprint stay exact.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gateway.status import START_TIME_DRIFT_TOLERANCE
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_worker_identity import fingerprints_match

ROOT = Path(__file__).resolve().parents[2]

# The two readings from aleph-vault #215, one per direction.
CLI_RECORDED, GATEWAY_READ = "|179246502392", "|179246502492"      # run 169, pid 70602
GATEWAY_RECORDED, CLI_READ = "|179246505617", "|179246505517"      # run 170, pid 71722


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


def _running_card(conn, *, pid: int, started_at) -> str:
    """A host-local running card whose launch grace and claim TTL have long passed."""
    tid = kb.create_task(conn, title="job", assignee="worker")
    kb.claim_task(conn, tid)
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_pid = ?, worker_started_at = ?, started_at = ?, claim_expires = ? "
                     "WHERE id = ?", (pid, started_at, old, old, tid))
        conn.execute("UPDATE task_runs SET worker_pid = ?, worker_started_at = ?, started_at = ? "
                     "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)", (pid, started_at, old, tid))
    return tid


@pytest.mark.parametrize(("recorded", "current"), [(CLI_RECORDED, GATEWAY_READ), (GATEWAY_RECORDED, CLI_READ)])
def test_the_215_readings_are_the_same_worker(recorded, current):
    assert fingerprints_match(recorded, current) is True


@pytest.mark.parametrize(("offset", "same"), [
    (0, True), (START_TIME_DRIFT_TOLERANCE, True), (-START_TIME_DRIFT_TOLERANCE, True),
    (START_TIME_DRIFT_TOLERANCE + 1, False), (-START_TIME_DRIFT_TOLERANCE - 1, False), (360000, False),
])
def test_start_half_tolerance_is_the_shared_two_seconds(offset, same):
    recorded = "4f1c-boot:7|179246502392"
    assert fingerprints_match(recorded, f"4f1c-boot:7|{179246502392 + offset}") is same


@pytest.mark.parametrize("current", [
    None, "", "179246502392", 179246502392,
    "4f1c-boot:7|", "4f1c-boot:7|abc", "4f1c-boot:7|1792465024.5",
    "other-boot:9|179246502392", "|179246502392",
])
def test_unreadable_malformed_or_other_boot_never_matches(current):
    assert fingerprints_match("4f1c-boot:7|179246502392", current) is False


@pytest.mark.parametrize("recorded", ["|", "|abc", "4f1c-boot:7|", "4f1c-boot:7|1e9"])
def test_malformed_recorded_fingerprint_never_matches(recorded):
    assert fingerprints_match(recorded, "4f1c-boot:7|179246502392") is False


def test_integer_fingerprint_keeps_the_exact_comparison():
    """A pre-epoch row has no reboot witness, so only an exact start time may count as ours."""
    assert fingerprints_match(179246502392, 179246502392) is True
    assert fingerprints_match(179246502392, 179246502492) is False
    assert fingerprints_match(179246502392, None) is False
    assert fingerprints_match(179246502392, "|179246502392") is False


@pytest.mark.parametrize(("recorded", "current"), [(CLI_RECORDED, GATEWAY_READ), (GATEWAY_RECORDED, CLI_READ)])
def test_crash_sweep_keeps_a_worker_read_a_second_off(board, monkeypatch, recorded, current):
    """The dispatcher path from #215: ``detect_crashed_workers`` must not reclaim the card."""
    tid = _running_card(board, pid=os.getpid(), started_at=recorded)
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: current)
    assert kbd.detect_crashed_workers(board) == []
    task = kb.get_task(board, tid)
    assert task.status == "running" and task.worker_pid == os.getpid()


def test_crash_sweep_still_reclaims_a_recycled_pid_without_signalling(board, monkeypatch):
    """Past the tolerance the live PID is a stranger: the card is reclaimed, and nothing on the
    crash path signals the PID (a reclaim of a dead worker never calls ``terminate_pid``)."""
    tid = _running_card(board, pid=os.getpid(), started_at=CLI_RECORDED)
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: f"|{179246502392 + START_TIME_DRIFT_TOLERANCE + 1}")
    assert kbd.detect_crashed_workers(board) == [tid]
    assert kb.get_task(board, tid).status == "ready"
    killed = []
    assert kbd._terminate_reclaimed_worker(os.getpid(), kb._claimer_id(), started_at=CLI_RECORDED,
                                           signal_fn=lambda pid, sig: killed.append((pid, sig)))["pid_recycled"]
    assert killed == []


def test_expired_claim_of_a_drifted_live_worker_is_extended(board, monkeypatch):
    tid = _running_card(board, pid=os.getpid(), started_at=GATEWAY_RECORDED)
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: CLI_READ)
    assert kb.release_stale_claims(board) == 0
    assert kb.get_task(board, tid).status == "running"
    assert "claim_extended" in [e.kind for e in kb.list_events(board, tid)]


_READER = (
    "import sys\n"
    "shift = float(sys.argv[3])\n"
    "if shift:\n"
    "    # What a long-lived process sees after kern.boottime moves: psutil adds the movement.\n"
    "    from psutil import _psosx\n"
    "    _psosx.INIT_BOOT_TIME = _psosx.boot_time() + shift\n"
    "from hermes_cli import kanban_db_dispatch as kbd\n"
    "pid, recorded = int(sys.argv[1]), sys.argv[2]\n"
    "print(kbd._process_fingerprint(pid), kbd._worker_alive(pid, recorded))\n"
)


def _read_in_another_process(pid: int, recorded: str, shift: float = 0.0) -> tuple[str, str]:
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    out = subprocess.run([sys.executable, "-c", _READER, str(pid), recorded, str(shift)], cwd=ROOT, env=env,
                         stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    read, alive = out.stdout.split()
    return read, alive


@pytest.fixture
def recorded_child(board):
    """A live child whose fingerprint this process records, as the spawning dispatcher does."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], stdin=subprocess.DEVNULL)
    try:
        tid = kb.create_task(board, title="job", assignee="worker")
        kb.claim_task(board, tid)
        kbd._set_worker_pid(board, tid, child.pid)
        recorded = board.execute("SELECT worker_started_at FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
        assert isinstance(recorded, str) and "|" in recorded
        yield child.pid, recorded
    finally:
        child.kill()
        child.wait(timeout=30)


def test_fingerprint_recorded_in_one_process_is_checked_in_another(recorded_child):
    """A separate interpreter reads the fingerprint back, as the next sweep does: the recorded value,
    copies whose start half is 1 s off (the size seen in #215), and a far value that is foreign."""
    pid, recorded = recorded_child
    epoch, _, start = recorded.rpartition("|")
    for stored in (recorded, f"{epoch}|{int(start) - 100}", f"{epoch}|{int(start) + 100}"):
        read, alive = _read_in_another_process(pid, stored)
        assert read.rpartition("|")[0] == epoch
        assert alive == "True", (stored, read)
    assert _read_in_another_process(pid, f"{epoch}|{int(start) + 360000}")[1] == "False"


@pytest.mark.skipif(sys.platform != "darwin", reason="psutil's kern.boottime adjustment is macOS-only")
@pytest.mark.parametrize("shift", [1.0, 2.0])
def test_reader_whose_psutil_saw_boottime_move_still_finds_the_worker(recorded_child, shift):
    """The #215 mechanism itself: the reading process's psutil adds the boot-time movement since
    it was imported, so its reading is ``shift`` seconds later than the recorded one."""
    pid, recorded = recorded_child
    read, alive = _read_in_another_process(pid, recorded, shift)
    assert int(read.rpartition("|")[2]) - int(recorded.rpartition("|")[2]) == round(shift * 100), (recorded, read)
    assert alive == "True", (recorded, read)
