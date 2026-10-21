"""A live worker stays alive to a dispatcher in another process, while a recycled PID stays foreign.

``worker_started_at`` is written by whichever process spawned the worker (a one-shot
``hermes kanban dispatch`` or the gateway's dispatcher) and read later by whichever process runs the
next sweep. On macOS psutil shifts ``create_time()`` by however far ``kern.boottime`` has moved since
psutil was imported, so a gateway and a fresh CLI read the same worker's start 1 s apart on 20 Oct
(``|179246502392`` recorded, ``|179246502492`` read) and 3 s apart on 21 Oct. The exact comparison
read that as a recycled PID, reclaimed the card and started a second worker beside the first
(aleph-vault #215). ``gateway.status.get_process_start_time`` now reads the kernel's unadjusted start
time on macOS, which every process reads alike, and the start half agrees within
``START_TIME_DRIFT_TOLERANCE`` for rows recorded before that. A row recorded by older code, with the
writer's adjustment baked in, also matches the reader's public (adjusted) reading, so a mixed-version
window can't reclaim a worker the older code would have kept. The epoch half, a malformed reading and an
integer (pre-epoch) fingerprint stay exact.
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
from hermes_cli.kanban_worker_identity import recorded_fingerprint_matches

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


_SHIFT = (
    "import sys\n"
    "shift = float(sys.argv[3])\n"
    "if shift:\n"
    "    # What a long-lived process sees after kern.boottime moves: psutil adds the movement.\n"
    "    from psutil import _psosx\n"
    "    _psosx.INIT_BOOT_TIME = _psosx.boot_time() + shift\n"
)
_READER = _SHIFT + (
    "from hermes_cli import kanban_db_dispatch as kbd\n"
    "pid, recorded = int(sys.argv[1]), sys.argv[2]\n"
    "print(kbd._process_fingerprint(pid), kbd._worker_alive(pid, recorded))\n"
)
# The fingerprint an unpatched process (base e23f76de) wrote on macOS: psutil's public create_time(),
# adjusted by however far kern.boottime had moved since that process imported psutil.
_OLD_WRITER = _SHIFT + (
    "import psutil\n"
    "from gateway.drain_control import current_instantiation_epoch\n"
    "pid = int(sys.argv[1])\n"
    "print(f'{current_instantiation_epoch()}|{int(round(psutil.Process(pid).create_time() * 100))}')\n"
)


def _run(source: str, pid: int, recorded: str, shift: float) -> str:
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    out = subprocess.run([sys.executable, "-c", source, str(pid), recorded, str(shift)], cwd=ROOT, env=env,
                         stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def _read_in_another_process(pid: int, recorded: str, shift: float = 0.0) -> tuple[str, str]:
    read, alive = _run(_READER, pid, recorded, shift).split()
    return read, alive


def _record_with_old_code(pid: int, shift: float) -> str:
    return _run(_OLD_WRITER, pid, "", shift).strip()


@pytest.fixture
def recorded_child(board):
    """A live child whose fingerprint this process records, as the spawning dispatcher does. It
    waits on stdin, so it outlives every reader and ends when the fixture closes its stdin."""
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
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
        if child.stdin is not None:
            child.stdin.close()


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


@pytest.mark.platforms("macos")
@pytest.mark.parametrize("shift", [1.0, 3.0, -3.0, 600.0])
def test_reader_whose_psutil_saw_boottime_move_reads_the_same_start(recorded_child, shift):
    """The #215 mechanism itself: the reading process imported psutil before ``kern.boottime`` moved,
    so psutil's ``create_time()`` there is off by ``|shift|`` (3 s on 21 Oct, past the 2 s tolerance).
    The fingerprint must still read exactly what the spawning process recorded."""
    pid, recorded = recorded_child
    read, alive = _read_in_another_process(pid, recorded, shift)
    assert read == recorded
    assert alive == "True", (recorded, read)


@pytest.mark.platforms("macos")
@pytest.mark.parametrize("reader_shift", [3.0, 1.0])
def test_row_an_old_gateway_recorded_during_the_step_stays_alive_to_a_patched_reader(recorded_child, reader_shift):
    """Codex's reproduction against 25afef39: the row was written by unpatched code whose psutil saw
    kern.boottime move 3 s (``|S+300``), and the reader is patched code whose psutil sees 3 s or 1 s.
    Older code kept this worker alive; the unadjusted reading alone (S) would reclaim its card."""
    pid, _ = recorded_child
    recorded = _record_with_old_code(pid, 3.0)
    epoch, _, start = recorded.rpartition("|")
    read, alive = _read_in_another_process(pid, recorded, reader_shift)
    assert int(start) - int(read.rpartition("|")[2]) == 300  # the raw reading alone would say "recycled"
    assert alive == "True", (recorded, read)


@pytest.mark.platforms("macos")
def test_old_row_out_of_reach_of_both_readings_is_judged_as_older_code_judged_it(recorded_child):
    """The compatibility rule only adds what older code accepted. A +3 s row read by a process whose
    psutil saw no move is 3 s from both readings, as it was for older code (runs 180/182): no looser."""
    pid, _ = recorded_child
    recorded = _record_with_old_code(pid, 3.0)
    assert _read_in_another_process(pid, recorded, 0.0)[1] == "False"


def _shift_boottime(monkeypatch, seconds):
    """kern.boottime moved ``seconds`` earlier after this process imported psutil (21 Oct: 3 s)."""
    from psutil import _psosx

    real = _psosx.boot_time()
    monkeypatch.setattr(_psosx, "INIT_BOOT_TIME", real)
    monkeypatch.setattr(_psosx, "boot_time", lambda: real - seconds)


@pytest.mark.platforms("macos")
def test_215_cli_recorded_worker_survives_a_gateway_sweep_during_the_step(monkeypatch):
    """Run 179: a fresh CLI recorded the worker, then the gateway swept while boot time sat 3 s off."""
    recorded = kbd._process_fingerprint(os.getpid())
    _shift_boottime(monkeypatch, 3)
    assert kbd._worker_alive(os.getpid(), recorded) is True


@pytest.mark.platforms("macos")
def test_215_worker_recorded_during_the_step_survives_the_step_back(monkeypatch):
    """Runs 180/182: the gateway recorded during the step and swept after boot time stepped back."""
    with monkeypatch.context() as m:
        _shift_boottime(m, 3)
        recorded = kbd._process_fingerprint(os.getpid())
    assert kbd._worker_alive(os.getpid(), recorded) is True


@pytest.mark.platforms("macos")
def test_215_old_gateway_row_survives_a_patched_sweep_during_the_step(monkeypatch):
    """Run 181's shape across the upgrade: the old gateway wrote ``|S+300`` during the step, and the
    patched one sweeps while its psutil still sees the step."""
    from gateway.drain_control import current_instantiation_epoch
    from gateway.status import get_process_start_time

    raw = get_process_start_time(os.getpid())
    _shift_boottime(monkeypatch, 3)
    recorded = f"{current_instantiation_epoch()}|{raw + 300}"
    assert fingerprints_match(recorded, kbd._process_fingerprint(os.getpid())) is False
    assert kbd._worker_alive(os.getpid(), recorded) is True


@pytest.mark.platforms("macos")
@pytest.mark.parametrize("offset", [300 + START_TIME_DRIFT_TOLERANCE + 1, -START_TIME_DRIFT_TOLERANCE - 1, -360000])
def test_recycled_pid_is_still_recycled_when_either_reading_is_tried(monkeypatch, offset):
    """A PID whose start is past the tolerance of both the raw and the adjusted (+3 s) reading is a
    stranger: not alive, and the reclaim path reports it recycled without signalling it."""
    from gateway.drain_control import current_instantiation_epoch
    from gateway.status import get_process_start_time

    raw = get_process_start_time(os.getpid())
    _shift_boottime(monkeypatch, 3)
    recorded = f"{current_instantiation_epoch()}|{raw + offset}"
    assert kbd._worker_alive(os.getpid(), recorded) is False
    assert kbd._pid_recycled(os.getpid(), recorded) is True


@pytest.mark.platforms("macos")
def test_start_time_is_step_invariant_while_the_public_reading_moves(monkeypatch):
    from gateway.status import get_process_start_time
    from hermes_cli.kanban_worker_identity import public_start_time

    pid = os.getpid()
    before = get_process_start_time(pid)
    assert public_start_time(pid) == before
    _shift_boottime(monkeypatch, 3)
    assert get_process_start_time(pid) == before
    assert public_start_time(pid) == before + 300


class _NoKernelValue:
    """A psutil.Process stand-in whose private ``_proc`` is missing or predates ``monotonic``."""

    def __init__(self, real, proc_attr):
        self._real = real
        if proc_attr is not None:
            self._proc = proc_attr

    def create_time(self):
        return self._real.create_time()


class _OldProc:
    def create_time(self):  # no ``monotonic`` keyword
        return 0.0


@pytest.mark.platforms("macos")
@pytest.mark.parametrize("proc_attr", [None, _OldProc()], ids=["no-_proc", "no-monotonic-keyword"])
def test_start_time_falls_back_to_the_public_reading(monkeypatch, proc_attr):
    """If a later psutil drops ``_proc`` or its keyword, the reading is the public one, as before."""
    import psutil

    from gateway import status

    real_process = psutil.Process
    pid = os.getpid()
    public = int(round(real_process(pid).create_time() * 100))
    monkeypatch.setattr(psutil, "Process", lambda p: _NoKernelValue(real_process(p), proc_attr))
    assert status.get_process_start_time(pid) == public


@pytest.mark.platforms("not macos")
def test_off_macos_there_is_no_compatibility_reading():
    """Off macOS the start time never changed, so only the ordinary reading is compared."""
    from hermes_cli.kanban_worker_identity import public_start_time

    assert public_start_time(os.getpid()) is None


@pytest.mark.parametrize(("current", "public", "same"), [
    ("e|179246502392", None, True),                  # raw reading matches
    ("e|179246502092", "e|179246502392", True),      # only the adjusted reading matches (old row)
    ("e|179246502092", "e|179246502092", False),     # neither: off by 3 s
    (None, "e|179246502392", False),                 # unreadable now: the public reading can't rescue it
    ("e|179246502092", "other|179246502392", False), # the epoch still has to agree
    ("e|179246502092", None, False),                 # off macOS / no public reading
])
def test_recorded_fingerprint_matches_either_reading(current, public, same):
    assert recorded_fingerprint_matches("e|179246502392", current, public) is same


def test_integer_row_keeps_exact_comparison_against_either_reading():
    assert recorded_fingerprint_matches(179246502392, 179246502092, 179246502392) is True
    assert recorded_fingerprint_matches(179246502392, 179246502092, 179246502492) is False
