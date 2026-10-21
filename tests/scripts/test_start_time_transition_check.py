"""scripts/start_time_transition_check.py: the drained-transition gate for aleph-vault #215.

Every test runs on a temporary HERMES_HOME and lock dir and asserts that neither is the OS user's
real ``~/.hermes``; the script must never be pointed at the live home from a test.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pwd
import select
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "start_time_transition_check.py"
RAW = 179248758869  # a start time in centiseconds
SHIFT = 300  # psutil's adjustment on 21 Oct: 3 s


def _load():
    spec = importlib.util.spec_from_file_location("start_time_transition_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


check = _load()


def _live_home() -> Path:
    return Path(pwd.getpwuid(os.getuid()).pw_dir) / ".hermes"


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    (home / "cron").mkdir(parents=True)
    locks = tmp_path / "locks"
    locks.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_REAL_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(locks))
    live = _live_home().resolve()
    for path in (home, locks):
        assert live != path.resolve() and live not in path.resolve().parents, "test reached the live home"
    from hermes_constants import get_hermes_home

    assert get_hermes_home().resolve() == home.resolve()
    yield home
    assert get_hermes_home().resolve() == home.resolve()


def _hooks(raw: dict[int, int], alive=None):
    return {
        "alive": lambda pid: pid in raw if alive is None else alive(pid),
        "raw": lambda pid: raw.get(pid),
        "kanban": lambda pid: None if raw.get(pid) is None else f"|{raw[pid]}",
    }


def _write_records(home: Path, locks: Path, start: int, pid: int = 4242) -> None:
    record = {"pid": pid, "kind": "hermes-gateway", "start_time": start}
    (home / "gateway.pid").write_text(json.dumps(record))
    (home / "gateway_state.json").write_text(json.dumps(
        {**record, "platforms": {"telegram": {"writer_pid": pid, "writer_start_time": start}}}))
    (locks / "telegram-bot-token-0.lock").write_text(json.dumps(record))
    (locks / "host-gateway.json").write_text(json.dumps({"pid": pid, "startTime": start, "role": "gateway"}))
    state = sqlite3.connect(home / "state.db")
    state.executescript(
        "CREATE TABLE delivery_obligations (obligation_id TEXT, state TEXT, owner_pid INTEGER, owner_started_at INTEGER);"
        "CREATE TABLE async_delegations (delegation_id TEXT, state TEXT, owner_pid INTEGER, owner_started_at INTEGER);")
    state.execute("INSERT INTO delivery_obligations VALUES ('ob', 'pending', ?, ?)", (pid, start))
    state.execute("INSERT INTO async_delegations VALUES ('dl', 'running', ?, ?)", (pid, start))
    state.commit()
    state.close()
    cron = sqlite3.connect(home / "cron" / "executions.db")
    cron.execute("CREATE TABLE executions (id TEXT, status TEXT, pid INTEGER, process_started_at INTEGER)")
    cron.execute("INSERT INTO executions VALUES ('ex', 'running', ?, ?)", (pid, start))
    cron.commit()
    cron.close()
    board = home / "kanban" / "boards" / "b"
    board.mkdir(parents=True)
    kanban = sqlite3.connect(board / "kanban.db")
    kanban.execute("CREATE TABLE tasks (id TEXT, worker_pid INTEGER, worker_started_at TEXT)")
    kanban.execute("CREATE TABLE task_runs (id INTEGER, worker_pid INTEGER, worker_started_at TEXT)")
    kanban.execute("INSERT INTO tasks VALUES ('t', ?, ?)", (pid, f"|{start}"))
    kanban.execute("INSERT INTO task_runs VALUES (1, ?, ?)", (pid, f"|{start}"))
    kanban.commit()
    kanban.close()


def _snapshot(*roots: Path) -> dict[str, str]:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() + f":{p.stat().st_mtime_ns}"
            for root in roots for p in sorted(root.rglob("*")) if p.is_file()}


FAMILIES = ("gateway.pid", "gateway_state.json:start_time", "writer_start_time", "delivery_obligations",
            "async_delegations", "executions", "tasks/t", "task_runs/1", ".lock", "host-gateway.json")


def test_drained_home_passes_and_is_left_untouched(home, tmp_path):
    locks = tmp_path / "locks"
    _write_records(home, locks, RAW)
    before = _snapshot(home, locks)
    lines: list[str] = []
    assert check.run(home, locks, out=lines.append, **_hooks({4242: RAW})) == 0
    assert _snapshot(home, locks) == before  # no write, no -wal/-shm sidecar
    assert len([line for line in lines if line.strip().startswith("OK")]) == len(FAMILIES)
    assert lines[-1].startswith("DRAINED")


def test_record_carrying_the_psutil_adjustment_fails_and_is_listed(home, tmp_path):
    """An old-code owner recorded +3 s during a clock step; the new reader sees the raw start."""
    locks = tmp_path / "locks"
    _write_records(home, locks, RAW + SHIFT)
    lines: list[str] = []
    assert check.run(home, locks, out=lines.append, **_hooks({4242: RAW})) == 1
    differs = [line for line in lines if line.strip().startswith("DIFFERS")]
    for family in FAMILIES:
        assert any(family in line and "diff +300 cs" in line for line in differs), family
    assert lines[-1].startswith("NOT DRAINED")


@pytest.mark.parametrize(("offset", "exact_ok", "tolerant_ok"), [(1, False, True), (200, False, True),
                                                                  (201, False, False)])
def test_each_family_uses_its_readers_comparator(home, tmp_path, offset, exact_ok, tolerant_ok):
    locks = tmp_path / "locks"
    _write_records(home, locks, RAW + offset)
    verdicts = [check.judge(r, **_hooks({4242: RAW})) for r in check.collect(home, locks)]
    for v in verdicts:
        expected = exact_ok if v.record.rule == check.EXACT else tolerant_ok
        assert (v.status == "ok") is expected, (v.record.source, v.status)


def test_dead_pids_are_skipped_and_unreadable_live_ones_fail(home, tmp_path):
    locks = tmp_path / "locks"
    _write_records(home, locks, RAW + SHIFT)
    lines: list[str] = []
    assert check.run(home, locks, out=lines.append, **_hooks({})) == 0  # every owner has exited
    assert "0 with a live pid" in lines[0]
    hooks = _hooks({}, alive=lambda pid: True)  # alive, but its start time can't be read
    assert check.run(home, locks, out=lines.append, **hooks) == 1
    assert any(line.strip().startswith("UNREADABLE") for line in lines)


def test_main_takes_the_home_as_an_argument(home, tmp_path, capsys):
    assert check.main(["--home", str(home), "--lock-dir", str(tmp_path / "locks")]) == 0
    assert f"home {home};" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        check.main(["--home", str(tmp_path / "missing")])


# An old-code owner in its own process: psutil's boot time shifted +3 s, and base code's macOS
# reading (the public create_time()) restored, then the real writers record their owner stamps.
_OWNER = """
import json, os, sys, time
import psutil
from psutil import _psosx
from gateway import status
if int(sys.argv[1]):
    _psosx.INIT_BOOT_TIME = _psosx.boot_time() + int(sys.argv[1])
    status._get_process_start_time = lambda pid: int(round(psutil.Process(pid).create_time() * 100))
status.write_pid_file()
assert status.acquire_scoped_lock("review", "transition-check")[0]
from gateway import delivery_ledger as dl
dl.record_obligation(obligation_id="ob-1", session_key="s", platform="slack", chat_id="c", thread_id=None, content="x")
from cron import executions
executions.mark_execution_running(executions.create_execution("job", source="builtin")["id"])
from tools import async_delegation as ad
ad._persist_dispatch({"delegation_id": "deleg-1", "dispatched_at": time.time(), "goal": "x"})
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_dispatch as kbd
from pathlib import Path
conn = kbc.connect(Path(os.environ["HERMES_HOME"]) / "kanban.db")
tid = kb.create_task(conn, title="job", assignee="worker")
kb.claim_task(conn, tid)
kbd._set_worker_pid(conn, tid, os.getpid())
conn.close()
from hermes_constants import get_hermes_home
print(json.dumps({"pid": os.getpid(), "home": str(get_hermes_home())}), flush=True)
sys.stdin.read()
"""


@pytest.mark.platforms("macos")
@pytest.mark.parametrize(("shift", "expected"), [(0, 0), (3, 1)])
def test_live_owner_written_by_old_code_during_a_step_blocks_the_transition(home, tmp_path, shift, expected):
    locks = tmp_path / "locks"
    env = {**os.environ, "PYTHONPATH": str(REPO), "HERMES_HOME": str(home), "HERMES_REAL_HOME": str(tmp_path),
           "HOME": str(tmp_path), "HERMES_GATEWAY_LOCK_DIR": str(locks)}
    owner = subprocess.Popen([sys.executable, "-c", _OWNER, str(shift)], cwd=REPO, env=env, text=True,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert select.select([owner.stdout], [], [], 60)[0], "owner timed out"
        line = owner.stdout.readline()
        assert line, owner.stderr.read()
        info = json.loads(line)
        assert Path(info["home"]).resolve() == home.resolve()
        lines: list[str] = []
        before = _snapshot(home, locks)
        assert check.run(home, locks, out=lines.append) == expected, "\n".join(lines)
        assert _snapshot(home, locks) == before
        live = [line for line in lines if f"pid {info['pid']:>7}" in line]
        for family in ("gateway.pid", ".lock", "delivery_obligations", "async_delegations", "executions",
                       "tasks/", "task_runs/"):
            assert any(family in line for line in live), family
        assert all(line.strip().startswith("DIFFERS" if shift else "OK") for line in live), "\n".join(live)
        assert owner.poll() is None
    finally:
        owner.stdin.close()
        owner.wait(timeout=30)
        owner.stdout.close()
        owner.stderr.close()
