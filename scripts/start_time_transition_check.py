#!/usr/bin/env python3
"""Install-time check for the switch to the unadjusted macOS start time (aleph-vault #215).

``gateway.status.get_process_start_time`` now reads the kernel's stored start time on macOS
(``proc._proc.create_time(monotonic=True)``) instead of psutil's public ``create_time()``, which
psutil 7.2.2 shifts by ``|boot time at psutil import - boot time now|`` once they differ by 1 s or
more. A record written by OLD code while such a step was active carries that shift, and NEW code
rejects it: Kanban keeps both readings (``kanban_worker_identity.recorded_fingerprint_matches``),
but nothing else does. Only an old-code process can write such a record, so once every Hermes
process runs the new code no new one can appear. This script proves the transition is drained:
every persisted start-time record whose PID is still alive must match that PID's unadjusted
reading under the comparator its new-code reader uses.

    python scripts/start_time_transition_check.py --home ~/.hermes [--lock-dir DIR] [--no-profiles]

Exit 0: every live record matches (dead PIDs are skipped). Exit 1: at least one live record
differs, can't be read, or sits in a file that can't be parsed; each is listed. Exit 2: bad arguments.
``--restarting PID,...`` names the processes the install restarts together (the gateway and the
dashboards): a shifted record they own is listed as RESTART instead of failing, because the
restart ends that owner and every new reader then treats it as gone. Run once before the restart
with ``--restarting``, and again after it without.
The report ends with the live owner pids: each must run the new code or be gone once the
restart is done, because an old-code owner that survives it can stamp a shifted record later.

One comparator for every family: the recorded start must EQUAL the pid's unadjusted reading (a
composed Kanban fingerprint also needs the same epoch). psutil's adjustment is whole seconds, a
new-code writer records the unadjusted value, and an old-code writer with no step active records
the same value, so any difference means a shifted record. That is stricter than every reader
(several tolerate 2 s), so DRAINED holds for all of them, including the exact ones.

Read-only: JSON and text are only read; SQLite opens with ``immutable=1`` when there is no
``-wal``, ``mode=ro`` when both ``-wal`` and ``-shm`` exist (no file is created and no database
or WAL byte changes; like every WAL reader it takes read marks in the shared ``-shm`` index), and
otherwise (a ``-wal`` without ``-shm``, where ``mode=ro`` would create
the ``-shm``) from a private copy of the database and its WAL in a temporary directory. Nothing
under ``--home`` or ``--lock-dir`` is written, bytecode writing is off, and HERMES_HOME is never
resolved: every path comes from ``--home`` and ``--lock-dir`` (the lock-dir default reads
``HERMES_GATEWAY_LOCK_DIR`` / ``XDG_STATE_HOME`` like gateway/status.py:324).

Records checked (writer -> new-code readers):
- Kanban ``tasks`` and ``task_runs`` ``worker_pid``/``worker_started_at``, in ``<home>/kanban.db``
  and ``<home>/kanban/boards/*/kanban.db``: hermes_cli/kanban_db_dispatch.py:1477 and :1496 ->
  :402 ``_pid_recycled`` and the :519 reaper. Boards pinned elsewhere (``HERMES_KANBAN_HOME``,
  ``HERMES_KANBAN_DB``) need ``--home`` pointed at them too.
- ``<home>/gateway.pid`` and ``gateway.lock`` ``start_time``: gateway/status.py:776
  ``_build_pid_record`` -> :875 ``_live_pid_from_record`` and :2069
  ``get_running_pid_identity_strict``; gateway/run.py:4988 ``--replace``.
- ``<home>/gateway_state.json`` ``start_time`` and ``platforms.*.writer_start_time``:
  gateway/status.py:1138/:1168 -> :1242 ``runtime_status_pid_is_live`` and
  hermes_cli/web_server_gateway.py:158.
- Delivery ledger, ``<home>/state.db`` ``delivery_obligations`` (pending/attempting/failed):
  gateway/delivery_ledger.py:173, :204, :275 -> :232 ``_owner_alive`` via the :398 sweep, and the
  exact :479 retry claim.
- Async delegation, ``<home>/state.db`` ``async_delegations``, running/stalling/finalizing and
  any row still awaiting delivery: tools/async_delegation.py:97, :157 -> :254 ``alive`` from :271
  (recovery) and :383-392 (orphaned completions).
- Cron executions, ``<home>/cron/executions.db`` (claimed/running): cron/executions.py:51, :66,
  :126, :275 -> :134 ``_owner_is_live`` from :375, and the exact
  hermes_cli/data_cleanup_holders.py:98-101 cleanup guard.
- Cron delivery queue, ``<home>/cron/deliveries.db`` ``deliveries`` (delivering):
  cron/delivery_queue.py:82, :118, :231-242 -> :296 ``recover_abandoned``.
- API-server durable runs, ``<home>/runs_idempotency.db`` ``run_idempotency`` (non-terminal):
  gateway/platforms/api_server_runs.py:217, :723 and api_server_run_idempotency.py:69, :148 ->
  api_server_runs.py:384 ``_owner_alive`` from :409.
- WhatsApp bridge ``bridge.pid`` (line 2) in ``<home>/platforms/whatsapp/session`` or the legacy
  ``<home>/whatsapp/session``: plugins/platforms/whatsapp/adapter.py:60, :175 -> :115
  ``_bridge_pid_is_ours`` (a custom ``session_path`` needs its own run).
- Scoped locks and host rendezvous in the lock dir (``HERMES_GATEWAY_LOCK_DIR``, else
  ``$XDG_STATE_HOME`` or ``~/.local/state`` ``/hermes/gateway-locks``; gateway/status.py:324):
  ``*.lock`` from :1559 ``acquire_scoped_lock`` -> :1529 and :1801; ``host-*.json`` ``startTime``
  from gateway/host_rendezvous.py:487 -> :239.
- Process registry ``<home>/processes.json`` ``host_start_time``: tools/process_registry.py:44,
  :552 -> :995 and :1004.
- Shared-metrics markers ``<home>/telemetry/shared_metrics/{process_markers,
  provider_setup_markers}/*.json``: hermes_cli/observability/shared_metrics_process.py:114 and
  shared_metrics_setup.py:163 -> gateway/status.py:1242.
- Lightpanda ``<home>/cache/browser-use/lightpanda/*.json``: tools/browser_lightpanda.py:189 ->
  :295.
- Local runtime ``<home>/runtimes/llamacpp/server.json`` ``start_time``/``owner_start_time``:
  hermes_cli/local_runtime/supervisor.py:233 and :236 -> recovery.py:40.
- Stop and takeover markers ``<home>/.gateway-planned-stop.json``, ``.gateway-takeover.json``
  and ``.hermes-update-paused-gateways.*.json`` (``target_start_time``, ``ct``):
  gateway/status.py:1952, :1687 and hermes_cli/update_pause_record.py:828,
  update_cmd_posix_pause.py:392.
Not checked, because their readers never used ``get_process_start_time`` and this change doesn't
touch them: hermes_cli/active_sessions.py:301, hermes_cli/process_identity.py:64 and
gateway/lifecycle_ledger.py:110 (public ``create_time()`` floats); in-memory baselines (MCP,
terminal pgid, browser daemon, ``--replace`` and profile kill guards), which end with their
process. Every ``<home>/profiles/*`` home is checked the same way unless ``--no-profiles``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

sys.dont_write_bytecode = True  # importing the checkout's modules must not write __pycache__ into it
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXACT, KANBAN = "exact", "kanban"
_TERMINAL_RUN_STATUSES = ("completed", "failed", "cancelled", "interrupted")  # api_server_run_idempotency.py:19
# JSON (pid key, start key) pairs persisted by the writers above; the generic walk finds them at
# any depth (``gateway_state.json``'s per-platform writer stamps, update-pause entries).
_JSON_PAIRS = (("pid", "start_time"), ("pid", "startTime"), ("pid", "ct"), ("pid", "host_start_time"),
               ("owner_pid", "owner_start_time"), ("target_pid", "target_start_time"),
               ("writer_pid", "writer_start_time"))


@dataclass(frozen=True)
class Record:
    source: str  # file and row/key, for the report
    pid: int
    recorded: Any
    rule: str  # EXACT | KANBAN (composed fingerprint: same epoch too)


@dataclass(frozen=True)
class Verdict:
    record: Record
    status: str  # "ok" | "differs" | "unreadable" | "dead" | "unparsed"
    current: Any = None
    diff_cs: Optional[int] = None


# ── reading, strictly read-only ────────────────────────────────────────────────────────────────

def _connect(uri: str) -> sqlite3.Connection:
    conn = sqlite3.connect(uri, uri=True, timeout=5)
    conn.execute("PRAGMA query_only = 1")
    return conn


def _rows(path: Path, sql: str) -> list[tuple]:
    """``sql`` over ``path`` without creating or changing any file beside it (see the module doc)."""
    if not path.is_file():
        return []
    wal, shm = (path.with_name(path.name + suffix) for suffix in ("-wal", "-shm"))
    with tempfile.TemporaryDirectory(prefix="start-time-check-") as scratch:
        if not wal.exists():
            uri = f"file:{path}?immutable=1"
        elif shm.exists():
            uri = f"file:{path}?mode=ro"
        else:
            copy = Path(scratch) / path.name
            shutil.copyfile(path, copy)
            shutil.copyfile(wal, copy.with_name(copy.name + "-wal"))
            uri = f"file:{copy}"
        conn = _connect(uri)
        try:
            return conn.execute(sql).fetchall()
        except sqlite3.OperationalError as exc:  # table or column absent in this schema version
            if "no such" in str(exc):
                return []
            raise
        finally:
            conn.close()


class _Unparsed(Exception):
    """A record file that exists and isn't empty but can't be parsed: fail closed."""


def _read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    if not text.strip():
        return None  # flock files (gateway-locks/host-*.lock) are empty by design
    try:
        return json.loads(text)
    except ValueError as exc:
        raise _Unparsed(str(path)) from exc


def _as_pid(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _walk_json(source: str, data: Any, rule: str) -> Iterator[Record]:
    if isinstance(data, list):
        for i, item in enumerate(data):
            yield from _walk_json(f"{source}[{i}]", item, rule)
        return
    if not isinstance(data, dict):
        return
    if data.get("pid_scope") not in (None, "host"):
        return  # a sandbox/container pid (tools/process_registry.py:552) isn't a host pid
    for pid_key, start_key in _JSON_PAIRS:
        pid = _as_pid(data.get(pid_key))
        if pid is not None and data.get(start_key) is not None:
            yield Record(f"{source}:{start_key}", pid, data[start_key], rule)
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            yield from _walk_json(f"{source}.{key}", value, rule)


def _json_records(paths: list[Path], rule: str = EXACT) -> Iterator[Record]:
    for path in paths:
        if path.is_file():
            try:
                data = _read_json(path)
            except _Unparsed:
                yield Record(f"{path} (unparseable)", 0, None, "unparsed")
                continue
            yield from _walk_json(str(path), data, rule)


# ── record families ───────────────────────────────────────────────────────────────────────────

def kanban_records(home: Path) -> Iterator[Record]:
    boards = [home / "kanban.db", *sorted((home / "kanban" / "boards").glob("*/kanban.db"))]
    for db in boards:
        for table in ("tasks", "task_runs"):
            sql = (f"SELECT id, worker_pid, worker_started_at FROM {table} "
                   "WHERE worker_pid IS NOT NULL AND worker_started_at IS NOT NULL")
            for row_id, pid, started in _rows(db, sql):
                if _as_pid(pid) is not None and started != "unverified":
                    yield Record(f"{db}:{table}/{row_id}", int(pid), started, KANBAN)


def _sql_records(db: Path, table: str, sql: str) -> Iterator[Record]:
    for row_id, pid, started in _rows(db, sql):
        if _as_pid(pid) is not None and started:
            yield Record(f"{db}:{table}/{row_id}", int(pid), started, EXACT)


def ledger_records(home: Path) -> Iterator[Record]:
    state_db = home / "state.db"
    yield from _sql_records(state_db, "delivery_obligations", (
        "SELECT obligation_id, owner_pid, owner_started_at FROM delivery_obligations "
        "WHERE state IN ('pending', 'attempting', 'failed')"))
    yield from _sql_records(state_db, "async_delegations", (
        "SELECT delegation_id, owner_pid, owner_started_at FROM async_delegations "
        "WHERE state IN ('running', 'stalling', 'finalizing') OR delivery_state = 'pending'"))
    yield from _sql_records(home / "cron" / "executions.db", "executions", (
        "SELECT id, pid, process_started_at FROM executions WHERE status IN ('claimed', 'running')"))
    yield from _sql_records(home / "cron" / "deliveries.db", "deliveries", (
        "SELECT execution_id, owner_pid, owner_started_at FROM deliveries WHERE status = 'delivering'"))
    runs = home / "runs_idempotency.db"
    for scope, run_id, status_json, pid, started in _rows(runs, (
            "SELECT scope, run_id, status_json, owner_pid, owner_started FROM run_idempotency")):
        try:
            status = (json.loads(status_json) or {}).get("status")
        except (TypeError, ValueError, AttributeError):
            status = None  # unreadable status: the reader treats it as non-terminal
        if status not in _TERMINAL_RUN_STATUSES and _as_pid(pid) is not None and started:
            yield Record(f"{runs}:run_idempotency/{scope}/{run_id}", int(pid), started, EXACT)


def whatsapp_records(home: Path) -> Iterator[Record]:
    for session in (home / "platforms" / "whatsapp" / "session", home / "whatsapp" / "session"):
        path = session / "bridge.pid"
        try:
            lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").split("\n")]
        except FileNotFoundError:
            continue
        pid = _as_pid(lines[0]) if lines else None
        start = lines[1] if len(lines) > 1 and lines[1] else None
        if pid is not None and start is not None:
            yield Record(f"{path}:line 2", pid, start, EXACT)


def home_file_records(home: Path) -> Iterator[Record]:
    exact = [home / "gateway.pid", home / "gateway.lock", home / "gateway_state.json",
             home / "processes.json", home / ".gateway-planned-stop.json", home / ".gateway-takeover.json",
             *sorted(home.glob(".hermes-update-paused-gateways.*.json"))]
    for sub in ("process_markers", "provider_setup_markers"):
        exact += sorted((home / "telemetry" / "shared_metrics" / sub).glob("*.json*"))
    exact += sorted((home / "cache" / "browser-use" / "lightpanda").glob("*.json"))
    exact.append(home / "runtimes" / "llamacpp" / "server.json")
    yield from _json_records(exact)
    yield from whatsapp_records(home)


def lock_dir_records(lock_dir: Path) -> Iterator[Record]:
    yield from _json_records(sorted(lock_dir.glob("*.lock")) + sorted(lock_dir.glob("host-*.json")))


def default_lock_dir() -> Path:
    """gateway/status.py:324 ``_get_lock_dir``, without importing it (it would read this process's env
    for nothing else)."""
    override = os.environ.get("HERMES_GATEWAY_LOCK_DIR")
    if override:
        return Path(override)
    state_home = os.environ.get("XDG_STATE_HOME") or ""
    base = Path(state_home) if os.path.isabs(state_home) else Path.home() / ".local" / "state"
    return base / "hermes" / "gateway-locks"


def collect(home: Path, lock_dir: Optional[Path], profiles: bool = True) -> list[Record]:
    homes = [home]
    if profiles:
        homes += sorted(p for p in (home / "profiles").glob("*") if p.is_dir())
    records: list[Record] = []
    for one in homes:
        records += [*home_file_records(one), *ledger_records(one)]
    records += list(kanban_records(home))
    if lock_dir is not None:
        records += list(lock_dir_records(lock_dir))
    return records


# ── judging ───────────────────────────────────────────────────────────────────────────────────

def _pid_alive(pid: int) -> bool:
    import psutil

    return psutil.pid_exists(pid)


def _raw_start(pid: int) -> Optional[int]:
    from gateway.status import get_process_start_time

    return get_process_start_time(pid)


def _kanban_current(pid: int) -> Optional[str]:
    """hermes_cli/kanban_db_dispatch.py:372 ``_process_fingerprint``, composed here so the check
    doesn't import the Kanban modules (they resolve the ambient HERMES_HOME at import)."""
    from gateway.drain_control import current_instantiation_epoch

    start = _raw_start(pid)
    return None if start is None else f"{current_instantiation_epoch()}|{start}"


def _start_half(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else None
    try:
        return int(str(value).rpartition("|")[2])
    except ValueError:
        return None


def _matches(record: Record, current: Any) -> bool:
    """Exact: the start halves are equal integers and, for a composed fingerprint, so are the epochs."""
    if record.rule == KANBAN and isinstance(record.recorded, str) and "|" in str(record.recorded):
        if str(record.recorded).rpartition("|")[0] != str(current).rpartition("|")[0]:
            return False
    recorded_half, current_half = _start_half(record.recorded), _start_half(current)
    return recorded_half is not None and recorded_half == current_half


def judge(record: Record, alive: Callable[[int], bool] = _pid_alive,
          raw: Callable[[int], Optional[int]] = _raw_start,
          kanban: Callable[[int], Optional[str]] = _kanban_current) -> Verdict:
    if record.rule == "unparsed":
        return Verdict(record, "unparsed")
    if not alive(record.pid):
        return Verdict(record, "dead")
    composed = record.rule == KANBAN and isinstance(record.recorded, str) and "|" in record.recorded
    current = kanban(record.pid) if composed else raw(record.pid)
    if current is None:
        return Verdict(record, "unreadable")
    recorded_half, current_half = _start_half(record.recorded), _start_half(current)
    diff = None if recorded_half is None or current_half is None else recorded_half - current_half
    return Verdict(record, "ok" if _matches(record, current) else "differs", current, diff)


def run(home: Path, lock_dir: Optional[Path], profiles: bool = True,
        out: Callable[[str], None] = print, restarting: frozenset[int] = frozenset(), **judge_hooks: Any) -> int:
    records = collect(home, lock_dir, profiles)
    verdicts = [judge(r, **judge_hooks) for r in records]
    verdicts = [Verdict(v.record, "restart", v.current, v.diff_cs)
                if v.status in ("differs", "unreadable") and v.record.pid in restarting else v for v in verdicts]
    live = [v for v in verdicts if v.status != "dead"]
    bad = [v for v in live if v.status not in ("ok", "restart")]
    out(f"home {home}; lock dir {lock_dir}; {len(records)} start-time records, "
        f"{len(live)} with a live pid or unparseable, {len(records) - len(live)} dead (skipped)")
    for v in live:
        diff = "" if v.diff_cs is None else f" diff {v.diff_cs:+d} cs"
        out(f"  {v.status.upper():10} pid {v.record.pid:>7} {v.record.rule:6} recorded {v.record.recorded} "
            f"now {v.current}{diff}  {v.record.source}")
    owners = sorted({v.record.pid for v in live if v.record.pid})
    out(f"live owners: {', '.join(map(str, owners)) or 'none'} (each must run the new code or be gone "
        "after the restart)")
    if bad:
        out(f"NOT DRAINED: {len(bad)} live record(s) differ from the unadjusted reading or can't be read. "
            "Don't restart onto the new code yet; let those owners exit or finish, then rerun.")
        return 1
    if any(v.status == "restart" for v in live):
        out(f"DRAINED except records owned by the processes being restarted ({', '.join(map(str, sorted(restarting)))}):"
            " restart them together, then rerun without --restarting.")
        return 0
    out("DRAINED: every live start-time record matches its pid's unadjusted reading.")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--home", required=True, type=Path, help="HERMES_HOME to inspect (read-only)")
    parser.add_argument("--lock-dir", type=Path, default=None,
                        help="gateway lock dir (default: HERMES_GATEWAY_LOCK_DIR or the XDG state dir)")
    parser.add_argument("--no-lock-dir", action="store_true", help="skip the machine-wide lock dir")
    parser.add_argument("--no-profiles", action="store_true", help="skip <home>/profiles/*")
    parser.add_argument("--restarting", default="", metavar="PID,...",
                        help="pids the install restarts together; their own shifted records don't fail")
    args = parser.parse_args(argv)
    home = args.home.expanduser()
    if not home.is_dir():
        parser.error(f"--home {home} is not a directory")
    lock_dir = None if args.no_lock_dir else (args.lock_dir.expanduser() if args.lock_dir else default_lock_dir())
    try:
        restarting = frozenset(int(p) for p in args.restarting.replace(",", " ").split())
    except ValueError:
        parser.error(f"--restarting {args.restarting!r} is not a list of pids")
    return run(home, lock_dir, profiles=not args.no_profiles, restarting=restarting)


if __name__ == "__main__":
    sys.exit(main())
