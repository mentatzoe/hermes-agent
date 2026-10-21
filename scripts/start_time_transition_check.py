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
differs or can't be read; each is listed. Exit 2: bad arguments. Read-only: JSON is read, SQLite is
opened with ``mode=ro`` when its ``-wal`` already exists and ``immutable=1`` otherwise (so no
``-wal``/``-shm`` is ever created), nothing under ``--home`` or ``--lock-dir`` is written, and no
path is resolved from ``HERMES_HOME`` or the user's home except the lock-dir default.

Records checked (writer -> new-code reader; ``exact`` = equality, ``2s`` =
``start_time_fingerprints_match`` / ``fingerprints_match``, 200 cs):
- Kanban ``tasks`` and ``task_runs`` ``worker_pid``/``worker_started_at``, in ``<home>/kanban.db``
  and ``<home>/kanban/boards/*/kanban.db``: hermes_cli/kanban_db_dispatch.py:1477 and :1496 ->
  :402 ``_pid_recycled`` (2s, raw arm of kanban_worker_identity.py:39; the public arm also rescues
  Kanban, but a shifted row still means the transition isn't drained) and :519 reaper.
- ``<home>/gateway.pid`` and ``gateway.lock`` ``start_time``: gateway/status.py:776
  ``_build_pid_record`` -> :875 ``_live_pid_from_record`` (exact) and :2069
  ``get_running_pid_identity_strict`` (exact, ``_start_times_agree``).
- ``<home>/gateway_state.json`` ``start_time`` and ``platforms.*.writer_start_time``:
  gateway/status.py:1138/:1168 -> :1242 ``runtime_status_pid_is_live`` (exact) and
  hermes_cli/web_server_gateway.py:158 (exact).
- Delivery ledger, ``<home>/state.db`` ``delivery_obligations`` ``owner_pid``/``owner_started_at``
  (states pending/attempting/failed): gateway/delivery_ledger.py:173, :204 -> :232 ``_owner_alive``
  (2s), swept at :398.
- Async delegation, ``<home>/state.db`` ``async_delegations`` (running/stalling/finalizing):
  tools/async_delegation.py:97, :157 -> :254 ``alive`` (2s), recovered from :271.
- Cron, ``<home>/cron/executions.db`` ``executions`` ``pid``/``process_started_at``
  (claimed/running): cron/executions.py:51, :66, :126 -> :134 ``_owner_is_live`` (2s), from :375.
- Scoped locks and host rendezvous in the lock dir (``HERMES_GATEWAY_LOCK_DIR``, else
  ``$XDG_STATE_HOME`` or ``~/.local/state`` ``/hermes/gateway-locks``; gateway/status.py:324):
  ``*.lock`` from :1559 ``acquire_scoped_lock`` -> :1529 and :1801 (exact);
  ``host-*.json`` ``startTime`` from gateway/host_rendezvous.py:487 -> :239 (2s).
- Process registry ``<home>/processes.json`` ``host_start_time``: tools/process_registry.py:44,
  :552 -> :995 and :1004 (exact).
- Shared-metrics markers ``<home>/telemetry/shared_metrics/{process_markers,
  provider_setup_markers}/*.json``: hermes_cli/observability/shared_metrics_process.py:114 and
  shared_metrics_setup.py:163 -> gateway/status.py:1242 (exact).
- Lightpanda ``<home>/cache/browser-use/lightpanda/*.json``: tools/browser_lightpanda.py:189 ->
  :295 (exact).
- Local runtime ``<home>/runtimes/llamacpp/server.json`` ``start_time``/``owner_start_time``:
  hermes_cli/local_runtime/supervisor.py:233 and :236 -> recovery.py:40 (2s).
- Stop and takeover markers ``<home>/.gateway-planned-stop.json``, ``.gateway-takeover.json``
  and ``.hermes-update-paused-gateways.*.json`` (``target_start_time``, ``ct``):
  gateway/status.py:1952, :1687 and hermes_cli/update_pause_record.py:828,
  update_cmd_posix_pause.py:392 (exact).
Not checked, because their readers never used ``get_process_start_time`` and this change doesn't
touch them: hermes_cli/active_sessions.py:301 and hermes_cli/process_identity.py:64 (public
``create_time()`` floats), and in-memory baselines (MCP, terminal pgid, browser daemon).
With ``--profiles`` (default) every ``<home>/profiles/*`` home is checked the same way.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gateway.status_start_time import _start_times_agree, start_time_fingerprints_match  # noqa: E402

EXACT, TOLERANT, KANBAN = "exact", "2s", "kanban"
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
    rule: str  # EXACT | TOLERANT | KANBAN


@dataclass(frozen=True)
class Verdict:
    record: Record
    status: str  # "ok" | "differs" | "unreadable" | "dead"
    current: Any = None
    diff_cs: Optional[int] = None


# ── reading, strictly read-only ────────────────────────────────────────────────────────────────

def open_ro(path: Path) -> sqlite3.Connection:
    """Read-only SQLite handle that never creates a file next to ``path``.

    ``mode=ro`` on a WAL database creates ``-wal``/``-shm`` when they are missing, so it is used
    only when the ``-wal`` already exists (its committed frames are then visible). Without one,
    ``immutable=1`` reads the main file alone, which then holds every committed page.
    """
    wal = path.with_name(path.name + "-wal")
    flag = "mode=ro" if wal.exists() else "immutable=1"
    conn = sqlite3.connect(f"file:{path}?{flag}", uri=True, timeout=5)
    conn.execute("PRAGMA query_only = 1")
    return conn


def _rows(path: Path, sql: str) -> list[tuple]:
    if not path.is_file():
        return []
    conn = open_ro(path)
    try:
        return conn.execute(sql).fetchall()
    except sqlite3.OperationalError as exc:  # table or column absent in this schema version
        if "no such" in str(exc):
            return []
        raise
    finally:
        conn.close()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None  # absent, empty flock file, or torn mid-write: not a record


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


def _json_records(paths: list[Path], rule: str) -> Iterator[Record]:
    for path in paths:
        if path.is_file():
            yield from _walk_json(str(path), _read_json(path), rule)


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


def ledger_records(home: Path) -> Iterator[Record]:
    state_db = home / "state.db"
    for oid, pid, started in _rows(state_db, (
            "SELECT obligation_id, owner_pid, owner_started_at FROM delivery_obligations "
            "WHERE state IN ('pending', 'attempting', 'failed') AND owner_started_at IS NOT NULL")):
        if _as_pid(pid) is not None:
            yield Record(f"{state_db}:delivery_obligations/{oid}", int(pid), started, TOLERANT)
    for did, pid, started in _rows(state_db, (
            "SELECT delegation_id, owner_pid, owner_started_at FROM async_delegations "
            "WHERE state IN ('running', 'stalling', 'finalizing') AND owner_started_at IS NOT NULL")):
        if _as_pid(pid) is not None:
            yield Record(f"{state_db}:async_delegations/{did}", int(pid), started, TOLERANT)
    cron_db = home / "cron" / "executions.db"
    for eid, pid, started in _rows(cron_db, (
            "SELECT id, pid, process_started_at FROM executions "
            "WHERE status IN ('claimed', 'running') AND process_started_at IS NOT NULL")):
        if _as_pid(pid) is not None:
            yield Record(f"{cron_db}:executions/{eid}", int(pid), started, TOLERANT)


def home_file_records(home: Path) -> Iterator[Record]:
    exact = [home / "gateway.pid", home / "gateway.lock", home / "gateway_state.json",
             home / "processes.json", home / ".gateway-planned-stop.json", home / ".gateway-takeover.json",
             *sorted(home.glob(".hermes-update-paused-gateways.*.json"))]
    for sub in ("process_markers", "provider_setup_markers"):
        exact += sorted((home / "telemetry" / "shared_metrics" / sub).glob("*.json*"))
    exact += sorted((home / "cache" / "browser-use" / "lightpanda").glob("*.json"))
    yield from _json_records(exact, EXACT)
    yield from _json_records([home / "runtimes" / "llamacpp" / "server.json"], TOLERANT)


def lock_dir_records(lock_dir: Path) -> Iterator[Record]:
    yield from _json_records(sorted(lock_dir.glob("*.lock")), EXACT)
    yield from _json_records(sorted(lock_dir.glob("host-*.json")), TOLERANT)


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
    from hermes_cli.kanban_db_dispatch import _process_fingerprint

    return _process_fingerprint(pid)


def _start_half(value: Any) -> Optional[int]:
    text = str(value).rpartition("|")[2]
    try:
        return int(text)
    except ValueError:
        return None


def _matches(record: Record, current: Any) -> bool:
    try:
        if record.rule == KANBAN:
            from hermes_cli.kanban_worker_identity import fingerprints_match

            return fingerprints_match(record.recorded, current)
        if record.rule == TOLERANT:
            return start_time_fingerprints_match(record.recorded, current)
        return _start_times_agree(current, record.recorded)
    except (TypeError, ValueError):
        return False


def judge(record: Record, alive: Callable[[int], bool] = _pid_alive,
          raw: Callable[[int], Optional[int]] = _raw_start,
          kanban: Callable[[int], Optional[str]] = _kanban_current) -> Verdict:
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
        out: Callable[[str], None] = print, **judge_hooks: Any) -> int:
    records = collect(home, lock_dir, profiles)
    verdicts = [judge(r, **judge_hooks) for r in records]
    live = [v for v in verdicts if v.status != "dead"]
    bad = [v for v in live if v.status != "ok"]
    out(f"home {home}; lock dir {lock_dir}; {len(records)} start-time records, "
        f"{len(live)} with a live pid, {len(records) - len(live)} dead (skipped)")
    for v in live:
        diff = "" if v.diff_cs is None else f" diff {v.diff_cs:+d} cs"
        out(f"  {v.status.upper():10} pid {v.record.pid:>7} {v.record.rule:6} recorded {v.record.recorded} "
            f"now {v.current}{diff}  {v.record.source}")
    if bad:
        out(f"NOT DRAINED: {len(bad)} live record(s) differ from the unadjusted reading or can't be read. "
            "Don't restart onto the new code yet; let those owners exit or finish, then rerun.")
        return 1
    out("DRAINED: every live start-time record matches its pid's unadjusted reading.")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--home", required=True, type=Path, help="HERMES_HOME to inspect (read-only)")
    parser.add_argument("--lock-dir", type=Path, default=None,
                        help="gateway lock dir (default: HERMES_GATEWAY_LOCK_DIR or the XDG state dir)")
    parser.add_argument("--no-lock-dir", action="store_true", help="skip the machine-wide lock dir")
    parser.add_argument("--no-profiles", action="store_true", help="skip <home>/profiles/*")
    args = parser.parse_args(argv)
    home = args.home.expanduser()
    if not home.is_dir():
        parser.error(f"--home {home} is not a directory")
    lock_dir = None if args.no_lock_dir else (args.lock_dir.expanduser() if args.lock_dir else default_lock_dir())
    return run(home, lock_dir, profiles=not args.no_profiles)


if __name__ == "__main__":
    sys.exit(main())
