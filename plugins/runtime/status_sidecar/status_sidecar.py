"""status_sidecar — durable HEARTBEAT status ledger.

A small SQLite-backed sidecar holding *current operational hints* for the
agent:

  * ``active_kanban_task_id``     — what the agent should be working on
  * ``active_workspace``          — where it's working
  * ``last_tool_invocation``      — last tool name (deterministic update path
                                    in ``post_tool_call``)
  * ``last_cron_job_id``          — last cron-job run id, if any
  * ``recent_drift_signals``      — bounded ring buffer of short annotations
                                    (e.g. "user asked to simplify scope")
  * ``status_updated_at``         — float epoch seconds; the TTL anchor

The plugin's ``pre_llm_call`` hook renders this into a fenced
``<system_status>`` block and the agent loop appends it to the user
message via ``_plugin_user_context`` — the system prompt is never touched
(preserves prompt cache).

Design rules (from card t_8a4caa0e):

1. Stale state must NEVER be injected as if it were current truth. If the
   timestamp is older than ``ttl_seconds``, the block is dropped (or rendered
   with an explicit "stale" marker if and only if it's useful — never bare).
2. Corruption (truncated SQLite, missing file, junk bytes) must NOT crash
   the agent loop. ``read_status()`` swallows ``sqlite3.OperationalError``
   and ``sqlite3.DatabaseError`` and returns ``{}``.
3. The injected block stays under ``BLOCK_CHAR_BUDGET`` (~300 tokens of
   ASCII). Drift signals are truncated and the ring buffer is bounded.
4. Writes are atomic at the SQLite layer (single-row UPSERT in a
   transaction).
5. The ledger lives at ``$HERMES_HOME/state/status.db``.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover — plugin may load before constants resolves
    def get_hermes_home() -> Path:  # type: ignore[no-redef]
        val = (os.environ.get("HERMES_HOME") or "").strip()
        return Path(val).resolve() if val else (Path.home() / ".hermes").resolve()


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bounds — keep the block small and the ledger bounded.
# ---------------------------------------------------------------------------

MAX_DRIFT_SIGNALS = 5
MAX_DRIFT_SIGNAL_CHARS = 200
BLOCK_CHAR_BUDGET = 1200  # ~300 tokens of ASCII English

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def get_state_dir() -> Path:
    """``$HERMES_HOME/state/`` — separate from logs/, sessions/, etc."""
    return get_hermes_home() / "state"


def get_ledger_path() -> Path:
    return get_state_dir() / "status.db"


# ---------------------------------------------------------------------------
# DB layer (process-local lock around the single-row write path)
# ---------------------------------------------------------------------------

_write_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    path = get_ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # ``isolation_level=None`` so we manage transactions explicitly via
    # ``BEGIN ... COMMIT`` for atomic UPSERTs.
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create the single-row ``status`` table if it doesn't exist.

    Schema is a fixed set of scalar columns plus one JSON blob for
    drift signals. One row only (``id=1``).
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS status (
            id                       INTEGER PRIMARY KEY CHECK (id = 1),
            active_kanban_task_id    TEXT,
            active_workspace         TEXT,
            last_tool_invocation     TEXT,
            last_cron_job_id         TEXT,
            recent_drift_signals     TEXT,         -- JSON list[str]
            status_updated_at        REAL NOT NULL
        )
        """
    )


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------

def read_status() -> Dict[str, Any]:
    """Return the current status record as a dict, or ``{}`` if the ledger
    is missing / corrupt / empty.

    NEVER raises. Callers must be safe under all conditions.
    """
    path = get_ledger_path()
    if not path.exists():
        return {}
    try:
        with _connect() as conn:
            row = conn.execute(
                """
                SELECT active_kanban_task_id,
                       active_workspace,
                       last_tool_invocation,
                       last_cron_job_id,
                       recent_drift_signals,
                       status_updated_at
                  FROM status
                 WHERE id = 1
                """
            ).fetchone()
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
        logger.debug("status_sidecar read failed: %s", exc)
        return {}
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("status_sidecar unexpected read error: %s", exc)
        return {}

    if row is None:
        return {}

    try:
        drift = json.loads(row[4]) if row[4] else []
        if not isinstance(drift, list):
            drift = []
    except (ValueError, TypeError):
        drift = []

    return {
        "active_kanban_task_id": row[0],
        "active_workspace": row[1],
        "last_tool_invocation": row[2],
        "last_cron_job_id": row[3],
        "recent_drift_signals": drift,
        "status_updated_at": row[5],
    }


def write_status(
    active_kanban_task_id: Optional[str] = None,
    active_workspace: Optional[str] = None,
    last_tool_invocation: Optional[str] = None,
    last_cron_job_id: Optional[str] = None,
) -> bool:
    """Upsert the single status row. Returns True on success.

    Each call refreshes ``status_updated_at``. Drift signals are written
    via :func:`append_drift_signal` so the bounded ring buffer is owned in
    one place.

    Values left as ``None`` are PRESERVED from the existing row. To clear
    a field, pass ``""``.
    """
    now = time.time()
    with _write_lock:
        try:
            with _connect() as conn:
                _init_schema(conn)
                existing = conn.execute(
                    "SELECT active_kanban_task_id, active_workspace, "
                    "last_tool_invocation, last_cron_job_id, "
                    "recent_drift_signals FROM status WHERE id = 1"
                ).fetchone()
                if existing is None:
                    cur_task, cur_ws, cur_tool, cur_cron, cur_drift = (
                        None, None, None, None, "[]"
                    )
                else:
                    cur_task, cur_ws, cur_tool, cur_cron, cur_drift = existing

                conn.execute("BEGIN;")
                conn.execute(
                    """
                    INSERT INTO status (
                        id, active_kanban_task_id, active_workspace,
                        last_tool_invocation, last_cron_job_id,
                        recent_drift_signals, status_updated_at
                    ) VALUES (1, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        active_kanban_task_id = excluded.active_kanban_task_id,
                        active_workspace      = excluded.active_workspace,
                        last_tool_invocation  = excluded.last_tool_invocation,
                        last_cron_job_id      = excluded.last_cron_job_id,
                        status_updated_at     = excluded.status_updated_at
                    """,
                    (
                        active_kanban_task_id if active_kanban_task_id is not None else cur_task,
                        active_workspace if active_workspace is not None else cur_ws,
                        last_tool_invocation if last_tool_invocation is not None else cur_tool,
                        last_cron_job_id if last_cron_job_id is not None else cur_cron,
                        cur_drift if cur_drift is not None else "[]",
                        now,
                    ),
                )
                conn.execute("COMMIT;")
            return True
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
            logger.warning("status_sidecar write failed: %s", exc)
            return False
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("status_sidecar unexpected write error: %s", exc)
            return False


def append_drift_signal(signal: str) -> bool:
    """Append a short annotation to the bounded ring buffer.

    Drift signals are FIFO: oldest entries are dropped first. Each signal
    is truncated to :data:`MAX_DRIFT_SIGNAL_CHARS`.
    """
    if not isinstance(signal, str) or not signal.strip():
        return False
    clipped = signal.strip()[:MAX_DRIFT_SIGNAL_CHARS]
    now = time.time()
    with _write_lock:
        try:
            with _connect() as conn:
                _init_schema(conn)
                row = conn.execute(
                    "SELECT recent_drift_signals FROM status WHERE id = 1"
                ).fetchone()
                if row is None or not row[0]:
                    signals: List[str] = []
                else:
                    try:
                        signals = json.loads(row[0])
                        if not isinstance(signals, list):
                            signals = []
                    except (ValueError, TypeError):
                        signals = []
                signals.append(clipped)
                # Bound the ring buffer.
                if len(signals) > MAX_DRIFT_SIGNALS:
                    signals = signals[-MAX_DRIFT_SIGNALS:]
                blob = json.dumps(signals)
                conn.execute("BEGIN;")
                conn.execute(
                    """
                    INSERT INTO status (id, recent_drift_signals, status_updated_at)
                    VALUES (1, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        recent_drift_signals = excluded.recent_drift_signals,
                        status_updated_at    = excluded.status_updated_at
                    """,
                    (blob, now),
                )
                conn.execute("COMMIT;")
            return True
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
            logger.warning("status_sidecar drift append failed: %s", exc)
            return False
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("status_sidecar unexpected drift error: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Test / internal helper — used only by tests to simulate ageing.
# ---------------------------------------------------------------------------

def _force_set_updated_at(ts: float) -> None:
    """Internal: overwrite ``status_updated_at`` without touching other
    fields. Used by tests to simulate stale ledgers.
    """
    with _write_lock:
        with _connect() as conn:
            _init_schema(conn)
            conn.execute(
                "UPDATE status SET status_updated_at = ? WHERE id = 1",
                (ts,),
            )


# ---------------------------------------------------------------------------
# Staleness / rendering
# ---------------------------------------------------------------------------

def is_fresh(record: Optional[Dict[str, Any]], ttl_seconds: float) -> bool:
    """True iff the record has a ``status_updated_at`` within ``ttl_seconds``
    of now. Empty / missing record → False.
    """
    if not record:
        return False
    ts = record.get("status_updated_at")
    if not isinstance(ts, (int, float)):
        return False
    age = time.time() - float(ts)
    if age < 0:
        # Clock skew — treat as fresh but odd.
        return True
    return age <= float(ttl_seconds)


def _fmt_iso(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return "unknown"


def _fmt_age_seconds(ts: float) -> int:
    try:
        return max(0, int(time.time() - float(ts)))
    except (TypeError, ValueError):
        return -1


def render_status_block(ttl_seconds: float = 4 * 3600) -> str:
    """Render the current status record as a fenced ``<system_status>`` block.

    Returns an empty string when:
      - the ledger doesn't exist,
      - the ledger is corrupt,
      - the record is empty,
      - the record is stale beyond ``ttl_seconds`` (no laundering).

    NEVER raises.
    """
    try:
        record = read_status()
    except Exception:  # pragma: no cover — read_status already swallows
        return ""

    if not record:
        return ""

    ts = record.get("status_updated_at")
    if not isinstance(ts, (int, float)):
        return ""

    fresh = is_fresh(record, ttl_seconds=ttl_seconds)
    if not fresh:
        # Strict policy: don't inject stale state. Returning empty avoids
        # the agent treating ancient hints as current truth.
        return ""

    age = _fmt_age_seconds(ts)
    iso = _fmt_iso(ts)

    lines: List[str] = []
    lines.append(f"<system_status updated_iso=\"{iso}\" age_seconds=\"{age}\">")
    lines.append(
        "  Operational hints from the local status ledger. Treat as a "
        "low-confidence pointer, not canonical truth. Verify with the "
        "relevant tool before acting on it."
    )

    def _add(key: str, label: str) -> None:
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            lines.append(f"  {label}: {val.strip()}")

    _add("active_kanban_task_id", "active_kanban_task_id")
    _add("active_workspace", "active_workspace")
    _add("last_tool_invocation", "last_tool_invocation")
    _add("last_cron_job_id", "last_cron_job_id")

    signals = record.get("recent_drift_signals") or []
    if isinstance(signals, list) and signals:
        lines.append("  recent_drift_signals:")
        # Show newest last (chronological).
        for sig in signals[-MAX_DRIFT_SIGNALS:]:
            if isinstance(sig, str) and sig.strip():
                lines.append(f"    - {sig.strip()[:MAX_DRIFT_SIGNAL_CHARS]}")

    lines.append("</system_status>")

    block = "\n".join(lines)
    if len(block) > BLOCK_CHAR_BUDGET:
        # Hard-truncate. Keep the opening tag + a truncated body + closing tag
        # so the block stays parseable.
        body_cap = BLOCK_CHAR_BUDGET - len("</system_status>") - 4
        block = block[:body_cap].rstrip() + "\n  [truncated]\n</system_status>"
    return block
