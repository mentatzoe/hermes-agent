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
MAX_ACTIVE_SESSIONS = 5
MAX_FIELD_CHARS = 240
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
            active_project_slug      TEXT,
            last_tool_invocation     TEXT,
            last_cron_job_id         TEXT,
            recent_drift_signals     TEXT,         -- JSON list[str]
            status_updated_at        REAL NOT NULL
        )
        """
    )
    try:
        conn.execute("ALTER TABLE status ADD COLUMN active_project_slug TEXT")
    except sqlite3.OperationalError:
        # Column already exists.
        pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS active_sessions (
            session_id              TEXT PRIMARY KEY,
            surface                 TEXT,
            model                   TEXT,
            active_project_slug     TEXT,
            active_kanban_task_id   TEXT,
            active_workspace        TEXT,
            last_tool_invocation    TEXT,
            last_seen_at            REAL NOT NULL
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
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = _connect()
        _init_schema(conn)
        row = conn.execute(
            """
            SELECT active_kanban_task_id,
                   active_workspace,
                   active_project_slug,
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
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # pragma: no cover — defensive
                pass

    if row is None:
        return {}

    try:
        drift = json.loads(row[5]) if row[5] else []
        if not isinstance(drift, list):
            drift = []
    except (ValueError, TypeError):
        drift = []

    return {
        "active_kanban_task_id": row[0],
        "active_workspace": row[1],
        "active_project_slug": row[2],
        "last_tool_invocation": row[3],
        "last_cron_job_id": row[4],
        "recent_drift_signals": drift,
        "status_updated_at": row[6],
    }

def _clip(value: Optional[str], limit: int = MAX_FIELD_CHARS) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    return value.strip()[:limit]


def _project_slug_from_path(path: Optional[str]) -> str:
    """Best-effort non-content project slug from a path.

    Deliberately heuristic and bounded: this is a current-work pointer, not a
    canonical project registry. Never reads the file or shells out to git.
    """
    if not isinstance(path, str) or not path.strip():
        return ""
    raw = path.strip()
    parts = [p for p in raw.replace("\\", "/").split("/") if p]
    if "hermes-agent" in parts:
        return "hermes-agent"
    if "github" in parts:
        try:
            i = parts.index("github")
            repo = parts[i + 1]
        except (ValueError, IndexError):
            return ""
        slug = f"github/{repo}"
        rel = parts[i + 2:]
        if repo == "aleph-vault" and len(rel) >= 3 and rel[0] == "projects":
            slug += "/" + "/".join(rel[:3])
        elif repo == "aleph-vault" and rel:
            slug += "/" + rel[0]
        return slug[:MAX_FIELD_CHARS]
    if "workspaces" in parts:
        try:
            i = parts.index("workspaces")
            task = parts[i + 1]
            return f"kanban/{task}"[:MAX_FIELD_CHARS]
        except (ValueError, IndexError):
            return ""
    return ""


def touch_session(
    session_id: Optional[str],
    surface: Optional[str] = None,
    model: Optional[str] = None,
    active_project_slug: Optional[str] = None,
    active_kanban_task_id: Optional[str] = None,
    active_workspace: Optional[str] = None,
    last_tool_invocation: Optional[str] = None,
) -> bool:
    """Record a bounded active-session pointer without refreshing status row TTL."""
    sid = _clip(session_id)
    if not sid:
        return False
    now = time.time()
    with _write_lock:
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = _connect()
            _init_schema(conn)
            existing = conn.execute(
                "SELECT surface, model, active_project_slug, active_kanban_task_id, "
                "active_workspace, last_tool_invocation FROM active_sessions "
                "WHERE session_id = ?",
                (sid,),
            ).fetchone()
            if existing is None:
                cur_surface, cur_model, cur_project, cur_task, cur_ws, cur_tool = (
                    None, None, None, None, None, None
                )
            else:
                cur_surface, cur_model, cur_project, cur_task, cur_ws, cur_tool = existing
            project = active_project_slug or _project_slug_from_path(active_workspace)
            conn.execute(
                """
                INSERT INTO active_sessions (
                    session_id, surface, model, active_project_slug,
                    active_kanban_task_id, active_workspace,
                    last_tool_invocation, last_seen_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    surface               = excluded.surface,
                    model                 = excluded.model,
                    active_project_slug   = excluded.active_project_slug,
                    active_kanban_task_id = excluded.active_kanban_task_id,
                    active_workspace      = excluded.active_workspace,
                    last_tool_invocation  = excluded.last_tool_invocation,
                    last_seen_at          = excluded.last_seen_at
                """,
                (
                    sid,
                    _clip(surface) if surface is not None else cur_surface,
                    _clip(model) if model is not None else cur_model,
                    _clip(project) if project is not None else cur_project,
                    _clip(active_kanban_task_id) if active_kanban_task_id is not None else cur_task,
                    _clip(active_workspace) if active_workspace is not None else cur_ws,
                    _clip(last_tool_invocation) if last_tool_invocation is not None else cur_tool,
                    now,
                ),
            )
            return True
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
            logger.debug("status_sidecar session touch failed: %s", exc)
            return False
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("status_sidecar unexpected session touch error: %s", exc)
            return False
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # pragma: no cover — defensive
                    pass


def read_active_sessions(ttl_seconds: float, limit: int = MAX_ACTIVE_SESSIONS) -> List[Dict[str, Any]]:
    """Return fresh active-session pointers, newest first. NEVER raises."""
    path = get_ledger_path()
    if not path.exists() or ttl_seconds < 0:
        return []
    cutoff = time.time() - float(ttl_seconds)
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = _connect()
        _init_schema(conn)
        rows = conn.execute(
            """
            SELECT session_id, surface, model, active_project_slug,
                   active_kanban_task_id, active_workspace,
                   last_tool_invocation, last_seen_at
              FROM active_sessions
             WHERE last_seen_at >= ?
             ORDER BY last_seen_at DESC
             LIMIT ?
            """,
            (cutoff, int(limit)),
        ).fetchall()
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
        logger.debug("status_sidecar active-session read failed: %s", exc)
        return []
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("status_sidecar unexpected active-session read error: %s", exc)
        return []
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # pragma: no cover — defensive
                pass
    return [
        {
            "session_id": row[0],
            "surface": row[1],
            "model": row[2],
            "active_project_slug": row[3],
            "active_kanban_task_id": row[4],
            "active_workspace": row[5],
            "last_tool_invocation": row[6],
            "last_seen_at": row[7],
        }
        for row in rows
    ]


def write_status(
    active_kanban_task_id: Optional[str] = None,
    active_workspace: Optional[str] = None,
    active_project_slug: Optional[str] = None,
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
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = _connect()
            _init_schema(conn)
            existing = conn.execute(
                "SELECT active_kanban_task_id, active_workspace, "
                "active_project_slug, last_tool_invocation, last_cron_job_id, "
                "recent_drift_signals FROM status WHERE id = 1"
            ).fetchone()
            if existing is None:
                cur_task, cur_ws, cur_project, cur_tool, cur_cron, cur_drift = (
                    None, None, None, None, None, "[]"
                )
            else:
                cur_task, cur_ws, cur_project, cur_tool, cur_cron, cur_drift = existing

            conn.execute("BEGIN;")
            conn.execute(
                """
                INSERT INTO status (
                    id, active_kanban_task_id, active_workspace,
                    active_project_slug, last_tool_invocation,
                    last_cron_job_id, recent_drift_signals, status_updated_at
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    active_kanban_task_id = excluded.active_kanban_task_id,
                    active_workspace      = excluded.active_workspace,
                    active_project_slug   = excluded.active_project_slug,
                    last_tool_invocation  = excluded.last_tool_invocation,
                    last_cron_job_id      = excluded.last_cron_job_id,
                    status_updated_at     = excluded.status_updated_at
                """,
                (
                    _clip(active_kanban_task_id) if active_kanban_task_id is not None else cur_task,
                    _clip(active_workspace) if active_workspace is not None else cur_ws,
                    _clip(active_project_slug) if active_project_slug is not None else cur_project,
                    _clip(last_tool_invocation) if last_tool_invocation is not None else cur_tool,
                    _clip(last_cron_job_id) if last_cron_job_id is not None else cur_cron,
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
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # pragma: no cover — defensive
                    pass

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
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = _connect()
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
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # pragma: no cover — defensive
                    pass


# ---------------------------------------------------------------------------
# Deterministic update derivation from observed tool calls.
#
# ``derive_status_from_tool_call`` is a pure function (no I/O). The plugin's
# ``post_tool_call`` hook calls this, then passes the returned dict into
# :func:`write_status`. Splitting derivation from I/O makes the policy
# trivially unit-testable and ensures we have ONE place that decides what
# is safe to land in the ledger.
#
# **Hard rule:** nothing from a tool's *result body* or *args content* may
# end up in the ledger. We only extract opaque identifiers (ids, paths) and
# the tool name itself. Anything that could carry user prose — task titles,
# Kanban bodies, cron prompts, file contents, terminal output, diffs — is
# explicitly NOT captured.
# ---------------------------------------------------------------------------

# Tools whose post-call we track. Anything not in this set is ignored.
_TRACKED_KANBAN_TOOLS = {
    "kanban_show",
    "kanban_create",
    "kanban_complete",
    "kanban_block",
    "kanban_comment",
    "kanban_heartbeat",
    "kanban_link",
}

# File-mutation tools — recorded as "last_tool_invocation" with the basename
# of the touched path appended as a short breadcrumb (no content).
_TRACKED_FILE_TOOLS = {"write_file", "patch"}

# Tools we explicitly skip: too noisy, recursive, or carry no useful pointer.
_IGNORED_TOOLS = {
    "terminal",        # runs every shell command; would churn the ledger.
    "status_update",   # avoid recursive write loops.
    "read_file",       # observational, no state change worth tracking.
    "search_files",    # ditto.
}


def _safe_json_loads(s: Any) -> Any:
    """Tolerant JSON-loader. Returns ``None`` on any failure rather than
    raising. Tool ``result`` strings are *usually* JSON but not guaranteed.
    """
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return None


def _short_basename(path: Optional[str], limit: int = 80) -> str:
    """Return the trailing path component truncated to ``limit`` chars.

    Used as a *non-content* breadcrumb on file-mutation tools. We
    intentionally do NOT capture full absolute paths because they can leak
    info about other users / projects / workspaces; the basename plus a
    truncation is enough to remind the agent "you just touched X".
    """
    if not isinstance(path, str) or not path.strip():
        return ""
    try:
        base = os.path.basename(path.strip()) or path.strip()
    except Exception:
        base = path.strip()
    if len(base) > limit:
        base = base[:limit]
    return base


def derive_status_from_tool_call(
    tool_name: str,
    args: Any,
    result: Any,
) -> Dict[str, Any]:
    """Decide which ledger fields to update from a single tool call.

    Returns a dict suitable for ``**kwargs`` to :func:`write_status`,
    plus an optional ``"_drift"`` key for an append-only ring-buffer
    annotation. The hook applies these.

    Hard guarantees:

    * Never returns tool-result *body*, file content, kanban prose, cron
      prompts, terminal output, or diff text.
    * Returns ``{}`` for any tool not explicitly tracked.
    * Never raises.
    """
    if not isinstance(tool_name, str) or not tool_name:
        return {}
    if tool_name in _IGNORED_TOOLS:
        return {}

    out: Dict[str, Any] = {}
    args_d = args if isinstance(args, dict) else {}
    parsed = _safe_json_loads(result)

    # --- Kanban -----------------------------------------------------------
    if tool_name in _TRACKED_KANBAN_TOOLS:
        out["last_tool_invocation"] = tool_name

        # kanban_show: pull active task id + workspace from the result's
        # ``task`` envelope, falling back to the input task_id arg.
        if tool_name == "kanban_show":
            task_id = args_d.get("task_id")
            ws = None
            if isinstance(parsed, dict):
                task_obj = parsed.get("task")
                if isinstance(task_obj, dict):
                    # Result task.id is authoritative.
                    if isinstance(task_obj.get("id"), str):
                        task_id = task_obj["id"]
                    cand_ws = task_obj.get("workspace_path")
                    if isinstance(cand_ws, str) and cand_ws.strip():
                        ws = cand_ws.strip()
            if isinstance(task_id, str) and task_id.strip():
                out["active_kanban_task_id"] = task_id.strip()
            if ws:
                out["active_workspace"] = ws
                project = _project_slug_from_path(ws)
                if project:
                    out["active_project_slug"] = project

        # kanban_create returns a new task_id; record only the id, never
        # the title/body the agent supplied as args.
        elif tool_name == "kanban_create":
            if isinstance(parsed, dict):
                tid = parsed.get("task_id")
                if isinstance(tid, str) and tid.strip():
                    # Don't override active_kanban_task_id here — the
                    # creator is not necessarily switching to the new card.
                    out["_drift"] = f"created kanban card {tid.strip()}"

        # kanban_complete / kanban_block / kanban_comment: just record the
        # tool name + the task id pointer from args. No summary / body / reason.
        elif tool_name in {"kanban_complete", "kanban_block", "kanban_comment", "kanban_link"}:
            tid = args_d.get("task_id") or args_d.get("parent_id")
            if isinstance(tid, str) and tid.strip():
                # Active task probably IS this id; update the pointer so the
                # next pre_llm_call shows the agent it just acted on tid.
                out["active_kanban_task_id"] = tid.strip()

        return out

    # --- Cron -------------------------------------------------------------
    if tool_name == "cronjob":
        out["last_tool_invocation"] = "cronjob"
        action = args_d.get("action") if isinstance(args_d.get("action"), str) else ""
        # Prefer result.job_id (returned by create); fall back to args.job_id
        # (passed for update/remove/run/etc).
        jid: Optional[str] = None
        if isinstance(parsed, dict):
            cand = parsed.get("job_id")
            if isinstance(cand, str) and cand.strip():
                jid = cand.strip()
        if not jid:
            cand = args_d.get("job_id")
            if isinstance(cand, str) and cand.strip():
                jid = cand.strip()
        if jid:
            out["last_cron_job_id"] = jid
        # No prompt / schedule / script content stored.
        if action:
            out["_drift"] = f"cronjob {action}" + (f" {jid}" if jid else "")
        return out

    # --- File mutations ---------------------------------------------------
    if tool_name in _TRACKED_FILE_TOOLS:
        path = args_d.get("path")
        base = _short_basename(path)
        if base:
            out["last_tool_invocation"] = f"{tool_name}:{base}"
        else:
            out["last_tool_invocation"] = tool_name
        project = _project_slug_from_path(path)
        if project:
            out["active_project_slug"] = project
        return out

    return {}


# ---------------------------------------------------------------------------
# Test / internal helper — used only by tests to simulate ageing.
# ---------------------------------------------------------------------------

def _force_set_updated_at(ts: float) -> None:
    """Internal: overwrite ``status_updated_at`` without touching other
    fields. Used by tests to simulate stale ledgers.
    """
    with _write_lock:
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = _connect()
            _init_schema(conn)
            conn.execute(
                "UPDATE status SET status_updated_at = ? WHERE id = 1",
                (ts,),
            )
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # pragma: no cover — defensive
                    pass


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
    """Render fresh status/session pointers as a fenced ``<system_status>`` block.

    Returns an empty string when no status row or active-session row is fresh.
    Stale task/project state is never rendered just because the current
    session is fresh.
    """
    try:
        record = read_status()
        sessions = read_active_sessions(ttl_seconds=ttl_seconds)
    except Exception:  # pragma: no cover — callees already swallow
        return ""

    fresh_status = bool(record) and is_fresh(record, ttl_seconds=ttl_seconds)
    if not fresh_status:
        record = {}
    if not fresh_status and not sessions:
        return ""

    timestamps: List[float] = []
    ts = record.get("status_updated_at")
    if isinstance(ts, (int, float)):
        timestamps.append(float(ts))
    for sess in sessions:
        s_ts = sess.get("last_seen_at")
        if isinstance(s_ts, (int, float)):
            timestamps.append(float(s_ts))
    anchor_ts = max(timestamps) if timestamps else time.time()
    age = _fmt_age_seconds(anchor_ts)
    iso = _fmt_iso(anchor_ts)

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
            lines.append(f"  {label}: {val.strip()[:MAX_FIELD_CHARS]}")

    if fresh_status:
        _add("active_project_slug", "active_project_slug")
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

    if sessions:
        lines.append("  active_sessions:")
        # Oldest first for readable chronology within the already-bounded set.
        for sess in reversed(sessions[-MAX_ACTIVE_SESSIONS:]):
            bits = []
            sid = sess.get("session_id")
            if isinstance(sid, str) and sid.strip():
                bits.append(f"session_id={sid.strip()[:80]}")
            surface = sess.get("surface")
            if isinstance(surface, str) and surface.strip():
                bits.append(f"surface={surface.strip()[:40]}")
            project = sess.get("active_project_slug")
            if isinstance(project, str) and project.strip():
                bits.append(f"project={project.strip()[:120]}")
            task = sess.get("active_kanban_task_id")
            if isinstance(task, str) and task.strip():
                bits.append(f"task={task.strip()[:80]}")
            tool = sess.get("last_tool_invocation")
            if isinstance(tool, str) and tool.strip():
                bits.append(f"last_tool={tool.strip()[:80]}")
            seen = sess.get("last_seen_at")
            if isinstance(seen, (int, float)):
                bits.append(f"age_seconds={_fmt_age_seconds(float(seen))}")
            if bits:
                lines.append("    - " + " ".join(bits))

    lines.append("</system_status>")

    block = "\n".join(lines)
    if len(block) > BLOCK_CHAR_BUDGET:
        # Hard-truncate. Keep the opening tag + a truncated body + closing tag
        # so the block stays parseable.
        body_cap = BLOCK_CHAR_BUDGET - len("</system_status>") - 4
        block = block[:body_cap].rstrip() + "\n  [truncated]\n</system_status>"
    return block
