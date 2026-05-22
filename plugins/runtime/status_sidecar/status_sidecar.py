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
``<status>`` block and the agent loop appends it to the user
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

import html
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
MAX_RENDERED_EVENTS = 3
MAX_FIELD_CHARS = 240
MAX_FOCUS_LABEL_CHARS = 120
MAX_FOCUS_STATE_CHARS = 160
MAX_FOCUS_REF_CHARS = 160
MAX_DIGEST_CHARS = 300
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
    """Create / migrate the small status ledger schema.

    The schema deliberately stays typed and bounded: a single current-status
    row, plus per-session pointers. New columns are additive so live ledgers
    survive plugin upgrades.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS status (
            id                       INTEGER PRIMARY KEY CHECK (id = 1),
            active_kanban_task_id    TEXT,
            active_workspace         TEXT,
            active_project_slug      TEXT,
            focus_label              TEXT,
            focus_state              TEXT,
            focus_ref                TEXT,
            recent_activity_digest   TEXT,
            last_tool_invocation     TEXT,
            last_cron_job_id         TEXT,
            recent_drift_signals     TEXT,         -- JSON list[str]
            status_updated_at        REAL NOT NULL
        )
        """
    )
    for column, decl in (
        ("active_project_slug", "TEXT"),
        ("focus_label", "TEXT"),
        ("focus_state", "TEXT"),
        ("focus_ref", "TEXT"),
        ("recent_activity_digest", "TEXT"),
    ):
        try:
            conn.execute(f"ALTER TABLE status ADD COLUMN {column} {decl}")
        except sqlite3.OperationalError:
            # Column already exists.
            pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS active_sessions (
            session_id              TEXT PRIMARY KEY,
            surface                 TEXT,
            model                   TEXT,
            activity_class          TEXT,
            focus_label             TEXT,
            focus_state             TEXT,
            focus_ref               TEXT,
            active_project_slug     TEXT,
            active_kanban_task_id   TEXT,
            active_workspace        TEXT,
            last_tool_invocation    TEXT,
            last_seen_at            REAL NOT NULL
        )
        """
    )
    for column, decl in (
        ("activity_class", "TEXT"),
        ("focus_label", "TEXT"),
        ("focus_state", "TEXT"),
        ("focus_ref", "TEXT"),
    ):
        try:
            conn.execute(f"ALTER TABLE active_sessions ADD COLUMN {column} {decl}")
        except sqlite3.OperationalError:
            pass


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
                   focus_label,
                   focus_state,
                   focus_ref,
                   recent_activity_digest,
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
        drift = json.loads(row[9]) if row[9] else []
        if not isinstance(drift, list):
            drift = []
    except (ValueError, TypeError):
        drift = []

    return {
        "active_kanban_task_id": row[0],
        "active_workspace": row[1],
        "active_project_slug": row[2],
        "focus_label": row[3],
        "focus_state": row[4],
        "focus_ref": row[5],
        "recent_activity_digest": row[6],
        "last_tool_invocation": row[7],
        "last_cron_job_id": row[8],
        "recent_drift_signals": drift,
        "status_updated_at": row[10],
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


def _infer_activity_class(
    last_tool_invocation: Optional[str],
    active_project_slug: Optional[str] = None,
    active_kanban_task_id: Optional[str] = None,
    active_workspace: Optional[str] = None,
    focus_label: Optional[str] = None,
    focus_state: Optional[str] = None,
    focus_ref: Optional[str] = None,
) -> str:
    """Classify session rows so render can filter heartbeat noise.

    ``pre_llm_call`` touches are useful as liveness, but they should not crowd
    out real work pointers. Any typed project/task/workspace/focus field turns
    the row into a meaningful current-work pointer.
    """
    has_pointer = any(
        isinstance(v, str) and v.strip()
        for v in (
            active_project_slug,
            active_kanban_task_id,
            active_workspace,
            focus_label,
            focus_state,
            focus_ref,
        )
    )
    tool = last_tool_invocation if isinstance(last_tool_invocation, str) else ""
    if not has_pointer and tool == "pre_llm_call":
        return "heartbeat"
    if has_pointer:
        return "work"
    return "tool"


def touch_session(
    session_id: Optional[str],
    surface: Optional[str] = None,
    model: Optional[str] = None,
    activity_class: Optional[str] = None,
    focus_label: Optional[str] = None,
    focus_state: Optional[str] = None,
    focus_ref: Optional[str] = None,
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
                "SELECT surface, model, activity_class, focus_label, focus_state, focus_ref, "
                "active_project_slug, active_kanban_task_id, active_workspace, "
                "last_tool_invocation FROM active_sessions WHERE session_id = ?",
                (sid,),
            ).fetchone()
            if existing is None:
                (
                    cur_surface,
                    cur_model,
                    cur_class,
                    cur_focus_label,
                    cur_focus_state,
                    cur_focus_ref,
                    cur_project,
                    cur_task,
                    cur_ws,
                    cur_tool,
                ) = (None, None, None, None, None, None, None, None, None, None)
            else:
                (
                    cur_surface,
                    cur_model,
                    cur_class,
                    cur_focus_label,
                    cur_focus_state,
                    cur_focus_ref,
                    cur_project,
                    cur_task,
                    cur_ws,
                    cur_tool,
                ) = existing

            project = active_project_slug or _project_slug_from_path(active_workspace)
            next_tool = _clip(last_tool_invocation) if last_tool_invocation is not None else cur_tool
            inferred_class = _infer_activity_class(
                next_tool,
                project if project is not None else cur_project,
                active_kanban_task_id if active_kanban_task_id is not None else cur_task,
                active_workspace if active_workspace is not None else cur_ws,
                focus_label if focus_label is not None else cur_focus_label,
                focus_state if focus_state is not None else cur_focus_state,
                focus_ref if focus_ref is not None else cur_focus_ref,
            )
            conn.execute(
                """
                INSERT INTO active_sessions (
                    session_id, surface, model, activity_class, focus_label,
                    focus_state, focus_ref, active_project_slug,
                    active_kanban_task_id, active_workspace,
                    last_tool_invocation, last_seen_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    surface               = excluded.surface,
                    model                 = excluded.model,
                    activity_class        = excluded.activity_class,
                    focus_label           = excluded.focus_label,
                    focus_state           = excluded.focus_state,
                    focus_ref             = excluded.focus_ref,
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
                    _clip(activity_class) if activity_class is not None else inferred_class,
                    _clip(focus_label, MAX_FOCUS_LABEL_CHARS) if focus_label is not None else cur_focus_label,
                    _clip(focus_state, MAX_FOCUS_STATE_CHARS) if focus_state is not None else cur_focus_state,
                    _clip(focus_ref, MAX_FOCUS_REF_CHARS) if focus_ref is not None else cur_focus_ref,
                    _clip(project) if project is not None else cur_project,
                    _clip(active_kanban_task_id) if active_kanban_task_id is not None else cur_task,
                    _clip(active_workspace) if active_workspace is not None else cur_ws,
                    next_tool,
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
            SELECT session_id, surface, model, activity_class, focus_label,
                   focus_state, focus_ref, active_project_slug,
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
            "activity_class": row[3],
            "focus_label": row[4],
            "focus_state": row[5],
            "focus_ref": row[6],
            "active_project_slug": row[7],
            "active_kanban_task_id": row[8],
            "active_workspace": row[9],
            "last_tool_invocation": row[10],
            "last_seen_at": row[11],
        }
        for row in rows
    ]


def write_status(
    active_kanban_task_id: Optional[str] = None,
    active_workspace: Optional[str] = None,
    active_project_slug: Optional[str] = None,
    focus_label: Optional[str] = None,
    focus_state: Optional[str] = None,
    focus_ref: Optional[str] = None,
    recent_activity_digest: Optional[str] = None,
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
                "SELECT active_kanban_task_id, active_workspace, active_project_slug, "
                "focus_label, focus_state, focus_ref, recent_activity_digest, "
                "last_tool_invocation, last_cron_job_id, recent_drift_signals "
                "FROM status WHERE id = 1"
            ).fetchone()
            if existing is None:
                (
                    cur_task,
                    cur_ws,
                    cur_project,
                    cur_focus_label,
                    cur_focus_state,
                    cur_focus_ref,
                    cur_digest,
                    cur_tool,
                    cur_cron,
                    cur_drift,
                ) = (None, None, None, None, None, None, None, None, None, "[]")
            else:
                (
                    cur_task,
                    cur_ws,
                    cur_project,
                    cur_focus_label,
                    cur_focus_state,
                    cur_focus_ref,
                    cur_digest,
                    cur_tool,
                    cur_cron,
                    cur_drift,
                ) = existing

            conn.execute("BEGIN;")
            conn.execute(
                """
                INSERT INTO status (
                    id, active_kanban_task_id, active_workspace,
                    active_project_slug, focus_label, focus_state,
                    focus_ref, recent_activity_digest, last_tool_invocation,
                    last_cron_job_id, recent_drift_signals, status_updated_at
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    active_kanban_task_id  = excluded.active_kanban_task_id,
                    active_workspace       = excluded.active_workspace,
                    active_project_slug    = excluded.active_project_slug,
                    focus_label            = excluded.focus_label,
                    focus_state            = excluded.focus_state,
                    focus_ref              = excluded.focus_ref,
                    recent_activity_digest = excluded.recent_activity_digest,
                    last_tool_invocation   = excluded.last_tool_invocation,
                    last_cron_job_id       = excluded.last_cron_job_id,
                    status_updated_at      = excluded.status_updated_at
                """,
                (
                    _clip(active_kanban_task_id) if active_kanban_task_id is not None else cur_task,
                    _clip(active_workspace) if active_workspace is not None else cur_ws,
                    _clip(active_project_slug) if active_project_slug is not None else cur_project,
                    _clip(focus_label, MAX_FOCUS_LABEL_CHARS) if focus_label is not None else cur_focus_label,
                    _clip(focus_state, MAX_FOCUS_STATE_CHARS) if focus_state is not None else cur_focus_state,
                    _clip(focus_ref, MAX_FOCUS_REF_CHARS) if focus_ref is not None else cur_focus_ref,
                    _clip(recent_activity_digest, MAX_DIGEST_CHARS) if recent_activity_digest is not None else cur_digest,
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
        action = args_d.get("action") if isinstance(args_d.get("action"), str) else ""
        action = action.strip().lower()
        # Listing/polling cron jobs is low-signal heartbeat noise; don't let it
        # become top-level focus or drift.
        if action in {"list", "status"}:
            return {}

        out["last_tool_invocation"] = "cronjob"
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


def _xml_attr(value: Any, limit: int = MAX_FIELD_CHARS) -> str:
    """Bound + XML-escape an attribute value."""
    clipped = _clip(value, limit) or ""
    return html.escape(clipped, quote=True)


def _xml_text(value: Any, limit: int = MAX_FIELD_CHARS) -> str:
    clipped = _clip(value, limit) or ""
    return html.escape(clipped, quote=False)


def _attrs(items: List[Tuple[str, Any, int]]) -> str:
    parts = []
    for key, value, limit in items:
        if isinstance(value, str) and value.strip():
            parts.append(f'{key}="{_xml_attr(value, limit)}"')
        elif isinstance(value, (int, float)):
            parts.append(f'{key}="{value}"')
    return " ".join(parts)


def _session_is_heartbeat(sess: Dict[str, Any]) -> bool:
    cls = sess.get("activity_class")
    if isinstance(cls, str) and cls.strip().lower() == "heartbeat":
        return True
    tool = sess.get("last_tool_invocation")
    has_pointer = any(
        isinstance(sess.get(key), str) and sess.get(key).strip()
        for key in (
            "active_project_slug",
            "active_kanban_task_id",
            "active_workspace",
            "focus_label",
            "focus_state",
            "focus_ref",
        )
    )
    return tool == "pre_llm_call" and not has_pointer


def _compact_render(
    *,
    iso: str,
    age: int,
    record: Dict[str, Any],
    fresh_status: bool,
    sessions: List[Dict[str, Any]],
    include_digest: bool = True,
    include_events: bool = True,
    session_limit: int = MAX_ACTIVE_SESSIONS,
) -> str:
    """Render in priority order; caller can drop optional sections to fit budget."""
    lines: List[str] = [
        f'<status updated_iso="{_xml_attr(iso, 40)}" age_seconds="{age}" '
        'confidence="pointer" scope="current-work">'
    ]
    lines.append('  <note kind="verification">verify_refs_before_acting; no_memory_facts</note>')

    if fresh_status:
        focus_attrs = _attrs([
            ("project", record.get("active_project_slug"), MAX_FIELD_CHARS),
            ("task", record.get("active_kanban_task_id"), 80),
            ("workspace", record.get("active_workspace"), MAX_FIELD_CHARS),
            ("label", record.get("focus_label"), MAX_FOCUS_LABEL_CHARS),
            ("state", record.get("focus_state"), MAX_FOCUS_STATE_CHARS),
            ("ref", record.get("focus_ref"), MAX_FOCUS_REF_CHARS),
            ("last_tool", record.get("last_tool_invocation"), 80),
            ("last_cron", record.get("last_cron_job_id"), 80),
        ])
        if focus_attrs:
            lines.append(f"  <focus {focus_attrs} />")

    hidden_heartbeat = sum(1 for sess in sessions if _session_is_heartbeat(sess))
    meaningful_sessions = [sess for sess in sessions if not _session_is_heartbeat(sess)]
    rendered_sessions = meaningful_sessions[:session_limit]
    if not rendered_sessions and sessions:
        # If all we know is current-session liveness, keep one row so a fresh
        # newly-started session still has a visible posture pointer.
        rendered_sessions = sessions[:1]
        hidden_heartbeat = max(0, hidden_heartbeat - len(rendered_sessions))

    if rendered_sessions or hidden_heartbeat:
        lines.append(
            f'  <sessions total="{len(sessions)}" rendered="{len(rendered_sessions)}" '
            f'hidden_heartbeat="{hidden_heartbeat}">'
        )
        for sess in rendered_sessions:
            seen = sess.get("last_seen_at")
            age_s: Any = _fmt_age_seconds(float(seen)) if isinstance(seen, (int, float)) else ""
            session_attrs = _attrs([
                ("id", sess.get("session_id"), 80),
                ("surface", sess.get("surface"), 40),
                ("class", sess.get("activity_class") or ("heartbeat" if _session_is_heartbeat(sess) else "work"), 40),
                ("project", sess.get("active_project_slug"), MAX_FIELD_CHARS),
                ("task", sess.get("active_kanban_task_id"), 80),
                ("focus", sess.get("focus_label"), MAX_FOCUS_LABEL_CHARS),
                ("ref", sess.get("focus_ref"), MAX_FOCUS_REF_CHARS),
                ("last_tool", sess.get("last_tool_invocation"), 80),
                ("age_s", age_s, 20),
            ])
            if session_attrs:
                lines.append(f"    <session {session_attrs} />")
        lines.append("  </sessions>")

    if include_events and fresh_status:
        signals = record.get("recent_drift_signals") or []
        if isinstance(signals, list):
            events = [s for s in signals[-MAX_RENDERED_EVENTS:] if isinstance(s, str) and s.strip()]
        else:
            events = []
        if events:
            lines.append(f'  <events count="{len(events)}">')
            for sig in events:
                lines.append(f"    <event>{_xml_text(sig, MAX_DRIFT_SIGNAL_CHARS)}</event>")
            lines.append("  </events>")

    if include_digest and fresh_status:
        digest = record.get("recent_activity_digest")
        if isinstance(digest, str) and digest.strip():
            lines.append(f'  <digest text="{_xml_attr(digest, MAX_DIGEST_CHARS)}" />')

    lines.append("</status>")
    return "\n".join(lines)


def render_status_block(ttl_seconds: float = 4 * 3600) -> str:
    """Render fresh status/session pointers as compact XML-ish current-work data.

    Returns an empty string when no status row or active-session row is fresh.
    Stale task/project state is never rendered just because the current
    session is fresh. Rendering is priority-based: focus first, then meaningful
    sessions, then optional events/digest. It should not hard-truncate.
    """
    try:
        record = read_status()
        sessions = read_active_sessions(ttl_seconds=ttl_seconds, limit=MAX_ACTIVE_SESSIONS * 4)
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

    # First pass: full compact block. Then drop optional sections rather than
    # using a misleading hard truncation marker.
    for include_events, include_digest, session_limit in (
        (True, True, MAX_ACTIVE_SESSIONS),
        (False, True, MAX_ACTIVE_SESSIONS),
        (False, False, MAX_ACTIVE_SESSIONS),
        (False, False, 2),
        (False, False, 1),
    ):
        block = _compact_render(
            iso=iso,
            age=age,
            record=record,
            fresh_status=fresh_status,
            sessions=sessions,
            include_events=include_events,
            include_digest=include_digest,
            session_limit=session_limit,
        )
        if len(block) <= BLOCK_CHAR_BUDGET:
            return block

    # Last resort: return the strictly highest-priority focus note only.
    return _compact_render(
        iso=iso,
        age=age,
        record=record,
        fresh_status=fresh_status,
        sessions=[],
        include_events=False,
        include_digest=False,
        session_limit=0,
    )
