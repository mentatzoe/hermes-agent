"""status-sidecar plugin — pre-response status anchoring for Aleph/Hermes.

Wires two behaviours:

1. ``pre_llm_call`` hook — reads the durable status ledger and, when fresh,
   returns a ``{"context": "<system_status>...</system_status>"}`` dict.
   The agent loop (``run_agent.py``) appends that to the current turn's
   user message via ``_plugin_user_context``. The system prompt is never
   mutated; prompt-cache prefix stays intact.

2. ``status_update`` tool — short-annotation write path for the agent. Lets
   it post a drift signal or update the active task/workspace pointers. NOT
   a general-purpose memory store; entries are length-bounded and the
   ring buffer is small.

Rollback:
  - Disable: remove ``status-sidecar`` from ``plugins.enabled`` in
    ``~/.hermes/config.yaml`` (or add it to ``plugins.disabled``).
  - The ledger file at ``$HERMES_HOME/state/status.db`` is harmless when
    not read. To wipe it: ``rm $HERMES_HOME/state/status.db*``.
  - Activation requires a gateway / CLI restart (plugins discovered at
    startup).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, Optional

from . import status_sidecar as ss

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_TTL_SECONDS = 4 * 3600  # 4 hours — matches the design doc

# Toolset the ``status_update`` tool is registered under.
#
# Why ``"todo"`` and not ``"memory"``:
# Aleph's actual DM surface is Discord, and Zoe's ``platform_toolsets.discord``
# in ``~/.hermes/config.yaml`` does NOT include ``memory`` (the built-in
# ``memory`` tool is intentionally gated off for Aleph). If we left
# ``status_update`` under ``"memory"``, it would be invisible on Discord and
# the deterministic ``post_tool_call`` hook would be the *only* write path.
# ``todo`` is in Discord's allowlist, is conceptually adjacent ("operational
# hints for current work"), and is exposed in safe profiles already.
STATUS_UPDATE_TOOLSET = "todo"


def _ttl_seconds() -> float:
    """Resolve the TTL. Env override → config fallback → default."""
    raw = os.environ.get("STATUS_SIDECAR_TTL_SECONDS")
    if raw:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    return float(DEFAULT_TTL_SECONDS)


# ---------------------------------------------------------------------------
# pre_llm_call hook — render the bounded ``<system_status>`` block
# ---------------------------------------------------------------------------

def _on_pre_llm_call(
    session_id: str = "",
    user_message: str = "",
    conversation_history: Optional[list] = None,
    is_first_turn: bool = False,
    model: str = "",
    platform: str = "",
    sender_id: str = "",
    **_: Any,
) -> Optional[Dict[str, str]]:
    """Return a context dict when the ledger has fresh status; else None.

    NEVER raises — any error swallowed and converted to ``None`` (skip
    injection). The agent core also wraps us in try/except, but defence in
    depth: we should not depend on that.
    """
    try:
        block = ss.render_status_block(ttl_seconds=_ttl_seconds())
    except Exception as exc:  # pragma: no cover — render_status_block already swallows
        logger.debug("status_sidecar render failed: %s", exc)
        return None
    if not block:
        return None
    return {"context": block}


# ---------------------------------------------------------------------------
# post_tool_call hook — deterministic ledger updates after observed tools
# ---------------------------------------------------------------------------
#
# Hard rule: no tool *result body* or *args content* ever lands in the
# ledger. The derivation policy is in
# :func:`status_sidecar.derive_status_from_tool_call`; this hook only
# applies its output.
#
# We also track tool-call counts per session for the
# :func:`_on_session_finalize` compact drift signal — that's why we keep a
# small per-session counter dict here. It does NOT store any tool args or
# results, only a count.
# ---------------------------------------------------------------------------

# Bounded counter so a long-lived gateway process never grows the dict
# unboundedly. We evict the smallest sessions when over the soft cap.
_session_counters_lock = threading.Lock()
_session_counters: Dict[str, Dict[str, Any]] = {}
_MAX_TRACKED_SESSIONS = 64


def _bump_session_counter(session_id: str, tool_name: str) -> None:
    if not isinstance(session_id, str):
        return
    sid = session_id or "_unknown_"
    with _session_counters_lock:
        rec = _session_counters.setdefault(sid, {"count": 0, "last_tool": ""})
        rec["count"] = int(rec.get("count", 0)) + 1
        if isinstance(tool_name, str) and tool_name:
            rec["last_tool"] = tool_name
        # Soft-evict if we've blown the cap. Drop the smallest-count sessions
        # first; for ties, drop arbitrarily (insertion order).
        if len(_session_counters) > _MAX_TRACKED_SESSIONS:
            # Pick the lowest-count entry to evict.
            victim = min(
                _session_counters.items(),
                key=lambda kv: (kv[1].get("count", 0), kv[0]),
            )[0]
            if victim != sid:
                _session_counters.pop(victim, None)


def _drain_session_counter(session_id: str) -> Optional[Dict[str, Any]]:
    if not isinstance(session_id, str) or not session_id:
        return None
    with _session_counters_lock:
        return _session_counters.pop(session_id, None)


def _on_post_tool_call(
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    duration_ms: int = 0,
    **_: Any,
) -> None:
    """Apply deterministic ledger updates derived from a tool call.

    NEVER raises. Errors are logged at debug level and swallowed — a
    failing status write must never break the agent loop.
    """
    try:
        # Track everything we see for the session-finalize signal — even
        # ignored tools count, so the agent gets a true "I made N calls"
        # signal.
        _bump_session_counter(session_id, tool_name)

        update = ss.derive_status_from_tool_call(tool_name, args, result)
    except Exception as exc:  # pragma: no cover — derive is pure and safe
        logger.debug("status_sidecar derive failed: %s", exc)
        return

    if not update:
        return

    drift = update.pop("_drift", None)
    try:
        if update:
            # Only call write_status if we have at least one scalar field
            # to set. Empty calls would still bump status_updated_at, which
            # would falsely advertise freshness on no real change.
            ss.write_status(**update)
        if isinstance(drift, str) and drift.strip():
            ss.append_drift_signal(drift)
    except Exception as exc:  # pragma: no cover — write_status swallows internally
        logger.debug("status_sidecar post_tool_call write failed: %s", exc)


# ---------------------------------------------------------------------------
# on_session_finalize hook — compact end-of-session drift signal
# ---------------------------------------------------------------------------

def _on_session_finalize(session_id: str = "", **_: Any) -> None:
    """Append a compact drift signal summarising the session.

    Pure breadcrumb: ``"session ended: N tool calls, last=X"``. We do NOT
    capture user prose, assistant prose, or tool results — only counts and
    the last tool name we observed.

    NEVER raises.
    """
    try:
        counter = _drain_session_counter(session_id)
        if counter is None:
            # No activity tracked this session — finalize is a no-op rather
            # than emitting a misleading "0 calls" signal that would just
            # add noise.
            return
        count = int(counter.get("count", 0))
        last_tool = counter.get("last_tool") or ""
        if count <= 0:
            return
        # Keep the signal extremely short and structured. The drift-signal
        # ring buffer is bounded; we don't need to bound this further.
        signal = f"session ended: {count} tool calls"
        if last_tool:
            signal += f", last={last_tool}"
        ss.append_drift_signal(signal)
    except Exception as exc:  # pragma: no cover — append_drift_signal swallows
        logger.debug("status_sidecar finalize failed: %s", exc)


# ---------------------------------------------------------------------------
# Tool: status_update
# ---------------------------------------------------------------------------

_STATUS_UPDATE_SCHEMA = {
    "name": "status_update",
    "description": (
        "Append a short note to the durable status ledger. Use this ONLY for "
        "*current operational hints* — what you're working on right now, a "
        "drift signal like 'user asked to simplify scope', or the active "
        "workspace path. Not a memory store: entries are length-bounded and "
        "the drift-signal ring buffer keeps only the most recent few. The "
        "ledger is appended to the user message on the next turn via the "
        "status-sidecar plugin's pre_llm_call hook. Most of the ledger is "
        "updated automatically by the plugin's post_tool_call observer — "
        "use this tool only for drift signals or when you want to override "
        "what the observer captured."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "drift_signal": {
                "type": "string",
                "description": (
                    "A short note about a recent change in direction or "
                    "intent (e.g. 'user asked to defer rate limiter', "
                    "'aborted test run, switching to manual'). Truncated "
                    "to 200 chars; only the most recent 5 are kept."
                ),
            },
            "active_kanban_task_id": {
                "type": "string",
                "description": (
                    "The kanban task you're actively working on, e.g. "
                    "'t_8a4caa0e'. Pass empty string to clear."
                ),
            },
            "active_workspace": {
                "type": "string",
                "description": (
                    "Absolute path of the workspace you're currently in. "
                    "Pass empty string to clear."
                ),
            },
            "last_tool_invocation": {
                "type": "string",
                "description": (
                    "Name of the last meaningful tool you ran (e.g. "
                    "'kanban_show', 'pytest'). Optional."
                ),
            },
            "last_cron_job_id": {
                "type": "string",
                "description": (
                    "Last cron-job id referenced, if any. Optional."
                ),
            },
        },
        "additionalProperties": False,
    },
}


def _status_update_handler(args: Dict[str, Any], **_: Any) -> str:
    """Tool handler. Always returns a JSON string.

    Accepts any subset of the schema fields. Returns
    ``{"ok": True, "wrote": [...]}`` on success, or
    ``{"ok": False, "error": "..."}`` on failure.
    """
    if not isinstance(args, dict):
        return json.dumps({"ok": False, "error": "args must be an object"})

    wrote: list = []
    # 1) drift_signal goes through append_drift_signal so the ring buffer
    #    bound is enforced.
    drift = args.get("drift_signal")
    if isinstance(drift, str) and drift.strip():
        if ss.append_drift_signal(drift):
            wrote.append("drift_signal")

    # 2) Other scalar pointers — collapse to a single write_status call.
    scalar_fields = (
        "active_kanban_task_id",
        "active_workspace",
        "last_tool_invocation",
        "last_cron_job_id",
    )
    kwargs: Dict[str, Any] = {}
    for f in scalar_fields:
        v = args.get(f)
        if isinstance(v, str):
            kwargs[f] = v
    if kwargs:
        if ss.write_status(**kwargs):
            wrote.extend(kwargs.keys())

    if not wrote and not drift:
        return json.dumps({"ok": False, "error": "no recognised fields provided"})

    return json.dumps({"ok": True, "wrote": sorted(set(wrote))})


def _status_update_check() -> bool:
    """Always available. No env requirements; the ledger is local."""
    return True


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("on_session_finalize", _on_session_finalize)
    ctx.register_tool(
        name="status_update",
        toolset=STATUS_UPDATE_TOOLSET,
        schema=_STATUS_UPDATE_SCHEMA,
        handler=_status_update_handler,
        check_fn=_status_update_check,
        description=(
            "Append a short operational hint to the status-sidecar ledger."
        ),
    )
