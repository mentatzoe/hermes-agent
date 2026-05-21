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
from typing import Any, Dict, Optional

from . import status_sidecar as ss

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_TTL_SECONDS = 4 * 3600  # 4 hours — matches the design doc


def _ttl_seconds() -> float:
    """Resolve the TTL. Env override → config fallback → default."""
    raw = os.environ.get("STATUS_SIDECAR_TTL_SECONDS")
    if raw:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    # No reach into hermes_cli.config here — keep the hook cheap and
    # crash-proof. Operators who want a non-default TTL set the env var.
    return float(DEFAULT_TTL_SECONDS)


# ---------------------------------------------------------------------------
# Hook
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
        "status-sidecar plugin's pre_llm_call hook."
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
    ctx.register_tool(
        name="status_update",
        toolset="memory",  # sits near memory tools — not auto-loaded for safe profiles
        schema=_STATUS_UPDATE_SCHEMA,
        handler=_status_update_handler,
        check_fn=_status_update_check,
        description=(
            "Append a short operational hint to the status-sidecar ledger."
        ),
    )
