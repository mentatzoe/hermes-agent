"""Best-effort wake lists for notices suppressed during restart, not notice replay.

Only kind/id/time and the routing key cross the restart. The board and process
records remain authoritative; losing or duplicating this list is acceptable.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re

logger = logging.getLogger(__name__)
_IDENTIFIERS = re.compile(r"\b(proc_|t_|deleg_)[A-Za-z0-9_-]+\b")
_KINDS = {"proc_": "process", "t_": "kanban", "deleg_": "delegation"}


def _path(runner) -> Path:
    # The host session index includes all served profiles. Never use ambient
    # get_hermes_home(): intake may run inside a routed profile's scope.
    return Path(runner.session_store.sessions_dir).parent / "restart-notice-summary.jsonl"


def _items(event) -> list[dict]:
    metadata = event.metadata or {}
    explicit = metadata.get("restart_notice")
    if isinstance(explicit, dict):
        return [{"kind": str(explicit.get("kind") or "internal"), "id": str(explicit.get("id") or "unknown")}]
    # Existing producer text carries these stable handles, including all members
    # of a coalesced batch. Do not retain result text, commands or tool output.
    items = [{"kind": _KINDS[match.group(1)], "id": match.group(0)}
             for match in _IDENTIFIERS.finditer(event.text or "")]
    if items:
        return items
    job = metadata.get("cron_job_id") or metadata.get("job_id")
    return [{"kind": "cron" if job else "internal", "id": str(job or event.message_id or "unknown")}]


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def suppress_restart_notice(runner, event, session_key: str) -> bool:
    """Suppress before queueing or replies, even when saving the optional list fails."""
    if (runner is None or not event.internal
            or not (getattr(runner, "_restart_requested", False) is True
                    or getattr(runner, "_draining", False) is True)):
        return False
    if not getattr(event, "_restart_summary_captured", False):
        event._restart_summary_captured = True
        try:
            _append(_path(runner), {"session_key": session_key, "items": _items(event),
                                   "time": datetime.now(timezone.utc).isoformat()})
        except Exception:
            logger.warning("Could not save optional restart notice summary; notice stays silent", exc_info=True)
    # The producer may retire this wake. This is not execution acknowledgement.
    event._gateway_accepted = True
    return True


def intercept_notice(adapter, event, session_key: str, expected_session_key: str) -> bool:
    """Keep the existing routing-pin check before recording or queueing anything."""
    if expected_session_key and session_key != expected_session_key:
        logger.warning("Dropping internally routed event: expected session=%s derived=%s",
                       expected_session_key, session_key)
        return True
    return suppress_restart_notice(adapter.gateway_runner, event, session_key)


def _take(runner) -> dict[str, list]:
    path = _path(runner)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8-sig")
    path.unlink()  # No receipt/retry journal: this wake is deliberately best-effort.
    groups: dict[str, list] = {}
    for line in text.splitlines():
        try:
            row = json.loads(line)
            groups.setdefault(row["session_key"], []).append(row)
        except (ValueError, KeyError, TypeError):
            logger.warning("Ignoring an incomplete restart notice summary row")
    return groups


async def _wake(runner, key: str, rows: list) -> None:
    from gateway.platforms.event import MessageEvent
    from gateway.wake import admit_internal_event

    entry = await runner.async_session_store.lookup_by_session_key(key)
    if entry is None:
        return
    source = runner._restored_source(entry)
    adapter = runner._delivery_adapter_for(source)
    if adapter is None or not runner._resume_owner_authorized(key, source):
        return
    lines = [f"- {item['kind']}: {item['id']} — {row['time']} UTC" for row in rows for item in row["items"]]
    event = MessageEvent(
        text=("Internal restart wake: these notices arrived while the gateway was restarting.\n"
              + "\n".join(lines) + "\n\nCheck the Kanban board and process/delegation records for current results and unfinished work. "
              "This is a best-effort list, not redelivery or proof that anything completed. "
              "Do not acknowledge this list in the chat; reply only if the actual result needs the user's attention."),
        source=source, internal=True, allow_gateway_control=False,
        metadata={"gateway_session_key": key, "gateway_session_id": entry.session_id},
    )
    async with runner._async_profile_scope_for_source(source):
        prime = getattr(adapter, "prime_routing_cache", None)
        if callable(prime):
            prime(event)
        await admit_internal_event(adapter, event)


async def wake_after_restart(runner) -> None:
    """One internal summary per affected session after the startup gate opens."""
    if getattr(runner, "_draining", False) or getattr(runner, "_restart_requested", False):
        return
    try:
        groups = _take(runner)
    except Exception:
        logger.warning("Could not read optional restart notice summary", exc_info=True)
        return
    for key, rows in groups.items():
        try:
            await _wake(runner, key, rows)
        except Exception:
            logger.warning("Optional restart wake was not accepted for %s", key, exc_info=True)
