"""Companion compaction: a recent saved copy of the memory provider's context.

The summary writer reads the memory provider's context (the Honcho peer cards) through
``memory_context``. Three paths compact with none: the gateway's hygiene helper is built
without a memory provider, Honcho turns itself off for cron, Kanban and webhook sessions, and
its card fetch gives up after two seconds. Between 29 Sep and 18 Oct 2026 the cards reached
24 of 104 companion compactions, and the three letters that misgendered Zoe were all written
without them (aleph-vault#202).

So companion compaction keeps the last provider context that arrived and stands it in, labelled
with its age, when a compaction gets none. The provider itself deliberately never hands a late
fetch to a different session, because one provider object is rebound across sessions. This copy
lives in the host instead: in the active profile's home, and used only by a compaction for the
same agent and user, so it never crosses profiles or people.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SAVED_RELPATH = Path("state") / "companion-compaction-provider-context.json"
# Peer cards are standing facts: across 24 compactions from 29 Sep to 16 Oct 2026 the user card
# never changed and the AI card changed once. A week-old copy would have covered 78 of the 80
# compactions that had none; a 72-hour limit, 73.
_DEFAULT_MAX_AGE_HOURS = 168.0


def _saved_path() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / _SAVED_RELPATH


def _who(companion: Any) -> str:
    """Whom the saved context describes; a different agent or user voids it."""
    return hashlib.sha256(f"{companion.agent_name}\n{companion.user_name}".encode("utf-8")).hexdigest()


def _max_age_seconds(companion: Any) -> float:
    raw = getattr(companion, "saved_context_max_age_hours", None)
    try:
        hours = _DEFAULT_MAX_AGE_HOURS if raw is None else float(raw)
    except (TypeError, ValueError):
        logger.warning("companion: saved_context_max_age_hours %r is not a number; using %s",
                       raw, _DEFAULT_MAX_AGE_HOURS)
        hours = _DEFAULT_MAX_AGE_HOURS
    return max(0.0, hours) * 3600


def _remember(companion: Any, memory_context: str) -> None:
    from agent.context_engine import sanitize_memory_context
    from utils import atomic_json_write

    record = {"saved_at": time.time(), "who": _who(companion),
              "context": sanitize_memory_context(memory_context)}
    atomic_json_write(_saved_path(), record, mode=0o600)


def _recall(companion: Any) -> Optional[str]:
    from datetime import datetime

    from hermes_time import get_timezone

    path = _saved_path()
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8-sig"))
    context = str(record.get("context") or "").strip()
    saved_at = float(record.get("saved_at") or 0)
    age = time.time() - saved_at
    if not context or record.get("who") != _who(companion) or not 0 <= age <= _max_age_seconds(companion):
        return None
    saved_on = datetime.fromtimestamp(saved_at, tz=get_timezone()).strftime("%Y-%m-%d %H:%M %Z").strip()
    old = f"{int(age // 60)} min" if age < 3600 else f"{age / 3600:.0f} h"
    return f"[Saved copy from {saved_on}, {old} old; the memory provider sent nothing this time.]\n{context}"


def companion_memory_context(companion: Any, memory_context: str) -> str:
    """The provider context for a companion compaction: save a fresh one, or stand in a recent copy.

    Non-companion compressors, and companion mode with ``saved_context_max_age_hours: 0``,
    get ``memory_context`` back unchanged. Reading or writing the copy never blocks compaction.
    """
    if companion is None or not getattr(companion, "enabled", False) or _max_age_seconds(companion) <= 0:
        return memory_context
    try:
        if memory_context and memory_context.strip():
            _remember(companion, memory_context)
            return memory_context
        return _recall(companion) or memory_context
    except Exception:  # never block compaction on the saved copy
        logger.warning("companion: saved provider context unavailable", exc_info=True)
        return memory_context
