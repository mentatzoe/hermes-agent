"""Flush pending messages and agent transcripts to disk before shutdown to prevent data loss.

When FTS5 index corruption prevents ``INSERT INTO messages``, the gateway
accumulates messages in ``_pending_messages`` (memory-only) and the live
``agent._session_messages`` cannot be flushed via ``_flush_messages_to_session_db``.
On shutdown, ``.clear()`` discards the only surviving copy — permanent user data loss.

This module provides three hooks:

1. ``flush_pending_to_file()`` — called BEFORE ``_pending_messages.clear()``
   during shutdown.  Serialises any non-empty pending slots to a JSON file
   under ``<hermes_home>/pending_messages/``.

2. ``recover_pending_to_db()`` — called AFTER ``runner.start()`` on startup.
   Reads flush files, inserts messages into state.db via ``SessionDB.append_message``
   (so FTS indexing, session metadata, and display_kind are handled correctly),
   then deletes the flush file on success.

3. ``flush_agent_history_to_file()`` — called from ``_finalize_shutdown_agents``
   when ``_flush_messages_to_session_db`` raises.  Dumps the live
   ``agent._session_messages`` to the same atomic JSON recovery directory.

See issue #72680 for the full incident report.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


def _get_flush_dir():
    """Return the pending-messages flush directory under the active HERMES_HOME."""
    from hermes_constants import get_hermes_home

    flush_dir = get_hermes_home() / "pending_messages"
    flush_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        os.chmod(flush_dir, 0o700)
    return flush_dir


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry on platforms that support directory fsync."""
    if os.name != "posix":
        return
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_payload(flush_dir: Path, payload: Dict[str, Any]) -> Path:
    """Atomically write one private, uniquely named recovery payload.

    Returns the path of the published payload file.
    """
    from utils import atomic_json_write

    file_id = uuid.uuid4().hex
    final_path = flush_dir / f"pending-{file_id}.json"
    atomic_json_write(
        final_path,
        payload,
        mode=0o600,
        default=str,
    )

    try:
        _fsync_directory(flush_dir)
    except OSError as exc:
        # The atomically published file is still the only recovery copy.
        # Keep it even if this filesystem cannot persist directory entries.
        logger.debug("Failed to fsync pending-message directory: %s", exc)
    return final_path


def _with_session_id(
    serialised: dict,
    session_key: str,
    session_id_resolver: Optional[Callable[[str], Optional[str]]],
) -> dict:
    """Stamp the routing key's current session id onto a flush payload.

    Queued ``MessageEvent`` objects never carry a session id, so without this
    every flushed inbound message was unrecoverable ("no session_id in flush
    file"); 30 room messages were lost that way on 30 Sep.
    """
    if serialised.get("session_id") or session_id_resolver is None or not session_key:
        return serialised
    try:
        session_id = session_id_resolver(session_key)
    except Exception:
        session_id = None
    if isinstance(session_id, str) and session_id:
        serialised = dict(serialised)
        serialised["session_id"] = session_id
    return serialised


def flush_pending_to_file(
    pending: Dict[str, Any],
    *,
    reason: str = "shutdown",
    session_id_resolver: Optional[Callable[[str], Optional[str]]] = None,
) -> int:
    """Serialise non-empty ``_pending_messages`` slots to disk.

    Parameters
    ----------
    pending:
        The adapter or runner ``_pending_messages`` dict.  Values may be
        ``MessageEvent`` objects (adapter) or plain strings (runner).
    reason:
        Logged context (``shutdown``, ``restart``, etc.).
    session_id_resolver:
        Optional ``session_key -> session_id`` lookup (the runner passes
        ``SessionStore.peek_session_id``) so recovery can file the message
        into the right transcript.

    Returns
    -------
    int
        Number of sessions flushed.
    """
    if not pending:
        return 0

    flush_dir = _get_flush_dir()
    ts = int(time.time())
    flushed = 0

    for session_key, value in list(pending.items()):
        if value is None:
            continue
        try:
            serialised = _serialise_value(value)
            if serialised is None:
                continue
            serialised = _with_session_id(serialised, session_key, session_id_resolver)
            _write_payload(
                flush_dir,
                {
                    "session_key": session_key,
                    "reason": reason,
                    "ts": ts,
                    "data": serialised,
                },
            )
            flushed += 1
        except Exception as exc:
            logger.debug(
                "Failed to flush pending message for %s: %s",
                session_key, exc,
            )

    if flushed:
        logger.info(
            "Flushed %d pending message(s) to %s (reason=%s)",
            flushed, flush_dir, reason,
        )
    return flushed


def flush_overflow_to_file(
    overflow_by_session: Dict[str, Any],
    *,
    reason: str = "shutdown",
    session_id_resolver: Optional[Callable[[str], Optional[str]]] = None,
) -> int:
    """Serialise the FIFO overflow tails (``queued_events``) to disk.

    Sibling of :func:`flush_pending_to_file` for the second half of the
    gateway FIFO (#99882): the adapter slot holds the queue head, and the
    per-session ``SessionState.conversation.queued_events`` list holds the
    tail.  Shutdown flushed only the slot, so every follow-up parked in
    overflow at restart time vanished with the process.  Each overflow
    event is written as its own payload in the same shape as a slot flush
    so ``recover_pending_to_db`` replays them unchanged; a ``seq`` field
    preserves arrival order within a session.

    Returns the number of events flushed.
    """
    if not overflow_by_session:
        return 0

    flush_dir = _get_flush_dir()
    ts = int(time.time())
    flushed = 0

    for session_key, events in list(overflow_by_session.items()):
        if not session_key or not events:
            continue
        for seq, value in enumerate(list(events)):
            if value is None:
                continue
            try:
                serialised = _serialise_value(value)
                if serialised is None:
                    continue
                serialised = _with_session_id(
                    serialised, session_key, session_id_resolver
                )
                _write_payload(
                    flush_dir,
                    {
                        "session_key": session_key,
                        "reason": reason,
                        "ts": ts,
                        "seq": seq,
                        "data": serialised,
                    },
                )
                flushed += 1
            except Exception as exc:
                logger.debug(
                    "Failed to flush overflow message for %s: %s",
                    session_key, exc,
                )

    if flushed:
        logger.info(
            "Flushed %d queued overflow message(s) to %s (reason=%s)",
            flushed, flush_dir, reason,
        )
    return flushed


# Reason tag for transcript messages dropped by the in-memory pending cap
# during live operation (#78182). These payloads carry the full transcript
# message dict so they can be replayed verbatim once the DB recovers.
TRANSCRIPT_CAP_DROP_REASON = "transcript_cap_drop"


def spool_dropped_transcript_message(
    session_id: str,
    message: Dict[str, Any],
) -> Optional[Path]:
    """Spool a transcript message evicted by the runtime pending cap.

    Uses the same on-disk pending spool as :func:`flush_pending_to_file`
    (one atomic JSON payload per message under
    ``<hermes_home>/pending_messages/``), so a runtime cap rotation no
    longer silently discards user data while the process stays up
    (#78182).

    Returns the written spool path, or ``None`` when spooling failed —
    callers must degrade to the previous drop-and-log behaviour.
    """
    try:
        flush_dir = _get_flush_dir()
        return _write_payload(
            flush_dir,
            {
                "session_key": session_id,
                "reason": TRANSCRIPT_CAP_DROP_REASON,
                "ts": int(time.time()),
                "seq": next(_TRANSCRIPT_SPOOL_SEQ),
                "data": {
                    "session_id": session_id,
                    "message": message,
                },
            },
        )
    except Exception as exc:
        logger.debug(
            "Failed to spool cap-dropped transcript message for %s: %s",
            session_id, exc,
        )
        return None


# Monotonic tiebreaker so same-second spool files replay in drop order.
_TRANSCRIPT_SPOOL_SEQ = itertools.count()


def drain_transcript_spool(session_id: str, replay) -> tuple[int, int]:
    """Replay cap-dropped transcript messages spooled for *session_id*.

    ``replay(message_dict)`` is invoked for each spooled message in drop
    order; the spool file is deleted only after a successful replay.  On
    the first replay failure the drain stops and remaining files are kept
    for the next attempt (the DB is likely still unhealthy).

    Returns ``(replayed, remaining)`` — messages replayed and spool files
    left behind for a later retry.
    """
    try:
        flush_dir = _get_flush_dir()
        candidates = list(flush_dir.glob("pending-*.json"))
    except Exception as exc:
        logger.debug("Cannot scan transcript spool: %s", exc)
        return 0, 0

    entries = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("reason") != TRANSCRIPT_CAP_DROP_REASON:
            continue
        if payload.get("session_key") != session_id:
            continue
        message = (payload.get("data") or {}).get("message")
        if not isinstance(message, dict):
            logger.warning(
                "Removing structurally invalid transcript spool file %s", path,
            )
            path.unlink(missing_ok=True)
            continue
        entries.append(
            (payload.get("ts", 0), payload.get("seq", 0), path.name, path, message)
        )

    replayed = 0
    ordered = sorted(entries, key=lambda e: e[:3])
    remaining = 0
    for idx, (_ts, _seq, _name, path, message) in enumerate(ordered):
        try:
            replay(message)
        except Exception as exc:
            logger.warning(
                "Replay of spooled transcript message %s for %s failed; "
                "keeping spool file for retry: %s",
                path, session_id, exc,
            )
            remaining = len(ordered) - idx
            break
        path.unlink(missing_ok=True)
        replayed += 1

    if replayed:
        logger.info(
            "Replayed %d spooled transcript message(s) for %s after DB recovery",
            replayed, session_id,
        )
    return replayed, remaining


def _serialise_value(value: Any) -> Optional[dict]:
    """Convert a pending message value to a JSON-serialisable dict."""
    # MessageEvent objects have a .text attribute and other fields
    if hasattr(value, "text"):
        result: Dict[str, Any] = {"text": getattr(value, "text", "")}
        # Preserve additional fields if present
        for attr in ("session_id", "platform", "sender_id", "sender_name",
                      "reply_to", "media", "raw_event"):
            val = getattr(value, attr, None)
            if val is not None:
                try:
                    json.dumps(val)
                    result[attr] = val
                except (TypeError, ValueError):
                    result[attr] = str(val)
        # The gateway's real MessageEvent has none of the legacy attribute
        # names above: its sender lives on ``user_id``/``user_name`` or
        # ``source``, and it never carries a session id.  Record what it does
        # have so recovery can attribute the row (30 Sep: 30 flushed room
        # messages were unrecoverable).
        source = getattr(value, "source", None)
        for key, candidates in (
            ("sender_id", (getattr(value, "user_id", None),
                           getattr(source, "user_id", None))),
            ("sender_name", (getattr(value, "user_name", None),
                             getattr(source, "user_name", None))),
            ("message_id", (getattr(value, "message_id", None),)),
        ):
            if key in result:
                continue
            for candidate in candidates:
                if isinstance(candidate, (str, int)) and str(candidate):
                    result[key] = str(candidate)
                    break
        if "sender_is_bot" not in result and isinstance(
            getattr(source, "is_bot", None), bool
        ):
            result["sender_is_bot"] = source.is_bot
        chat_type = getattr(source, "chat_type", None)
        if "chat_type" not in result and isinstance(chat_type, str) and chat_type:
            result["chat_type"] = chat_type
        # Recovery renders the sender label with the live
        # ``format_sender_label``, which needs the platform (Slack keeps the
        # verifiable user id) and must leave gateway-internal notices
        # unlabelled.
        platform = getattr(source, "platform", None)
        platform = getattr(platform, "value", platform)
        if "platform" not in result and isinstance(platform, str) and platform:
            result["platform"] = platform
        if getattr(value, "internal", False) is True:
            result["internal"] = True
        return result
    # Plain string (runner-level _pending_messages)
    if isinstance(value, str):
        return {"text": value}
    # Dict — try direct serialisation
    if isinstance(value, dict):
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return {"text": str(value)}
    return {"text": str(value)}


# Flushed inbound messages older than this are not injected into a
# transcript: a restart hours later must not surface a stale backlog as if it
# were current input.  Their files are kept for manual inspection.
RECOVERY_MAX_AGE_SECONDS = 6 * 60 * 60


def _flush_order_key(path: Path) -> tuple:
    """Sort key giving arrival order for flush files (ts, seq, name)."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        ts = payload.get("ts")
        seq = payload.get("seq")
        return (
            ts if isinstance(ts, (int, float)) else 0,
            seq if isinstance(seq, int) else -1,
            path.name,
        )
    except Exception:
        return (0, -1, path.name)


def _recovered_user_content(data: Dict[str, Any], text: str) -> str:
    """Label a recovered inbound message with its sender.

    Every live delivery path labels shared-session turns with
    ``gateway.session.format_sender_label``: peer bots are marked
    ``[Name [bot]]``, brackets in a display name are neutralised so it cannot
    forge a second speaker, and Slack keeps the user id.  A message recovered
    after a restart must carry the same label, or a peer bot's queued words
    would read as the operator's.  Gateway-internal notices stay unlabelled.
    """
    name = data.get("sender_name")
    if not isinstance(name, str) or not name.strip():
        return text
    if data.get("internal") is True:
        return text
    # Label only when the flush recorded a shared (non-DM) chat; unknown or
    # legacy payloads keep their text verbatim.
    if data.get("chat_type") not in ("group", "channel", "thread"):
        return text
    platform = data.get("platform")
    try:
        from gateway.config import Platform

        platform = Platform(platform) if isinstance(platform, str) and platform else None
    except Exception:
        platform = None
    sender = SimpleNamespace(
        user_name=name,
        user_id=data.get("sender_id"),
        platform=platform,
        is_bot=data.get("sender_is_bot") is True,
    )
    try:
        from gateway.session import format_sender_label

        label = format_sender_label(sender)
    except Exception:
        # Same neutralisation as format_sender_label, minus the Slack id.
        safe = " ".join(name.split())[:64].replace("[", "(").replace("]", ")")
        label = f"[{safe}{' [bot]' if sender.is_bot else ''}]" if safe else None
    if not label:
        return text
    prefix = f"{label} "
    return text if text.startswith(prefix) else prefix + text


def recover_pending_to_db(
    session_db=None,
    *,
    session_id_resolver: Optional[Callable[[str], Optional[str]]] = None,
    max_age_seconds: Optional[float] = RECOVERY_MAX_AGE_SECONDS,
) -> int:
    """Recover flushed pending messages into state.db via SessionDB.

    Reads all ``*.json`` files from the flush directory, inserts messages
    using ``SessionDB.append_message`` (so FTS indexing, session metadata
    updates, and all required columns are handled correctly), and deletes
    the flush file on success.

    Parameters
    ----------
    session_db:
        An existing ``SessionDB`` instance.  If ``None``, a new one is
        opened on the default ``state.db`` path.
    session_id_resolver:
        Optional ``session_key -> session_id`` lookup used when a flush file
        predates session-id stamping (the gateway passes
        ``SessionStore.peek_session_id``).
    max_age_seconds:
        Flush files older than this are skipped and preserved.  ``None`` or
        ``0`` disables the age check.

    Returns
    -------
    int
        Number of messages recovered.
    """
    flush_dir = _get_flush_dir()
    # Replay in arrival order: flush files carry random names, so sort by
    # flush time, then the per-session ``seq`` written for FIFO overflow.
    flush_files = sorted(flush_dir.glob("*.json"), key=_flush_order_key)
    if not flush_files:
        return 0

    # Use the provided SessionDB or open one on the default path.
    own_db = False
    if session_db is None:
        from hermes_state import get_shared_session_db
        session_db = get_shared_session_db()
        own_db = True

    def _close_owned_db() -> None:
        if not own_db:
            return
        try:
            from hermes_state import release_or_close
            release_or_close(session_db)
        except Exception:
            pass

    recovered = 0
    for path in flush_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            # Agent-history snapshots use a different schema (reason +
            # messages list) and are meant for manual operator recovery,
            # not automatic DB insertion. Skip them silently.
            if payload.get("reason") == "shutdown-with-unpersisted-agent-history":
                continue
            # Cap-dropped transcript payloads carry the full message dict
            # keyed by session_id — replay directly (#78182). This handles
            # spool files that were never drained before a restart.
            if payload.get("reason") == TRANSCRIPT_CAP_DROP_REASON:
                data = payload.get("data", {}) or {}
                spooled_sid = data.get("session_id", "")
                message = data.get("message")
                if not spooled_sid or not isinstance(message, dict):
                    logger.warning(
                        "Cannot recover structurally invalid transcript spool "
                        "file %s; preserved for manual inspection",
                        path,
                    )
                    continue
                session_db.append_message(
                    session_id=spooled_sid,
                    role=message.get("role", "unknown"),
                    content=message.get("content") or "",
                    timestamp=message.get("timestamp") or payload.get("ts"),
                )
                recovered += 1
                path.unlink(missing_ok=True)
                continue
            session_key = payload.get("session_key", "")
            data = payload.get("data", {})
            text = data.get("text", "")
            if not text or not session_key:
                logger.warning(
                    "Cannot recover structurally invalid pending message from %s; "
                    "the flush file has been preserved",
                    path,
                )
                continue

            flushed_at = payload.get("ts")
            if (
                max_age_seconds
                and isinstance(flushed_at, (int, float))
                and time.time() - flushed_at > max_age_seconds
            ):
                logger.warning(
                    "Not recovering pending message for %s: flushed %.1f h ago "
                    "(limit %.1f h), too stale to add to the transcript. "
                    "Preserved in %s",
                    session_key,
                    (time.time() - flushed_at) / 3600,
                    max_age_seconds / 3600,
                    path,
                )
                continue

            # The session_key is a gateway routing key (e.g.
            # "agent:main:telegram:supergroup:...").  We need the actual
            # session_id (e.g. "20260728_120000_abc123") to append a
            # message row.  Flushes stamp it at shutdown; for older files
            # ask the caller's resolver (SessionStore.peek_session_id).
            session_id = data.get("session_id", "")
            if not session_id and session_id_resolver is not None:
                try:
                    resolved = session_id_resolver(session_key)
                except Exception:
                    resolved = None
                if isinstance(resolved, str) and resolved:
                    session_id = resolved

            if not session_id:
                logger.warning(
                    "Cannot recover pending message for %s: no session_id "
                    "in flush file and no session is bound to that key. "
                    "The message text is preserved in %s",
                    session_key, path,
                )
                continue

            session_db.append_message(
                session_id=session_id,
                role="user",
                content=_recovered_user_content(data, text),
                timestamp=payload.get("ts", int(time.time())),
            )
            recovered += 1
            path.unlink(missing_ok=True)
        except BaseException:
            # Shutdown cancellation/interrupt must not strand an owned DB.
            _close_owned_db()
            raise
        except Exception as exc:
            logger.warning(
                "Failed to recover pending message from %s: %s",
                path, exc,
            )
            # Leave the file for next startup retry.

    _close_owned_db()

    if recovered:
        logger.info(
            "Recovered %d pending message(s) from shutdown flush", recovered,
        )
    return recovered


def flush_agent_history_to_file(
    session_id: Optional[str],
    history: list,
) -> None:
    """Best-effort dump of an agent's in-memory transcript before teardown.

    Used when ``_flush_messages_to_session_db`` raises (e.g. FTS/SQLite
    index corruption, #72680): the live ``agent._session_messages`` could
    not be written to disk, and a plain debug log would lose it permanently
    when the process exits. Serialize to an atomic JSON file outside the
    broken DB so an operator can salvage the conversation after repairing
    state.db.

    Failures are swallowed — shutdown must never block on a best-effort
    backup.
    """
    if not history:
        return
    try:
        flush_dir = _get_flush_dir()
        snapshot = []
        for _m in history:
            try:
                snapshot.append(
                    _m if isinstance(_m, (dict, list, str, int, float, bool, type(None)))
                    else str(_m)
                )
            except Exception:
                continue
        _write_payload(
            flush_dir,
            {
                "reason": "shutdown-with-unpersisted-agent-history",
                "issue": "#72680",
                "session_id": session_id,
                "count": len(snapshot),
                "messages": snapshot,
            },
        )
        logger.warning(
            "Preserved %d in-memory message(s) for session %s "
            "(possible FTS corruption — recover after repairing state.db)",
            len(snapshot),
            session_id,
        )
    except Exception as _e:
        logger.warning(
            "Agent-history shutdown preservation failed for session %s: %s",
            session_id, _e,
        )
