"""Durable handoff of internal messaging notices across gateway restart drains.

The gateway's existing per-home runtime lock owns replay. SQLite commits precede
producer admission; an in-process replay lock and inflight set prevent duplicate
scheduling. Rows settle after the message handler returns, not after scheduling.
An unexpected crash during a handler is retryable (at-least-once execution), as
with the other gateway completion paths; this is not an exactly-once tool ledger.
"""

from __future__ import annotations

import asyncio
from contextlib import closing
import json
import logging
import os
from pathlib import Path
import sqlite3
import uuid

from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.session_identity import restore_identity, transport_profile_of

logger = logging.getLogger(__name__)


def _connect(path: Path):
    from hermes_constants import assert_named_profile_home_live
    assert_named_profile_home_live(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(fd)
    conn = sqlite3.connect(path, timeout=5)
    conn.execute("CREATE TABLE IF NOT EXISTS notices (id TEXT PRIMARY KEY, payload TEXT NOT NULL, done INTEGER NOT NULL DEFAULT 0)")
    return conn


def _save(path: Path, notice_id: str, payload: dict) -> None:
    encoded = json.dumps(payload, ensure_ascii=False)
    with closing(_connect(path)) as conn, conn:
        conn.execute("INSERT OR IGNORE INTO notices (id, payload) VALUES (?, ?)", (notice_id, encoded))


def _pending(path: Path) -> list:
    if not path.is_file():
        return []
    with closing(_connect(path)) as conn:
        return [(notice_id, json.loads(payload)) for notice_id, payload in
                conn.execute("SELECT id, payload FROM notices WHERE done = 0 ORDER BY rowid")]


def _complete(path: Path, notice_id: str) -> None:
    with closing(_connect(path)) as conn, conn:
        # Keep only the receipt, not the delivered private notice text.
        conn.execute("UPDATE notices SET done = 1, payload = '{}' WHERE id = ?", (notice_id,))


def _inflight(runner) -> set:
    inflight = getattr(runner, "_restart_notification_inflight", None)
    if inflight is None:
        inflight = runner._restart_notification_inflight = set()
    return inflight


def _draining(runner) -> bool:
    return (getattr(runner, "_draining", False) is True
            or getattr(runner, "_restart_requested", False) is True)


async def defer_during_restart(adapter, event: MessageEvent, session_key: str) -> bool:
    """Transfer a notice to disk before acknowledging it; humans keep normal dispatch."""
    runner = getattr(adapter, "gateway_runner", None)
    if not event.internal or runner is None or not _draining(runner):
        return False
    from gateway.wake import WakeNotAccepted
    try:
        entry = await asyncio.to_thread(runner.session_store.lookup_by_session_key, session_key)
        metadata = dict(event.metadata or {})
        session_id = metadata.get("gateway_session_id") or (entry.session_id if entry else None)
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("internal notice has no owning session")
        metadata.update(gateway_session_key=session_key, gateway_session_id=session_id)
        home = Path(runner._resolve_profile_home_for_source(event.source))
        path = home / "restart_notifications" / "pending.sqlite3"
        notice_id = getattr(event, "_restart_notification_id", None) or uuid.uuid4().hex
        event._restart_notification_id = notice_id
        payload = {
            "source": event.source.to_dict(), "transport_profile": transport_profile_of(event.source),
            "relay_transport": event.source.delivered_via_upstream_relay is True,
            "text": event.text, "metadata": metadata, "message_type": event.message_type.value,
            "allow_gateway_control": event.allow_gateway_control, "reply_expected": event.reply_expected,
            "media_urls": event.media_urls, "media_types": event.media_types,
            "auto_skill": event.auto_skill, "channel_prompt": event.channel_prompt,
        }
        await asyncio.to_thread(_save, path, notice_id, payload)
        event._restart_notification_id = notice_id
        event._restart_notification_path = path
        event._restart_notification_deferred = True
        _inflight(runner).discard((path, notice_id))
        return True
    except Exception as exc:
        logger.warning("Could not retain internal notice during restart: %s", exc)
        raise WakeNotAccepted("internal notice restart handoff failed") from exc


async def defer_or_reject(adapter, event, session_key, expected_session_key) -> bool:
    """Shared explicit-route check followed by the restart admission boundary."""
    if expected_session_key and session_key != expected_session_key:
        logger.warning("Dropping internally routed event: expected session=%s derived=%s",
                       expected_session_key, session_key)
        return True
    if await defer_during_restart(adapter, event, session_key):
        event._gateway_accepted = True
        return True
    return False


def _queued(runner, adapter, event, key) -> bool:
    if getattr(adapter, "_pending_messages", {}).get(key) is event:
        return True
    overflow = runner._overflow_queue(key)
    return overflow is not None and any(item is event for item in overflow)


def _settled(runner) -> set:
    settled = getattr(runner, "_restart_notification_settled", None)
    if settled is None:
        settled = runner._restart_notification_settled = set()
    return settled


async def complete_notice(runner, event) -> None:
    """Record consumption at the real agent boundary, including in-band FIFO followups."""
    path = getattr(event, "_restart_notification_path", None)
    notice_id = getattr(event, "_restart_notification_id", None)
    if path and notice_id:
        await asyncio.to_thread(_complete, path, notice_id)
        receipt = path, notice_id
        # A replay pass may already hold an older done=0 SQLite snapshot.
        _settled(runner).add(receipt)
        _inflight(runner).discard(receipt)


async def dispatch_message(adapter, event: MessageEvent):
    """Recheck after profile-scope awaits too; a runner refusal is not consumption."""
    key = adapter._event_session_key(event)
    if await defer_during_restart(adapter, event, key):
        return None
    runner = getattr(adapter, "gateway_runner", None)
    path = getattr(event, "_restart_notification_path", None)
    notice_id = getattr(event, "_restart_notification_id", None)
    receipt = (path, notice_id) if path and notice_id else None
    event._restart_notification_deferred = False
    try:
        response = await adapter._message_handler(event)
        executed = getattr(event, "_heartbeat_execution_started", False) is True
        if event.internal and not executed and await defer_during_restart(adapter, event, key):
            return None
        if receipt and not event._restart_notification_deferred:
            if executed:
                await complete_notice(runner, event)
            elif not _queued(runner, adapter, event, key):
                _inflight(runner).discard(receipt)
                return None  # capacity/routing refusal: keep the row retryable, without a chat reply
        return response
    except BaseException:
        if receipt:
            _inflight(runner).discard(receipt)
        raise


async def preserve_pending_notices(runner, adapter, session_key=None) -> None:
    """Transfer already-admitted queue heads and FIFO tails before a drain can discard them."""
    if adapter is None or not _draining(runner):
        return
    keys = {session_key} if session_key else set(adapter._pending_messages) | set(getattr(runner, "_queued_events", {}) or {})
    for key in keys:
        head = adapter._pending_messages.get(key)
        overflow = runner._overflow_queue(key)
        if head is None and overflow and runner._delivery_adapter_for(overflow[0].source) is not adapter:
            continue  # another transport must save its head before its FIFO tail
        events = [head, *list(overflow or [])]
        for event in events:
            if event is None or not event.internal:
                continue
            if await defer_during_restart(adapter, event, key):
                if adapter._pending_messages.get(key) is event:
                    adapter._pending_messages.pop(key)
                if overflow is not None:
                    overflow[:] = [item for item in overflow if item is not event]


def _journal_paths(runner) -> list[Path]:
    # SessionStore captures the gateway's routing home before per-turn scopes exist.
    homes = {Path(runner.session_store.sessions_dir).parent}
    if getattr(runner.config, "multiplex_profiles", False):
        from gateway.run import _multiplex_profile_homes
        homes.update(Path(home) for _profile, home in _multiplex_profile_homes(runner.config))
    return [home / "restart_notifications" / "pending.sqlite3" for home in sorted(homes)]


async def _replay_one(runner, path: Path, notice_id: str, payload: dict) -> None:
    receipt = path, notice_id
    if receipt in _inflight(runner) or receipt in _settled(runner):
        return
    source = SessionSource.from_dict(payload["source"])
    source.delivered_via_upstream_relay = payload.get("relay_transport") is True
    restore_identity(source, runner=runner, transport_profile=payload["transport_profile"])
    key = payload["metadata"]["gateway_session_key"]
    if runner._session_key_for_source(source) != key:
        logger.warning("Retaining restart notice %s: session routing changed", notice_id)
        return
    adapter = runner._delivery_adapter_for(source)
    if (adapter is None or not adapter._message_handler
            or key in adapter._active_sessions or runner._is_session_running(key)):
        return
    event = MessageEvent(
        text=payload["text"], source=source, internal=True,
        message_type=MessageType(payload["message_type"]), metadata=payload["metadata"],
        allow_gateway_control=payload["allow_gateway_control"], reply_expected=payload["reply_expected"],
        media_urls=payload["media_urls"], media_types=payload["media_types"],
        auto_skill=payload["auto_skill"], channel_prompt=payload["channel_prompt"],
    )
    event._restart_notification_path = path
    event._restart_notification_id = notice_id
    _inflight(runner).add(receipt)
    try:
        prime = getattr(adapter, "prime_routing_cache", None)
        if callable(prime):
            prime(event)
        from gateway.wake import admit_internal_event
        await admit_internal_event(adapter, event)
    except BaseException:
        _inflight(runner).discard(receipt)
        raise


async def replay_pending(runner) -> None:
    """Retry retained notices after startup restoration; unavailable adapters keep their rows."""
    if _draining(runner) or getattr(runner, "_startup_restore_in_progress", False):
        return
    lock = getattr(runner, "_restart_notification_replay_lock", None)
    if lock is None:
        lock = runner._restart_notification_replay_lock = asyncio.Lock()
    async with lock:
        for path in _journal_paths(runner):
            for notice_id, payload in await asyncio.to_thread(_pending, path):
                if _draining(runner):
                    return
                try:
                    await _replay_one(runner, path, notice_id, payload)
                except Exception:
                    logger.warning("Restart notice %s remains pending", notice_id, exc_info=True)


async def resolve_restart_session(runner, event):
    """Use the exact saved session or its verified compression tip, never an idle replacement."""
    metadata = event.metadata
    entry = await runner.async_session_store.lookup_by_session_key(metadata["gateway_session_key"])
    pinned = metadata["gateway_session_id"]
    if entry is not None and entry.session_id == pinned:
        return entry
    if entry is not None and runner._session_db is not None:
        row = await runner._session_db.get_session(pinned)
        if row and row.get("ended_at") and row.get("end_reason") == "compression":
            target = await runner._resolve_compression_lineage_target(runner._session_db, entry, pinned)
            if target == entry.session_id:
                return entry
    logger.warning("Retaining restart notice %s: its owning session is no longer current", event._restart_notification_id)
    return None


async def watch_restart_notifications(runner) -> None:
    """One supervised consumer: reconnects can make a retained destination available later."""
    while runner._running and not runner._shutdown_event.is_set():
        await replay_pending(runner)
        await asyncio.sleep(5)
