"""A durable admission/completion journal for internal messaging notices.

Each producer occurrence has its own row, even in a coalesced turn. A turn marks
only its owned rows before draining later work. Teardown never reconstructs
ownership from task/queue positions. Finished-but-unmarked turns remain retryable;
this is not an exactly-once ledger for tool effects.
"""

from __future__ import annotations

import asyncio
from contextlib import closing
import json
import logging
import os
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING
import uuid

from gateway.session import SessionSource
from gateway.session_identity import restore_identity, transport_profile_of

if TYPE_CHECKING:
    from gateway.platforms.event import MessageEvent

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


def _save(path: Path, notice_id: str, payload: dict) -> tuple:
    """Atomically admit individual occurrences; preserve coalescing of the new ones."""
    base = {key: value for key, value in payload.items() if key != "_notice_rows"}
    rows = payload.get("_notice_rows") or [(notice_id, payload["text"])]
    inserted = []
    with closing(_connect(path)) as conn, conn:
        for identity, text in rows:
            encoded = json.dumps({**base, "text": text}, ensure_ascii=False)
            result = conn.execute("INSERT OR IGNORE INTO notices (id, payload) VALUES (?, ?)", (identity, encoded))
            if result.rowcount == 1:
                inserted.append(identity)
        batch_text = (payload["text"] if len(inserted) == len(rows) else
                      "\n\n".join(text for identity, text in rows if identity in inserted))
        batch = {"batch_id": uuid.uuid4().hex, "batch_members": inserted, "batch_text": batch_text}
        for identity, text in rows:
            if identity in inserted:
                encoded = json.dumps({**base, **batch, "text": text}, ensure_ascii=False)
                conn.execute("UPDATE notices SET payload = ? WHERE id = ?", (encoded, identity))
    return tuple(inserted)


def _pending(path: Path) -> list:
    if not path.is_file():
        return []
    with closing(_connect(path)) as conn:
        return [(notice_id, json.loads(payload)) for notice_id, payload in
                conn.execute("SELECT id, payload FROM notices WHERE done = 0 ORDER BY rowid")]


def _is_pending(path: Path, notice_id: str) -> bool:
    with closing(_connect(path)) as conn:
        row = conn.execute("SELECT done FROM notices WHERE id = ?", (notice_id,)).fetchone()
    return row is not None and row[0] == 0


def _complete(path: Path, notice_ids) -> None:
    identities = (notice_ids,) if isinstance(notice_ids, str) else notice_ids
    with closing(_connect(path)) as conn, conn:
        # Keep only receipts, not delivered private text; the owned set settles atomically.
        conn.executemany("UPDATE notices SET done = 1, payload = '{}' WHERE id = ?",
                         ((identity,) for identity in identities))


def _inflight(runner) -> set:
    inflight = getattr(runner, "_restart_notification_inflight", None)
    if inflight is None:
        inflight = runner._restart_notification_inflight = set()
    return inflight


def _draining(runner) -> bool:
    return (getattr(runner, "_draining", False) is True
            or getattr(runner, "_restart_requested", False) is True)


def _producer_id(session_key, identity) -> str:
    if identity is None:
        return uuid.uuid4().hex
    return uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(["hermes-notice", session_key, identity])).hex


def _notice_rows(event, session_key) -> list:
    members = event._restart_notification_members or [(event._restart_notification_identity, event.text)]
    records = event._restart_notification_records
    if not records:
        records = tuple(_producer_id(session_key, identity) for identity, _text in members)
        if len(records) == 1 and event._restart_notification_id:
            records = (event._restart_notification_id,)
    # Preserve allocated identities even when cancellation precedes the save receipt.
    event._restart_notification_records = records
    event._restart_notification_id = records[0]
    return list(dict((identity, member[1]) for identity, member in zip(records, members)).items())


def _notice_payload(event, metadata, rows) -> dict:
    return {
        "source": event.source.to_dict(), "transport_profile": transport_profile_of(event.source),
        "relay_transport": event.source.delivered_via_upstream_relay is True,
        "text": event.text, "metadata": metadata, "message_type": event.message_type.value,
        "allow_gateway_control": event.allow_gateway_control, "reply_expected": event.reply_expected,
        "media_urls": event.media_urls, "media_types": event.media_types,
        "auto_skill": event.auto_skill, "channel_prompt": event.channel_prompt, "_notice_rows": rows,
    }


def _bind_admission(runner, event, path, metadata, rows, inserted) -> None:
    inserted = tuple(inserted or ())
    for identity, _text in rows:
        if identity not in inserted:
            _inflight(runner).discard((path, identity))
    event._restart_notification_path = path
    event._restart_notification_records = inserted or tuple(identity for identity, _text in rows)
    event._restart_notification_id = event._restart_notification_records[0]
    event.metadata = metadata
    event._restart_notification_deferred = not inserted
    if inserted and len(inserted) < len(rows):
        event.text = "\n\n".join(text for identity, text in rows if identity in inserted)


async def admit_notice(runner, event, session_key) -> None:
    """Commit before scheduling/queueing, with an intake claim over every member row."""
    if not event.internal or runner is None or event._restart_notification_path:
        return
    from gateway.wake import WakeNotAccepted
    lock = getattr(runner, "_restart_notification_admission_lock", None)
    if lock is None:
        lock = runner._restart_notification_admission_lock = asyncio.Lock()
    async with lock:
        if event._restart_notification_path:
            return
        receipts = []
        try:
            entry = await asyncio.to_thread(runner.session_store.lookup_by_session_key, session_key)
            metadata = dict(event.metadata or {})
            session_id = metadata.get("gateway_session_id") or (entry.session_id if entry else None)
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("internal notice has no owning session")
            metadata.update(gateway_session_key=session_key, gateway_session_id=session_id)
            path = Path(runner._resolve_profile_home_for_source(event.source)) / "restart_notifications" / "pending.sqlite3"
            rows = _notice_rows(event, session_key)
            receipts = [(path, identity) for identity, _text in rows]
            _inflight(runner).update(receipts)
            inserted = await asyncio.to_thread(_save, path, rows[0][0], _notice_payload(event, metadata, rows))
            _bind_admission(runner, event, path, metadata, rows, inserted)
        except BaseException as exc:
            _inflight(runner).difference_update(receipts)
            if not isinstance(exc, Exception):
                raise
            raise WakeNotAccepted("internal notice restart handoff failed") from exc


def finish_admission(adapter, event) -> None:
    if not event._restart_notification_path:
        return
    # The adapter's synchronous session guard now prevents concurrent execution.
    release_notice(adapter.gateway_runner, event)
    event._gateway_accepted = True


def _event_records(event) -> tuple:
    return event._restart_notification_records or (event._restart_notification_id,)


def release_notice(runner, event) -> None:
    if event is not None and event._restart_notification_path:
        _inflight(runner).difference_update((event._restart_notification_path, identity)
                                            for identity in _event_records(event))


async def defer_during_restart(adapter, event: MessageEvent, session_key: str) -> bool:
    runner = getattr(adapter, "gateway_runner", None)
    if not event.internal or runner is None or not _draining(runner):
        return False
    await admit_notice(runner, event, session_key)
    event._restart_notification_deferred = True
    release_notice(runner, event)
    return True


async def defer_or_reject(adapter, event, session_key, expected_session_key) -> bool:
    if expected_session_key and session_key != expected_session_key:
        logger.warning("Dropping internally routed event: expected session=%s derived=%s",
                       expected_session_key, session_key)
        return True
    await admit_notice(getattr(adapter, "gateway_runner", None), event, session_key)
    if event._restart_notification_deferred or await defer_during_restart(adapter, event, session_key):
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
    """Mark this completed turn's owned rows before preparing another turn."""
    if event is None or not event.internal or event._restart_notification_consumed:
        return
    path = event._restart_notification_path
    if path and event._restart_notification_id:
        identities = _event_records(event)
        await asyncio.to_thread(_complete, path, identities)
        _settled(runner).update((path, identity) for identity in identities)
        release_notice(runner, event)
    event._restart_notification_consumed = True


async def dispatch_message(adapter, event: MessageEvent):
    if not event.internal:
        return await adapter._message_handler(event)
    runner = getattr(adapter, "gateway_runner", None)
    if runner is None:
        return await adapter._message_handler(event)
    key = adapter._event_session_key(event)
    if await defer_during_restart(adapter, event, key):
        return None
    event._restart_notification_deferred = False
    try:
        response = await adapter._message_handler(event)
        executed = getattr(event, "_heartbeat_execution_started", False) is True
        if executed and getattr(event, "_restart_notification_turn_owned", False) is not True:
            await complete_notice(runner, event)
        if not executed and not _queued(runner, adapter, event, key):
            release_notice(runner, event)
            return None
        return response
    except BaseException:
        release_notice(runner, event)
        raise


async def begin_execution(runner, event, session_key) -> bool:
    from gateway.run_heartbeat_acceptance import heartbeat_owner_is_current
    if not heartbeat_owner_is_current(runner, event, session_key):
        return False
    if event is not None:
        if event.internal and _draining(runner):
            await admit_notice(runner, event, session_key)
            event._restart_notification_deferred = True
            release_notice(runner, event)
            return False
        event._heartbeat_execution_started = True
    return True


def _journal_paths(runner) -> list[Path]:
    homes = {Path(runner.session_store.sessions_dir).parent}
    if getattr(runner.config, "multiplex_profiles", False):
        from gateway.run import _multiplex_profile_homes
        homes.update(Path(home) for _profile, home in _multiplex_profile_homes(runner.config))
    return [home / "restart_notifications" / "pending.sqlite3" for home in sorted(homes)]


def _pending_groups(rows) -> list:
    groups = {}
    for identity, payload in rows:
        groups.setdefault(payload.get("batch_id", identity), []).append((identity, payload))
    return list(groups.values())


def _replay_text(rows) -> str:
    identities = [identity for identity, _payload in rows]
    payload = rows[0][1]
    if identities == payload.get("batch_members"):
        return payload["batch_text"]
    return "\n\n".join(payload["text"] for _identity, payload in rows)


async def _replay_group(runner, path: Path, rows) -> None:
    # A cancelled completion await can have committed without updating the local cache.
    rows = [(identity, payload) for identity, payload in rows
            if (path, identity) not in _inflight(runner) and (path, identity) not in _settled(runner)
            and _is_pending(path, identity)]
    if not rows:
        return
    notice_id, payload = rows[0]
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
    from gateway.platforms.event import MessageEvent, MessageType
    event = MessageEvent(
        text=_replay_text(rows), source=source, internal=True,
        message_type=MessageType(payload["message_type"]), metadata=payload["metadata"],
        allow_gateway_control=payload["allow_gateway_control"], reply_expected=payload["reply_expected"],
        media_urls=payload["media_urls"], media_types=payload["media_types"],
        auto_skill=payload["auto_skill"], channel_prompt=payload["channel_prompt"],
    )
    event._restart_notification_path = path
    event._restart_notification_id = notice_id
    event._restart_notification_records = tuple(identity for identity, _payload in rows)
    _inflight(runner).update((path, identity) for identity in event._restart_notification_records)
    try:
        prime = getattr(adapter, "prime_routing_cache", None)
        if callable(prime):
            prime(event)
        from gateway.wake import admit_internal_event
        await admit_internal_event(adapter, event)
    except BaseException:
        release_notice(runner, event)
        raise


async def replay_pending(runner) -> None:
    if _draining(runner) or getattr(runner, "_startup_restore_in_progress", False):
        return
    lock = getattr(runner, "_restart_notification_replay_lock", None)
    if lock is None:
        lock = runner._restart_notification_replay_lock = asyncio.Lock()
    async with lock:
        for path in _journal_paths(runner):
            rows = await asyncio.to_thread(_pending, path)
            for group in _pending_groups(rows):
                if _draining(runner):
                    return
                try:
                    await _replay_group(runner, path, group)
                except Exception:
                    logger.warning("Restart notice group remains pending", exc_info=True)


async def resolve_restart_session(runner, event):
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
    while runner._running and not runner._shutdown_event.is_set():
        await replay_pending(runner)
        await asyncio.sleep(5)
