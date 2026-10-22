"""Resolve a gateway turn's session, including restart-retained notice ownership."""

import asyncio
import dataclasses
from contextlib import suppress
import logging

logger = logging.getLogger("gateway.run")


async def resolve_turn_session(runner, event, source):
    """Resolve topic routing and pinned ownership before the turn may create session state."""
    event_metadata = getattr(event, "metadata", None) or {}
    expected_session_key = str(event_metadata.get("gateway_session_key") or "").strip()
    recovered = (await asyncio.to_thread(runner._recover_telegram_topic_thread_id, source)
                 if not expected_session_key else None)
    if recovered is not None:
        logger.info("telegram topic recovery: chat=%s user=%s %r -> %s",
                    source.chat_id, source.user_id, source.thread_id, recovered)
        source = dataclasses.replace(source, thread_id=recovered)
        with suppress(Exception):
            event.source = source

    if expected_session_key:
        derived_session_key = runner._session_key_for_source(source)
        if derived_session_key != expected_session_key:
            logger.warning(
                "Dropping internally routed event after route recovery: expected session=%s derived=%s",
                expected_session_key, derived_session_key,
            )
            return

    strict_session = bool(event_metadata.get("gateway_session_strict"))
    pinned_session_id = str(event_metadata.get("gateway_session_id") or "").strip()
    restart_notice = bool(getattr(event, "_restart_notification_id", None))
    if restart_notice:
        from gateway.restart_notifications import resolve_restart_session
        session_entry = await resolve_restart_session(runner, event)
        if session_entry is None:
            return
    elif strict_session:
        session_entry = await runner.async_session_store.lookup_by_session_key(expected_session_key)
        if session_entry is None or not pinned_session_id or session_entry.session_id != pinned_session_id:
            logger.warning(
                "Dropping internally routed event: expected session id=%s is no longer current for key=%s",
                pinned_session_id or "missing", expected_session_key or "missing",
            )
            return
    else:
        # Internal wakes must not keep a lane alive across daily/idle reset boundaries.
        session_entry = await runner.async_session_store.get_or_create_session(
            source, touch_activity=not bool(getattr(event, "internal", False)),
        )
    session_key = session_entry.session_key
    if not strict_session and pinned_session_id and not restart_notice:
        resolved_entry = await runner._resolve_async_delegation_session(session_entry, pinned_session_id)
        if resolved_entry is None:
            return
        session_entry = resolved_entry
    runner._cache_session_source(session_key, source)
    if await asyncio.to_thread(runner._is_telegram_topic_lane, source):
        session_entry = await runner._hmwa_heal_telegram_topic_binding(source, session_entry, session_key)
    from gateway.run_heartbeat_acceptance import resolve_heartbeat_owner
    if not await resolve_heartbeat_owner(runner, event, session_entry):
        return
    return source, session_entry, session_key
