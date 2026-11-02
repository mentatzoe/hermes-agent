"""Adapter retry handoff over the existing pending slot, shared with the turn drain."""

import asyncio


async def drain_pending(adapter, session_key, delay, guard):
    """Claim queued work only when the runner no longer owns a turn for this key.

    A replacement adapter has its own task/guard, but the original turn can still
    drain this slot. Popping before awaiting dispatch hid admitted humans from
    that consumer and re-enqueued them at their class's tail (fork #61).
    """
    while True:
        if delay > 0:
            await asyncio.sleep(delay)
        await adapter._flush_text_debounce_now(session_key)
        pending = adapter._pending_messages.get(session_key)
        if pending is None:
            adapter._cleanup_finished_session_task(session_key, guard)
            return
        runner = adapter.gateway_runner
        if runner is not None and runner._is_session_running(session_key):
            # Keep dispatch's stale/reaped-slot recovery, in the pending event's
            # owning profile (a replacement transport may serve several homes).
            with runner._profile_scope_for_source(pending.source):
                runner._hm_evict_idle_stale_agent(session_key)
                if runner._is_session_running(session_key):
                    runner._hm_evict_reaped_agent(session_key)
            if runner._is_session_running(session_key):
                delay = adapter._requeue_backoff_delay(session_key, pending, pending)
                continue
        # No await between checking ownership and taking the slot. The next
        # adapter task reads the current selection, not a pre-handoff snapshot.
        adapter._pending_messages.pop(session_key)
        if runner is not None:
            # Use the same class-priority selection and stage the successor. An
            # empty slot with overflow would make idle admission mistake this
            # already-selected head for a new arrival and move it to the tail.
            pending = runner._promote_queued_event(session_key, adapter, pending)
        await adapter._process_message_background(pending, session_key)
        return
