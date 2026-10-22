"""Adapter shutdown: retain restart notices before cancelling their scheduled tasks."""

import asyncio
import contextlib
import logging

from gateway.restart_notifications import preserve_pending_notices

logger = logging.getLogger("gateway.platforms.base")


async def cancel_adapter_background_tasks(adapter) -> None:
    """Bound cancellation while preserving admitted notices that have not executed."""
    # Re-drain: a message arriving mid-gather can create another tracked task.
    for _ in range(5):
        await preserve_pending_notices(adapter.gateway_runner, adapter)
        tasks = [task for task in adapter._background_tasks if not task.done()]
        if not tasks:
            break
        for task in tasks:
            adapter._expected_cancelled_tasks.add(task)
            task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*(asyncio.shield(task) for task in tasks), return_exceptions=True),
                timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("[%s] %d background task(s) did not exit within 5s; "
                           "releasing tracking and letting them unwind in the background",
                           adapter.name, sum(not task.done() for task in tasks))
            break
    await preserve_pending_notices(adapter.gateway_runner, adapter)
    with contextlib.suppress(Exception):
        from gateway.shutdown_flush import flush_pending_to_file
        flush_pending_to_file(adapter._pending_messages, reason="adapter_shutdown")
    for state in adapter._text_debounce_store().values():
        state.cancel_timer()
    for bucket in (adapter._background_tasks, adapter._expected_cancelled_tasks,
                   adapter._session_tasks, adapter._pending_messages, adapter._active_sessions,
                   adapter._requeue_counts, adapter._text_debounce_store(),
                   getattr(adapter, "_restart_notification_tasks", {})):
        bucket.clear()
