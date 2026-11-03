"""Collect acknowledged assistant-text deliveries for one gateway turn."""

import asyncio
import logging
from contextlib import suppress


logger = logging.getLogger("gateway.run")


async def edit_streamed_message(
    _sc, source, response, content, *, _sk, ok, fail_result: str, fail_exc: str,
) -> None:
    """Edit the stream consumer's message in place with ``content``; on success mark
    ``response["already_sent"]`` and log ``ok``. A returned failure logs ``fail_result`` as
    ``(session, error)`` and an exception logs ``fail_exc`` as ``(session, exc)``; either way
    ``already_sent`` stays unset so the normal final send delivers the content."""
    try:
        _res = await _sc.adapter.edit_message(
            chat_id=source.chat_id, message_id=_sc.message_id, content=content, finalize=True,
        )
    except Exception as _edit_err:
        logger.warning(fail_exc, _sk, _edit_err, exc_info=True)
        return
    if not getattr(_res, "success", True):
        logger.warning(fail_result, _sk, getattr(_res, "error", None))
        return
    response["already_sent"] = True
    logger.info(*ok)


async def record_visible_text_delivery(result, turn_ctx) -> None:
    """Carry delivery evidence to completion without treating scheduling as success.

    The context belongs to one agent turn, not the whole recursive queued chain.
    Status/progress sends are deliberately not in its interim-future collection.
    """
    if not isinstance(result, dict):
        return
    futures = [asyncio.wrap_future(f) for f in getattr(turn_ctx, "interim_delivery_futures", ())]
    if futures:
        done, _ = await asyncio.wait(futures, timeout=5.0)
        for future in done:
            with suppress(Exception, asyncio.CancelledError):
                if getattr(future.result(), "success", False) is True:
                    result["visible_text_delivered"] = True
    consumer = turn_ctx.stream_consumer_holder[0]
    delivered = getattr(consumer, "has_durably_delivered_text", None)
    if callable(delivered):
        with suppress(Exception):
            if delivered() is True:
                result["visible_text_delivered"] = True
