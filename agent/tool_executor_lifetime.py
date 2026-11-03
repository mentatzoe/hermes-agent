"""Audit outcomes from tool workers whose executor has already stopped waiting."""
from __future__ import annotations

import time


def log_abandoned_tool_completion(done, *, ref, session_id, started, skipped_dispatch, logger):
    """Log once without replaying post-tool hooks or publishing a second result."""
    if done.cancelled():
        outcome = "cancelled_before_worker_start"
    else:
        try:
            done.result()
            outcome = "dispatch_prevented" if skipped_dispatch.is_set() else "finished_after_abandonment"
        except BaseException:
            # A Future stores BaseException too; its callback must not propagate
            # it into the abandoning caller or the worker pool's cleanup.
            logger.exception(
                "sequential tool %s late outcome=raised session=%s tool_call_id=%s elapsed=%.1fs",
                ref.name, session_id, ref.call_id, time.monotonic() - started,
            )
            return
    logger.warning(
        "sequential tool %s late outcome=%s session=%s tool_call_id=%s elapsed=%.1fs",
        ref.name, outcome, session_id, ref.call_id, time.monotonic() - started,
    )
