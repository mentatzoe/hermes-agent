"""Save and interrupt native waiting-only turns before the restart drain."""
import logging

from agent.interrupt_compat import request_hard_interrupt
from tools.restart_wait import claim_waits

logger = logging.getLogger(__name__)


def _owns_work(agent, session_key, task_ids):
    from tools.async_delegation import has_live_for_session
    from tools.process_registry import process_registry
    with agent._active_children_lock:
        if agent._active_children:
            return True
    if has_live_for_session(session_key=session_key, parent_session_id=agent.session_id):
        return True
    # Do not refresh detached processes here: kernel liveness probes can wedge
    # the gateway loop. Unreaped work conservatively keeps its drain protection.
    with process_registry._lock:
        return any(
            not process.exited and (process.session_key == session_key or process.owner_task_id in task_ids)
            for process in process_registry._running.values()
        )


async def yield_restart_waiters(runner):
    """Claim/freeze → persist → interrupt → release; never subtract a live executor.

    A failed persistence or ownership observation leaves the turn blocking. New
    queued workers cannot cross a claimed boundary, and an old claim cannot
    interrupt a successor turn. Normal drain accounting sees the actual unwind.
    """
    from gateway.run import _INTERRUPT_REASON_GATEWAY_RESTART, _INTERRUPT_TOOL_REASON_GATEWAY_SHUTDOWN
    for key, agent in list(runner._running_agents.items()):
        claim = claim_waits(agent) if hasattr(agent, "__dict__") else None
        if claim is None:
            continue
        try:
            if _owns_work(agent, key, claim.task_ids):
                continue
            saved = await runner.async_session_store.mark_resume_pending(
                key, "restart_wait", expected_session_id=agent.session_id,
            )
            if saved is not True:
                continue
            if runner._running_agents.get(key) is not agent or agent._current_turn_id != claim.turn_id:
                continue
            request_hard_interrupt(
                agent, _INTERRUPT_REASON_GATEWAY_RESTART, tool_reason=_INTERRUPT_TOOL_REASON_GATEWAY_SHUTDOWN,
            )
            claim.release(interrupted=True)
            logger.info("Restart: saved and released waiting-only turn %s", key)
        except Exception:
            logger.warning("Restart: could not safely release waiting turn %s", key, exc_info=True)
        finally:
            claim.release()
