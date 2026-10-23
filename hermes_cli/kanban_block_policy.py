"""Block routing for ordinary tasks and externally supervised issue cards."""

from __future__ import annotations


def is_controller_owned(created_by: str | None) -> bool:
    """The reserved ``<name>-controller`` creator namespace owns hand-back policy.

    Use persisted provenance, never issue/body text or the calling profile's
    settings: a question and a later installation hand-back may share a kind.
    This is a routing contract, not an authentication boundary.
    """
    suffix = "-controller"
    return bool(created_by and created_by.endswith(suffix) and created_by[:-len(suffix)].strip())


def route_block(
    kind: str | None, reason: str | None, source_status: str, *,
    prev_kind: str | None, prev_recurrences: int, created_by: str | None,
) -> tuple[str, str, str, tuple, dict]:
    """Keep controller hand-backs with their owner; escalate ordinary loops.

    Dependency waits retain their parent-gated ``todo`` lane. Callers re-kind
    impossible dependency waits to sticky ``needs_input`` before routing.
    Recurrences survive unblocks for both policies, but a controller, not the
    native triage decomposer, decides what its repeated hand-backs mean.
    """
    from hermes_cli.kanban_db import BLOCK_RECURRENCE_LIMIT

    payload = {"reason": reason, "kind": kind, "source_status": source_status}
    if kind == "dependency":
        return "todo", "dependency_wait", "block_kind    = ?", (kind,), payload
    recurrences = prev_recurrences + 1 if prev_kind == kind else 1
    set_sql = "block_kind    = ?,\n                       block_recurrences = ?"
    payload["recurrences"] = recurrences
    if recurrences >= BLOCK_RECURRENCE_LIMIT and not is_controller_owned(created_by):
        payload["limit"] = BLOCK_RECURRENCE_LIMIT
        return "triage", "block_loop_detected", set_sql, (kind, recurrences), payload
    return "blocked", "blocked", set_sql, (kind, recurrences), payload
