"""Text and durable occurrence membership for coalesced internal notices."""


def format_coalesced_process_completions(entries) -> str:
    """Build one bounded synthetic event from several redacted completions."""
    from gateway.run import _redact_gateway_user_facing_secrets
    lines = [
        f"[IMPORTANT: {len(entries)} background processes completed for this session.",
        "Treat these results as one completion batch and send at most one "
        "consolidated user-facing response.",
    ]
    shown = entries[:10]
    for _text, evt, _future in shown:
        session_id = str(evt.get("session_id") or "unknown")
        exit_code = evt.get("exit_code")
        reason = str(evt.get("completion_reason") or "exited")
        # Redact before slicing so a truncated token cannot evade the redaction floor.
        output = _redact_gateway_user_facing_secrets(str(evt.get("output") or "")).strip()
        if len(output) > 800:
            output = f"[… truncated …]\n{output[-800:]}"
        lines.append(f"\n- {session_id}: exit_code={exit_code}, reason={reason}")
        if output:
            lines.append(output)
    omitted = len(entries) - len(shown)
    if omitted:
        lines.append(f"\n- … and {omitted} more completion(s); inspect them with "
                     "the process tool if they affect the conclusion.")
    lines.append("If a result does not change the current conclusion, absorb it silently.]")
    return "\n".join(lines)


def with_notice_members(runner, primary, entries) -> dict:
    """Keep each producer occurrence addressable after the coalescing window changes."""
    from gateway.run_notifications import _mark_internal_notification
    members = [(runner._completion_delivery_identity(event), _mark_internal_notification(text))
               for text, event in entries]
    return {**primary, "_restart_notice_members": members}


def kanban_notice_members(notification, events) -> list:
    """Capture each wake group's actual occurrences and standalone handoff text."""
    from copy import copy
    members = []
    for event in events:
        single = copy(notification)
        single.d = {**notification.d, "events": [event]}
        single.wake_handoff = single.wake_review_detail = ""
        single.format_event(event)
        single.build_wake_text()
        if single.wake_kinds:
            identity = ("kanban", single.board_slug, single.task_id, event.id)
            members.append((identity, single.synth))
    return members
