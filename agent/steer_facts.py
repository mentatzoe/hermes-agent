"""Delivery of per-call steer facts; model text stays separate from provenance."""
from agent.prompt_builder import steer_user_row


def delivered_steer_row(agent, entries):
    """Build the delivered user row and add its owned attachments to this turn."""
    if isinstance(entries, str):
        return steer_user_row(entries)
    paths = [path for entry in entries for path in
             (entry.get('facts') or {}).get('recoverable_audio_paths', [])]
    if paths:
        current = getattr(agent, '_turn_recoverable_audio_paths', None) or []
        agent._turn_recoverable_audio_paths = list(dict.fromkeys([*current, *paths]))
    return steer_user_row('\n'.join(entry['text'] for entry in entries), entries=entries)


def handoff_pending_steer(agent, result):
    """Return late entries for the next turn, keeping the text-only API intact."""
    entries = agent._drain_pending_steer(with_entries=True)
    if not entries:
        return
    result['pending_steer'] = '\n'.join(entry['text'] for entry in entries)
    if any('facts' in entry for entry in entries):
        result['pending_steer_entries'] = entries
