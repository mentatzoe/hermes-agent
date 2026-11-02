"""Delivery of per-call steer facts; model text stays separate from provenance."""
from agent.mid_turn_input import mid_turn_human_row
from agent.prompt_builder import STEER_DISPLAY_KIND, format_steer_marker


def delivered_steer_row(agent, entries):
    """Build the delivered user row and add its owned attachments to this turn."""
    text = entries if isinstance(entries, str) else '\n'.join(entry['text'] for entry in entries)
    return mid_turn_human_row(agent, entries, content=format_steer_marker(text).lstrip(),
                              display_kind=STEER_DISPLAY_KIND)


def handoff_pending_steer(agent, result):
    """Return late entries for the next turn, keeping the text-only API intact."""
    entries = agent._drain_pending_steer(with_entries=True)
    if not entries:
        return
    result['pending_steer'] = '\n'.join(entry['text'] for entry in entries)
    if any('facts' in entry for entry in entries):
        result['pending_steer_entries'] = entries
