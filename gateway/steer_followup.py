"""Host-owned facts for accepted steers delivered as a later gateway turn.

The rendered text has already been prepared. Keep it unchanged, and carry each
sender's facts in persistence-only metadata rather than parsing its labels.
"""
from copy import deepcopy

from gateway.platforms.event import MessageEvent


def steer_followup_event(runner, session_key, source, text, entries=None):
    prompt, source = runner._pinned_channel_inputs(session_key, None, source, internal=True)
    event = MessageEvent(text=text, source=source, channel_prompt=prompt)
    if entries is not None:
        event._gateway_steer_entries = deepcopy(entries)
    return event


def prepare_steer_followup(event):
    """Restore ingress facts after a FIFO delay; no second STT or sender framing."""
    entries = getattr(event, '_gateway_steer_entries', None)
    if entries is None:
        return False
    facts = [entry['facts'] for entry in entries if 'facts' in entry]
    event._gateway_typed_user_content = '\n\n'.join(
        item['typed_user_content'] for item in facts if item.get('typed_user_content'))
    event._gateway_recoverable_audio_paths = list(dict.fromkeys(
        path for item in facts for path in item.get('recoverable_audio_paths', [])))
    return True


def steer_followup_metadata(event):
    entries = getattr(event, '_gateway_steer_entries', None)
    return {'steer_entries': deepcopy(entries)} if entries is not None else {}
