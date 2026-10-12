"""Auto voice replies to synthetic prompt events must reach the bound Discord voice channel.

Heartbeat ticks, /goal continuations and loop wakeups are injected through
``GatewayRunner._synthetic_prompt_event``, which builds a ``MessageEvent`` with no
``raw_message``. ``_get_guild_id`` reads the guild only from ``raw_message``, so the
reply used to fall back to ``send_voice`` (a voice-note attachment in the thread) even
while the bot sat in the voice channel bound to that thread via ``/voice join``. That
also meant heartbeat speech never reset the adapter's voice inactivity timer.

The guild is resolved the way ``DiscordAdapter.play_tts`` does: the connected guild whose
``_voice_text_channels`` binding is the event's chat.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource

GUILD = 1484971544512958597
THREAD = "1559186588863566044"


def _source(chat_id=THREAD):
    return SessionSource(
        platform=Platform.DISCORD, chat_id=chat_id, chat_type="thread",
        user_id="42", thread_id=chat_id, message_id="registering-msg",
    )


def _heartbeat_event(chat_id=THREAD):
    # The real builder the heartbeat/goal/loop paths use.
    return GatewayRunner._synthetic_prompt_event(_source(chat_id), "[Heartbeat] tick")


def _adapter(bound_chat=THREAD, connected=True):
    return SimpleNamespace(
        _voice_text_channels={GUILD: int(bound_chat)},
        is_in_voice_channel=lambda gid: connected and gid == GUILD,
        play_in_voice_channel=AsyncMock(return_value=True),
        send_voice=AsyncMock(),
        get_voice_channel_context=lambda gid: "Lounge: Zoe" if gid == GUILD else "",
    )


def _runner(adapter):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: adapter}
    return runner


def test_synthetic_event_has_no_raw_message():
    """Premise of the bug: the guild cannot come from raw_message."""
    event = _heartbeat_event()
    assert event.raw_message is None
    assert GatewayRunner._get_guild_id(event) is None


@pytest.mark.asyncio
async def test_synthetic_event_reply_plays_in_bound_voice_channel():
    adapter = _adapter()
    await _runner(adapter)._deliver_voice_reply(_heartbeat_event(), ["/x/a.mp3", "/x/b.mp3"])
    assert [c.args for c in adapter.play_in_voice_channel.await_args_list] == [
        (GUILD, "/x/a.mp3"), (GUILD, "/x/b.mp3")]
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_synthetic_event_not_in_voice_channel_falls_back_to_send_voice():
    adapter = _adapter(connected=False)
    await _runner(adapter)._deliver_voice_reply(_heartbeat_event(), ["/x/a.mp3"])
    adapter.play_in_voice_channel.assert_not_awaited()
    adapter.send_voice.assert_awaited_once()
    assert adapter.send_voice.await_args.kwargs["chat_id"] == THREAD
    assert adapter.send_voice.await_args.kwargs["metadata"]["notify"] is True


@pytest.mark.asyncio
async def test_synthetic_event_in_other_chat_does_not_hijack_voice_channel():
    """A heartbeat in a chat the voice channel is NOT bound to keeps the attachment path."""
    adapter = _adapter(bound_chat="999")
    await _runner(adapter)._deliver_voice_reply(_heartbeat_event(), ["/x/a.mp3"])
    adapter.play_in_voice_channel.assert_not_awaited()
    adapter.send_voice.assert_awaited_once()


@pytest.mark.asyncio
async def test_real_message_still_uses_raw_guild():
    """Unchanged path: a real Discord message resolves its guild from raw_message."""
    adapter = _adapter(bound_chat="999")
    event = MessageEvent(text="hi", message_type=MessageType.TEXT, source=_source(),
                         raw_message=SimpleNamespace(guild=SimpleNamespace(id=GUILD)))
    await _runner(adapter)._deliver_voice_reply(event, ["/x/a.mp3"])
    adapter.play_in_voice_channel.assert_awaited_once_with(GUILD, "/x/a.mp3")
    adapter.send_voice.assert_not_awaited()


def test_synthetic_event_gets_voice_channel_sidecar_note():
    """Same blind spot in run_agent_cache: synthetic turns saw no voice-channel context."""
    event = _heartbeat_event()
    note = _runner(_adapter())._voice_channel_sidecar_note(event, event.source, "")
    assert note == "[Voice channel now: Lounge: Zoe]"
