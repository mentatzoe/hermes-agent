"""Zoe's voice must keep reaching the receiver while Aleph's reply plays in the call.

On 3 Oct and again on 12 Oct 2026 (16:52:30-16:52:55) the start of what Zoe said during
playback never reached transcription: the legacy one-shot playback path paused the
VoiceReceiver for the whole clip, so ``_on_packet`` returned early and every frame was
discarded. On 15 Jun 2026 Zoe set the requirement that inbound voice wins over bot
playback. Echo isn't a reason to pause: the receiver already drops the bot's own SSRC.
"""

import asyncio
import struct
from unittest.mock import MagicMock, patch

import pytest


def _make_adapter():
    from plugins.platforms.discord.adapter import DiscordAdapter
    from gateway.config import PlatformConfig, Platform

    config = PlatformConfig(enabled=True, extra={})
    config.token = "fake-token"
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter.config = config
    adapter._voice_clients = {}
    adapter._voice_locks = {}
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_input_callback = None
    adapter._on_voice_disconnect = None
    adapter._client = None
    adapter._allowed_user_ids = set()
    adapter._voice_fx_cfg = {"lead_silence_ms": 0}
    return adapter


def _make_receiver():
    from plugins.platforms.discord.adapter import VoiceReceiver

    vc = MagicMock()
    vc._connection.secret_key = [0] * 32
    vc._connection.dave_session = None
    vc._connection.ssrc = 9999
    vc._connection.add_socket_listener = MagicMock()
    vc._connection.remove_socket_listener = MagicMock()
    vc._connection.hook = None
    return VoiceReceiver(vc)


@pytest.mark.asyncio
async def test_legacy_playback_leaves_the_receiver_listening():
    """The receiver is never paused while a reply plays through the legacy path."""
    adapter = _make_adapter()
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = False
    paused_during_play = []

    receiver = MagicMock()
    receiver._paused = False
    receiver.pause.side_effect = lambda: setattr(receiver, "_paused", True)
    receiver.resume.side_effect = lambda: setattr(receiver, "_paused", False)

    def fake_play(source, after=None):
        paused_during_play.append(receiver._paused)
        after(None)

    vc.play.side_effect = fake_play
    adapter._voice_clients[111] = vc
    adapter._voice_receivers[111] = receiver
    adapter._voice_timeout_tasks[111] = MagicMock()

    with patch("discord.FFmpegPCMAudio"), \
            patch("discord.PCMVolumeTransformer", side_effect=lambda s, **kw: s):
        played = await adapter.play_in_voice_channel(111, "/tmp/reply.mp3")

    assert played is True
    assert paused_during_play == [False]
    receiver.pause.assert_not_called()


def test_receiver_still_ignores_the_bots_own_packets():
    """Echo protection without pausing: packets from the bot's own SSRC are dropped."""
    receiver = _make_receiver()
    receiver.start()
    receiver._bot_ssrc = 9999
    header = struct.pack(">BBHII", 0x80, 0x78, 1, 160, 9999)
    receiver._on_packet(header + b"\x00" * 40)
    assert len(receiver._buffers) == 0
    assert receiver._decode_ok == 0
