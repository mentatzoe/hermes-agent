"""Regression tests for gateway voice-mode TTS deduplication."""

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _runner(mode="all"):
    runner = object.__new__(GatewayRunner)
    runner._voice_mode = {"discord:chan": mode}
    return runner


def _event(message_type=MessageType.TEXT):
    return MessageEvent(
        text="hello",
        message_type=message_type,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="chan",
            user_id="user",
            user_name="user",
        ),
    )


def test_voice_mode_all_sends_builtin_tts_for_plain_text_reply():
    runner = _runner("all")

    assert runner._should_send_voice_reply(
        _event(MessageType.TEXT),
        "Plain text response",
        agent_messages=[],
        already_sent=False,
    ) is True


def test_voice_mode_all_does_not_send_default_tts_when_mcp_tts_was_called():
    runner = _runner("all")
    agent_messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "mcp_tts_local_synthesize"},
                }
            ],
        }
    ]

    assert runner._should_send_voice_reply(
        _event(MessageType.TEXT),
        "MCP audio is coming separately.",
        agent_messages=agent_messages,
        already_sent=False,
    ) is False


def test_voice_mode_all_does_not_send_default_tts_when_response_has_audio_media():
    runner = _runner("all")

    assert runner._should_send_voice_reply(
        _event(MessageType.TEXT),
        "Short spoken reply.\n\nMEDIA:/tmp/tts-local/reply-123.ogg",
        agent_messages=[],
        already_sent=False,
    ) is False


def test_voice_mode_all_keeps_existing_text_to_speech_dedup():
    runner = _runner("all")
    agent_messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "text_to_speech"},
                }
            ],
        }
    ]

    assert runner._should_send_voice_reply(
        _event(MessageType.TEXT),
        "Built-in TTS audio is coming separately.",
        agent_messages=agent_messages,
        already_sent=False,
    ) is False
