"""Own TTS recordings are attributed separately from their unchanged STT transcript."""
import asyncio
import json
import sqlite3
import threading
import time
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from gateway import run_voice

SAID = "the quick brown fox jumps over the lazy dog while seven wizards quietly judge every boxing match tonight"


def make_db(home, *, filler=0):
    db = sqlite3.connect(home / "state.db")
    db.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, role TEXT, tool_calls TEXT, timestamp REAL)")
    calls = json.dumps([{"function": {"name": "text_to_speech", "arguments": json.dumps({"text": SAID})}}])
    db.execute("INSERT INTO messages VALUES (1, 'assistant', ?, ?)", (calls, time.time()))
    db.executemany("INSERT INTO messages VALUES (?, 'user', NULL, ?)",
                   [(i + 2, time.time()) for i in range(filler)])
    db.commit()
    db.close()


def runner():
    r = GatewayRunner.__new__(GatewayRunner)
    r.config = GatewayConfig(stt_enabled=True)
    return r


@pytest.mark.asyncio
@pytest.mark.parametrize("text,own", [(SAID, True), ("yes thank you", False),
    ("a completely different discussion about gardens and vegetables with tomatoes cucumbers carrots beans and peas", False)])
async def test_forwarded_tts_attribution_keeps_raw_transcript(tmp_path, monkeypatch, text, own):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    make_db(tmp_path)
    with patch("tools.transcription_tools.transcribe_audio", return_value={"success": True, "transcript": text}):
        enriched, transcripts = await runner()._enrich_message_with_transcription("a caption", ["/tmp/voice.ogg"])
    assert transcripts == [text]
    assert ("Aleph's own earlier voice reply" in enriched) is own
    assert f'"{text}"' in enriched
    assert enriched.endswith("a caption")


@pytest.mark.asyncio
async def test_transcription_can_collect_only_sender_authored_words(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    make_db(tmp_path)
    typed = []
    with patch("tools.transcription_tools.transcribe_audio", return_value={"success": True, "transcript": SAID}):
        _, transcripts = await runner()._enrich_message_with_transcription(
            "a caption", ["/tmp/voice.ogg"], typed_transcripts=typed,
        )
    assert transcripts == [SAID]  # the visible transcription still includes the forwarded recording
    assert typed == []  # Aleph's own earlier reply is not the sender's typed/spoken statement


def test_own_voice_scan_stays_within_recent_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    make_db(tmp_path, filler=50)
    monkeypatch.setattr(run_voice, "_OWN_VOICE_SCAN_ROWS", 100)
    assert run_voice._is_agents_own_voice("unused", SAID)
    monkeypatch.setattr(run_voice, "_OWN_VOICE_SCAN_ROWS", 10)
    assert not run_voice._is_agents_own_voice("unused", SAID)


def test_query_budget_aborts(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    make_db(tmp_path, filler=5000)
    monkeypatch.setattr(run_voice, "_OWN_VOICE_QUERY_BUDGET_S", -1)
    assert not run_voice._is_agents_own_voice("unused", SAID)


@pytest.mark.asyncio
async def test_own_voice_match_runs_off_loop(monkeypatch):
    started, progressed = threading.Event(), threading.Event()
    blocked = []
    def slow_match(*args):
        started.set()
        if not progressed.wait(2):
            blocked.append(True)
        return True
    async def witness():
        for _ in range(2000):
            if started.is_set():
                progressed.set()
                return
            await asyncio.sleep(.005)
        raise AssertionError("matcher never started")
    monkeypatch.setattr(run_voice, "_is_agents_own_voice", slow_match)
    with patch("tools.transcription_tools.transcribe_audio", return_value={"success": True, "transcript": SAID}):
        (text, transcripts), _ = await asyncio.gather(
            runner()._enrich_message_with_transcription("", ["/tmp/voice.ogg"]), witness())
    assert not blocked
    assert transcripts == [SAID]
    assert "not the sender's" in text


@pytest.mark.asyncio
async def test_busy_voice_path_keeps_attribution(tmp_path, monkeypatch):
    from tests.gateway.test_busy_session_ack import _make_runner, _make_event
    from gateway.platforms.event import MessageType
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    make_db(tmp_path)
    r, _ = _make_runner()
    r.config = GatewayConfig(stt_enabled=True)
    r._should_echo_stt_transcripts = lambda *_: False
    event = _make_event(text="")
    event.message_type = MessageType.VOICE
    event.media_urls = ["/tmp/voice.ogg"]
    event.media_types = ["audio/ogg"]
    with patch("tools.transcription_tools.transcribe_audio", return_value={"success": True, "transcript": SAID}):
        text = await r._prepare_busy_steer_text(event)
    assert "Aleph's own earlier voice reply" in text
    assert "not the sender's" in text
    assert SAID in text
