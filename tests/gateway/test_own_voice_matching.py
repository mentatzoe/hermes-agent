"""Only recognized speech calls can supply an own-voice text match."""
import json
import sqlite3
import time

import pytest

from gateway.run_voice import _is_agents_own_voice

WORDS = "tomorrow please bring twelve purple notebooks green pencils orange folders and several maps into our meeting room"
OTHER = "this is an unrelated spoken greeting from the assistant about gardens and flowers"


def call(name, text):
    return {"function": {"name": name, "arguments": json.dumps({"text": text})}}


@pytest.fixture
def voice_db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = sqlite3.connect(tmp_path / "state.db")
    db.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, role TEXT, tool_calls TEXT, timestamp REAL)")

    def insert(calls):
        db.execute("INSERT INTO messages (role, tool_calls, timestamp) VALUES ('assistant', ?, ?)",
                   (json.dumps(calls), time.time()))
        db.commit()

    yield insert
    db.close()


@pytest.mark.parametrize("name", ["browser_type", "mcp__tts_local__get_voice_mode", "other_synthesize"])
def test_non_tts_call_in_tts_row_is_not_own_voice(voice_db, name):
    voice_db([call("text_to_speech", OTHER), call(name, WORDS)])
    assert not _is_agents_own_voice("human-recording.ogg", WORDS)


def test_reversed_sentence_is_not_own_voice(voice_db):
    voice_db([call("text_to_speech", WORDS)])
    assert not _is_agents_own_voice("human-recording.ogg", " ".join(reversed(WORDS.split())))


@pytest.mark.parametrize("name", ["text_to_speech", "text_to_speech_tool", "mcp__tts_local__synthesize"])
def test_recognized_recent_speech_matches(voice_db, name):
    voice_db([call(name, WORDS)])
    assert _is_agents_own_voice("forwarded-recording.ogg", WORDS)
