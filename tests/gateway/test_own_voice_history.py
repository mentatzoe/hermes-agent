"""The attribution-enriched user input survives an actual agent/session round trip."""
import json
from unittest.mock import Mock, patch

import pytest

from tests.agent.test_refusal_notice_privacy import agent, response
from tests.gateway.test_own_voice_attribution import runner, SAID


@pytest.mark.asyncio
async def test_forwarded_recording_attribution_reaches_stored_history(agent, monkeypatch):
    db = agent._session_db
    db.create_session("earlier-tts", source="discord")
    db.append_message("earlier-tts", "assistant", tool_calls=[{
        "id": "tts-1", "type": "function", "function": {
            "name": "text_to_speech", "arguments": json.dumps({"text": SAID}),
        },
    }])
    with patch("tools.transcription_tools.transcribe_audio", return_value={"success": True, "transcript": SAID}):
        text, raw = await runner()._enrich_message_with_transcription("listen to this", ["/tmp/voice.ogg"])
    monkeypatch.setattr(agent, "_interruptible_api_call", Mock(return_value=response("Recognized.", finish_reason="stop")))
    result = agent.run_conversation(text)
    assert result["final_response"] == "Recognized."
    assert raw == [SAID]
    stored = db.get_messages(agent.session_id)
    user = next(row for row in stored if row["role"] == "user")
    assert "Aleph's own earlier voice reply" in user["content"]
    assert "not the sender's" in user["content"]
    assert SAID in user["content"]
