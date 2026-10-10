"""Characterization of the companion helper copied from the live snapshot."""
import pytest

from agent.companion_compaction import (
    CompanionConfig, IDENTITY_HEADING, IDENTITY_END, load_companion_config,
    session_origin, strip_identity_header,
)
from agent.context_compressor import ContextCompressor


def test_disabled_and_malformed_configuration_is_inert():
    for raw in [None, [], {}, {"companion": False}]:
        assert not load_companion_config(raw).enabled
    c = ContextCompressor(model="test", config_context_length=200000, quiet_mode=True)
    c._companion = CompanionConfig({})
    assert c._companion_prepend_identity("plain") == "plain"
    assert c._serialize_for_summary([{"role": "user", "content": "hello"}]) == "[USER]: hello"


def test_identity_file_stop_marker_bounds_and_card_dedup(tmp_path):
    path = tmp_path / "identity.md"
    path.write_text("Zoe uses they/them.\n## Style\nDo not include this part.")
    cfg = CompanionConfig({"enabled": True, "people": ["Aleph", "Zoe"],
                           "identity_files": [str(path) + "#until:## Style", str(tmp_path / "missing")]})
    text = cfg.identity_header("Fresh fact.\nFresh fact.\nOther fact.")
    assert "Do not include" not in text
    assert text.count("Fresh fact.") == 1
    assert "Zoe uses they/them." in text
    assert strip_identity_header(text + "Letter body") == "Letter body"
    path.write_text("A line of identity.\n" * 1000)
    text = cfg.identity_header()
    assert len(text) < 6200
    assert "truncated" in text
    assert text.rstrip().endswith(IDENTITY_END)
    assert strip_identity_header(IDENTITY_HEADING + " incomplete") == IDENTITY_HEADING + " incomplete"


def test_lane_selection_uses_origin_metadata():
    cfg = CompanionConfig({"lanes": [
        {"name": "room", "match": {"chat_ids": ["parent"]}, "note": "Keep our register."},
        {"name": "named", "match": {"name_contains": ["hangout"]}, "note": "Keep the jokes."},
    ]})
    class DB:
        def get_session(self, sid):
            return {"origin_json": '{"parent_chat_id":"parent"}', "chat_id": "thread"}
    assert cfg.lane_note(session_origin(DB(), "s")) == "LANE (room): Keep our register."
    assert "Keep the jokes" in cfg.lane_note({"display_name": "THE HANGOUT"})
    assert cfg.lane_note(None) == ""
    class BrokenDB:
        def get_session(self, sid):
            raise OSError("offline")
    assert session_origin(BrokenDB(), "s") is None


def test_header_failure_does_not_block_summary(monkeypatch):
    c = ContextCompressor(model="test", config_context_length=200000, quiet_mode=True)
    c._companion = CompanionConfig({"enabled": True, "agent_name": "Aleph", "user_name": "Zoe"})
    def fail(*args):
        raise OSError("fixture: unavailable identity")
    monkeypatch.setattr(c._companion, "identity_header", fail)
    assert c._companion_prepend_identity("body") == "body"


def test_names_change_only_record_labels_not_quoted_markers():
    c = ContextCompressor(model="test", config_context_length=200000, quiet_mode=True)
    c._companion = CompanionConfig({"enabled": True, "agent_name": "Aleph", "user_name": "Zoe"})
    literal = "A quoted label:\n\n[USER]: don't rewrite this"
    records = c._serialize_records_for_summary([
        {"role": "user", "content": "[Other speaker]: their words"},
        {"role": "assistant", "content": literal},
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
    ])
    assert records == ["[ZOE]: [Other speaker]: their words", "[ALEPH]: " + literal,
                       "[TOOL RESULT call-1]: result"]
