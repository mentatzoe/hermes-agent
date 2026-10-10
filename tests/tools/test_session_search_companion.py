"""Row 9: recall returns the conversation, not the embedded handoff payload."""
import json

import pytest

from hermes_state import SessionDB
from tools.session_search_tool import session_search, _shape_message


@pytest.mark.parametrize("mode", ["read", "scroll", "fts", "title"])
def test_recall_omits_handoffs_without_dropping_messages(tmp_path, mode):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("history", source="cli")
        db._conn.execute("UPDATE sessions SET title = ? WHERE id = ?", ("Promise archive", "history"))
        db._conn.commit()
        user_id = db.append_message("history", role="user", content="Remember the lantern promise.\n\n[CONTEXT COMPACTION — REFERENCE ONLY]\nMACHINE-HANDOFF " * 1)
        db.append_message("history", role="assistant", content="I remember it.")
        db.append_message("history", role="user", content="[CONTEXT SUMMARY]:\nMACHINE-HANDOFF body")
        args = {"db": db}
        if mode == "read":
            args["session_id"] = "history"
        elif mode == "scroll":
            args.update(session_id="history", around_message_id=user_id)
        else:
            args["query"] = "lantern" if mode == "fts" else "Promise archive"
        result = json.loads(session_search(**args))
        assert result["success"]
        messages = result["messages"] if mode in ("read", "scroll") else result["results"][0]["messages"]
        user = next(m for m in messages if m["id"] == user_id)
        assert user["role"] == "user"
        assert "Remember the lantern promise." in user["content"]
        assert "compaction handoff omitted" in user["content"]
        assert all("MACHINE-HANDOFF" not in (m.get("content") or "") for m in messages)
        if mode == "read":
            assert result["message_count"] == 3
            assert len(messages) == 3
        if mode == "scroll":
            assert user["anchor"]
    finally:
        db.close()


@pytest.mark.parametrize("limit", [1200, 2000, 4000])
def test_bounded_recall_slims_tool_arguments_without_mutating_store(limit):
    args = "x" * 110000
    call = {"id": "call-1", "function": {"name": "write_file", "arguments": args}}
    row = {"id": 42, "role": "assistant", "tool_calls": [call], "content": ""}
    shaped = _shape_message(row, anchor_id=42, max_content_len=limit)
    assert len(shaped["tool_calls"][0]["function"]["arguments"]) < limit
    assert shaped["tool_calls"][0]["id"] == "call-1"
    assert shaped["anchor"]
    assert row["tool_calls"][0]["function"]["arguments"] == args
    assert _shape_message(row)["tool_calls"][0]["function"]["arguments"] == args


@pytest.mark.parametrize("mode", ["read", "scroll", "fts", "title"])
def test_names_follow_the_selected_profile_not_the_first_import(tmp_path, monkeypatch, mode):
    from hermes_cli import profiles
    homes = {name: tmp_path / name for name in ("first", "second")}
    for name, home in homes.items():
        home.mkdir()
        (home / "config.yaml").write_text(json.dumps({"compression": {"companion": {
            "user_name": name + " human", "agent_name": name + " companion"}}}))
        db = SessionDB(home / "state.db")
        db.create_session("history", source="cli")
        db._conn.execute("UPDATE sessions SET title = ? WHERE id = ?", ("Named conversation", "history"))
        db._conn.commit()
        db.append_message("history", role="user", content="Remember the lantern.")
        db.append_message("history", role="assistant", content="The lantern is lit.")
        db.close()
    monkeypatch.setenv("HERMES_HOME", str(homes["first"]))
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name in homes)
    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: homes[name])
    for name in ("first", "second", "first"):
        args = {"profile": name}
        if mode in ("read", "scroll"):
            args["session_id"] = "history"
            if mode == "scroll":
                args["around_message_id"] = 1
        else:
            args["query"] = "lantern" if mode == "fts" else "Named conversation"
        from tools.registry import registry
        payload = registry.dispatch("session_search", args)
        data = json.loads(payload) if isinstance(payload, str) else payload
        assert data["success"]
        messages = data["messages"] if mode in ("read", "scroll") else data["results"][0]["messages"]
        assert [(m["role"], m.get("speaker")) for m in messages] == [
            ("user", name + " human"), ("assistant", name + " companion")]
    assert _shape_message({"role": "user", "content": "ambient"}).get("speaker") == "first human"


@pytest.mark.parametrize("content", [
    "The literal marker `[CONTEXT COMPACTION]` is not a handoff. Keep this sentence.",
    "Here is quoted text:\n> [CONTEXT SUMMARY]: old notes\nKeep the discussion.",
    "Example:\n```text\n[CONTEXT COMPACTION — REFERENCE ONLY]\nexample\n```\nKeep the discussion.",
])
def test_quoted_compaction_markers_remain_conversation(content):
    assert _shape_message({"role": "user", "content": content})["content"] == content
