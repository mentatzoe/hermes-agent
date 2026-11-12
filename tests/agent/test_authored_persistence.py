from unittest.mock import patch

import pytest

from agent.authored_turn_context import capture, snapshots, block, render
from agent.turn_context import _stamp_api_content_sidecar
from hermes_state import SessionDB
from tests.agent.test_api_content_row_addressed_backfill import _RealPersistenceAgent
from tests.agent.test_authored_admission import configure
from tests.agent.test_api_content_sidecar import _build


def test_early_flush_backfills_exact_row_and_metadata(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="cli")
    agent = configure(_RealPersistenceAgent(db), tmp_path)
    staged = {"role": "user", "content": "hello", "display_metadata": {"actor": "human", "other": {"keep": True}}}
    agent._pending_cli_user_message = staged
    agent._flush_messages_to_session_db([staged], None)
    first_id = staged["_row_id"]
    db.append_message("s1", "user", content="hello", display_metadata={"other": "another occurrence"})
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        ctx = _build(agent)
    row = ctx.messages[ctx.current_turn_user_idx]
    stored = {r["id"]: r for r in db.get_messages("s1")}
    assert stored[first_id]["display_metadata"] == row["display_metadata"]
    assert stored[first_id]["content"] == "hello"
    assert block(snapshots(row)[0]) in stored[first_id]["api_content"]
    assert not snapshots(list(stored.values())[-1])
    assert snapshots(db.get_messages_as_conversation("s1")[0]) == snapshots(row)
    db.close()


@pytest.mark.parametrize("content", ["hello", [{"type": "text", "text": "hello"}]])
def test_row_addressed_snapshot_guard_and_metadata_merge(tmp_path, content):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="cli")
    db.create_session("other", source="cli")
    db.append_message("s1", "user", content=content, display_metadata={"actor": "human"})
    row_id = db.get_messages("s1")[0]["id"]
    agent = configure(_RealPersistenceAgent(db), tmp_path)
    row = {"role": "user", "content": content}
    capture(agent, row)
    values = snapshots(row)
    assert db.set_message_authored_snapshot("other", row_id, content, values) == 0
    assert db.set_message_authored_snapshot("s1", row_id, "wrong", values) == 0
    assert db.set_message_authored_snapshot("s1", row_id, content, values) == 1
    # A concurrent metadata writer's unrelated key survives the narrow JSON merge.
    db._write_rowcount("UPDATE messages SET display_metadata=json_set(display_metadata, '$.later', 1) WHERE id=?", (row_id,))
    assert db.set_message_authored_snapshot("s1", row_id, content, values) == 1
    assert db.get_messages("s1")[0]["display_metadata"]["later"] == 1
    assert db.get_messages("s1")[0]["display_metadata"]["actor"] == "human"
    db.deactivate_message("s1", row_id)
    assert db.set_message_authored_snapshot("s1", row_id, content, values) == 0
    db.close()


def test_midturn_flush_persists_wire_body_without_altering_words(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="cli")
    agent = configure(_RealPersistenceAgent(db), tmp_path)
    row = {"role": "user", "content": "correction", "display_metadata": {"actor": "human"}}
    capture(agent, row)
    assert agent._flush_messages_to_session_db([row], None)
    stored = db.get_messages("s1")[0]
    assert stored["content"] == "correction"
    assert stored["api_content"] == render(row, row["content"])
    assert stored["display_metadata"] == row["display_metadata"]
    db.close()
