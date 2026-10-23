"""Warning-split Kanban wakes retain their own occurrence records and handoffs."""

import asyncio
import pytest

from gateway.config import Platform
from gateway.kanban_watchers_notifier import _KanbanNotification, _notifier_collect
from gateway.restart_notifications import replay_pending
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn
from tests.gateway.test_restart_internal_notifications import route
from tests.gateway.test_kanban_wake_acceptance import drain


@pytest.mark.asyncio
@pytest.mark.parametrize("suppress_warnings", [False, True])
async def test_split_kanban_wakes_keep_result_handoff(tmp_path, monkeypatch, suppress_warnings):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    (tmp_path / "config.yaml").write_text("display: {suppress_warning_notifications: " + str(suppress_warnings).lower() + "}")
    old, adapter, _, key, _, _ = route(tmp_path)
    old._kanban_dispatcher_lock_handle = object()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="split wake result", assignee="worker", session_id=key)
        kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="42",
                           user_id="42", chat_type="dm", delivery_mode="wake")
        kb.block_task(conn, tid, reason="temporary diagnostic", kind="transient")
        kb.complete_task(conn, tid, summary="UNIQUE RESULT HANDOFF")
    finally:
        conn.close()
    deliveries = await asyncio.to_thread(_notifier_collect, old, kb,
        notifier_profile=None, gc_due=False, gc_retention_days=30)
    assert len(deliveries) == 1
    assert {"blocked", "completed"} <= {e.kind for e in deliveries[0]["events"]}
    await _KanbanNotification(old, deliveries[0], platform_cls=Platform, sub_fail_counts={}).deliver()
    conn = kbc.connect()
    try:
        assert not kbn.unseen_events_for_sub(conn, task_id=tid, platform="telegram", chat_id="42",
                                             kinds=["blocked", "completed"])[1]
    finally:
        conn.close()
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    for _ in range(4):
        await replay_pending(new)
        await drain(fresh)
    assert any("UNIQUE RESULT HANDOFF" in row[0] for row in received), received
    assert [row[1]["notification_category"] for row in received] == (
        ["diagnostic", "result"] if suppress_warnings else ["result"])
