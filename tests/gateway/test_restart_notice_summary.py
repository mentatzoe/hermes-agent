"""Best-effort restart wake summaries, not per-notice redelivery (aleph-vault#241)."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.platforms.event import MessageEvent
from gateway.session import SessionStore
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def make_runner(home):
    runner, adapter = make_restart_runner()
    runner.session_store = SessionStore(home / "sessions", runner.config)
    adapter.gateway_runner = runner
    runner._hm_pending_reply_intercepts = AsyncMock(return_value=None)
    runner._hm_evict_idle_stale_agent = Mock()
    runner._hm_evict_reaped_agent = Mock()
    runner._hm_resolve_command = AsyncMock(return_value=(False, None, None, None))
    runner._hm_dispatch_canonical_command = AsyncMock(return_value=(False, None))
    adapter.set_message_handler(runner._handle_message)
    return runner, adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("active", [False, True], ids=["idle", "busy"])
async def test_restart_notices_stay_silent_and_boot_sends_one_summary(tmp_path, active):
    runner, adapter = make_runner(tmp_path)
    source = make_restart_source()
    entry = runner.session_store.get_or_create_session(source)
    runner._restart_requested = runner._draining = True
    if active:
        # The real busy callback runs without the cold-path message handler.
        adapter._active_sessions[entry.session_key] = asyncio.Event()
        adapter._heal_stale_session_lock = Mock()
    for kind, identity in [("kanban", "t_example"), ("process", "proc_example"),
                           ("cron", "daily"), ("delegation", "deleg_example")]:
        event = MessageEvent(text="Private result text must not be saved", source=source, internal=True,
                             metadata={"restart_notice": {"kind": kind, "id": identity}})
        await adapter.handle_message(event)
    await asyncio.gather(*list(adapter._background_tasks))
    assert adapter.sent == []
    runner.session_store.close_all_db_handles()

    # A new runner and real on-disk session store, with only the model boundary replaced.
    fresh, fresh_adapter = make_runner(tmp_path)
    received = []

    async def handle(event):
        received.append(event)
        return None

    fresh_adapter.set_message_handler(handle)
    fresh._await_startup_warmup = AsyncMock()
    fresh._startup_restore_in_progress = True
    await fresh._finish_startup_restore()
    await asyncio.gather(*list(fresh_adapter._background_tasks))
    assert len(received) == 1
    event = received[0]
    assert event.internal and not event.allow_gateway_control
    assert event.metadata["gateway_session_key"] == entry.session_key
    assert event.metadata["gateway_session_id"] == entry.session_id
    for kind, identity in [("kanban", "t_example"), ("process", "proc_example"),
                           ("cron", "daily"), ("delegation", "deleg_example")]:
        assert kind in event.text and identity in event.text
    assert "UTC" in event.text and "board" in event.text and "process" in event.text
    assert "Private result text" not in event.text
    assert fresh_adapter.sent == []
    await fresh._finish_startup_restore()
    assert len(received) == 1
    fresh.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_human_message_keeps_restart_busy_reply(tmp_path):
    runner, _adapter = make_runner(tmp_path)
    runner._restart_requested = runner._draining = True
    reply = await runner._handle_message(MessageEvent(text="Are you there?", source=make_restart_source()))
    assert "restarting" in reply and "not accepting new work" in reply
    runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_failed_summary_write_still_never_replies_to_notice(tmp_path, monkeypatch):
    runner, adapter = make_runner(tmp_path)
    runner._restart_requested = runner._draining = True
    source = make_restart_source()
    runner.session_store.get_or_create_session(source)
    from gateway import restart_notice_summary
    monkeypatch.setattr(restart_notice_summary, "_append", Mock(side_effect=OSError("disk full")))
    await adapter.handle_message(MessageEvent(text="a notice", source=source, internal=True))
    await asyncio.gather(*list(adapter._background_tasks))
    assert adapter.sent == []
    runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_real_kanban_and_process_producers_are_summarized(tmp_path):
    from gateway.config import Platform
    from gateway.kanban_watchers_notifier import _KanbanNotification
    from gateway.restart_notice_summary import _path

    runner, adapter = make_runner(tmp_path)
    source = make_restart_source()
    entry = runner.session_store.get_or_create_session(source)
    runner._restart_requested = runner._draining = True
    notification = _KanbanNotification(runner, {
        "board": "example", "sub": {"platform": "telegram", "chat_id": source.chat_id,
            "task_id": "t_aabbccdd", "chat_type": "dm", "user_id": source.user_id, "delivery_mode": "wake"},
        "task": SimpleNamespace(title="Example", assignee="worker", session_id=entry.session_id),
        "events": [SimpleNamespace(kind="completed")],
    }, platform_cls=Platform, sub_fail_counts={})
    notification.adapter, notification.plat = adapter, Platform.TELEGRAM
    notification.build_wake_text()
    await notification.wake()
    assert await runner._inject_watch_notification("Process proc_aabbccdd exited", {
        "type": "completion", "session_id": "proc_aabbccdd", "session_key": entry.session_key,
        "platform": "telegram", "chat_id": source.chat_id, "chat_type": "dm",
    }) is True
    await asyncio.gather(*list(adapter._background_tasks))
    rows = [json.loads(line) for line in _path(runner).read_text(encoding="utf-8-sig").splitlines()]
    items = [item for row in rows for item in row["items"]]
    assert {"kind": "kanban", "id": "t_aabbccdd"} in items
    assert {"kind": "process", "id": "proc_aabbccdd"} in items
    assert all(row["session_key"] == entry.session_key and row["time"] for row in rows)
    assert adapter.sent == []
    runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_normal_internal_notice_still_runs_without_restart(tmp_path):
    runner, adapter = make_runner(tmp_path)
    received = []

    async def handle(event):
        received.append(event)

    adapter.set_message_handler(handle)
    event = MessageEvent(text="normal notification", source=make_restart_source(), internal=True)
    await adapter.handle_message(event)
    await asyncio.gather(*list(adapter._background_tasks))
    assert received == [event]
    assert not (tmp_path / "restart-notice-summary.jsonl").exists()
    runner.session_store.close_all_db_handles()
