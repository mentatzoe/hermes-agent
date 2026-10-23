"""Admission journal I/O cannot race teardown or replay into duplicate ownership."""

import asyncio
from pathlib import Path
import subprocess
import sys
import threading
from unittest.mock import AsyncMock

import pytest

from gateway import restart_notifications as notices
from gateway.platforms.event import MessageEvent, ProcessingOutcome
from gateway.turn_context import TurnContext
from gateway.wake import deliver_wake
from tests.gateway.test_kanban_wake_acceptance import drain
from tests.gateway.test_restart_internal_notifications import route


@pytest.mark.asyncio
@pytest.mark.parametrize("admission_await", ["lookup", "save"])
async def test_admission_commits_before_receipt_despite_restart(tmp_path, monkeypatch, admission_await):
    old, adapter, source, key, _, old_received = route(tmp_path, restarting=False)
    entered, release = threading.Event(), threading.Event()
    original_lookup, original_save = old.session_store.lookup_by_session_key, notices._save

    def lookup(session_key):
        entered.set()
        assert release.wait(10)
        return original_lookup(session_key)

    def save(*args):
        original_save(*args)
        entered.set()  # committed, but the admission await has not yet returned
        assert release.wait(10)

    if admission_await == "lookup":
        monkeypatch.setattr(old.session_store, "lookup_by_session_key", lookup)
    else:
        monkeypatch.setattr(notices, "_save", save)
    admission = asyncio.create_task(deliver_wake(adapter, text="admitted across restart", source=source))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert not adapter._background_tasks
        # Even a replay snapshot taken after the commit cannot steal intake's claim.
        await notices.replay_pending(old)
        assert not adapter._background_tasks
        old._draining = old._restart_requested = True
        await adapter.cancel_background_tasks()
    finally:
        release.set()
    await admission
    assert adapter.wire == [] and old_received == []
    assert len(notices._pending(tmp_path / "restart_notifications" / "pending.sqlite3")) == 1
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    await notices.replay_pending(new)
    await drain(fresh)
    await notices.replay_pending(new)
    await drain(fresh)
    assert [item[0] for item in received] == ["admitted across restart"]


def test_restart_notifications_can_be_imported_before_platform_adapters():
    result = subprocess.run(
        [sys.executable, "-I", "-c", "import os, sys; sys.path.insert(0, os.getcwd()); "
         "import gateway.restart_notifications; import gateway.platforms.base_shutdown"],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_queued_followup_hands_off_if_restart_starts_during_preparation(tmp_path, monkeypatch):
    old, adapter, source, key, session_id, _ = route(tmp_path, restarting=False)
    entered, release = asyncio.Event(), asyncio.Event()
    phases = []

    async def cache_refresh(*args):
        entered.set()
        await release.wait()

    async def processing_hook(adapter, event, phase, *outcome):
        phases.append((phase, outcome))

    import gateway.run_turn_followup_ack as ack
    monkeypatch.setattr(ack, "_run_followup_processing_hook", processing_hook)
    old._refresh_agent_cache_message_count = cache_refresh
    old._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="late followup")
    old._is_goal_continuation_event = lambda event: False
    old._pinned_channel_inputs = lambda key, prompt, source, **kwargs: (prompt, source)
    old._reply_anchor_for_event = lambda event: None
    old._intake_adapter_for = lambda source: adapter
    old._run_agent = AsyncMock(return_value={})
    ctx = TurnContext(source=source, session_key=key, session_id=session_id,
                      run_generation=1, history=[], context_prompt="context")
    event = MessageEvent(text="late followup", source=source, internal=True)
    result = {"final_response": "done", "messages": [], "interrupted": True}
    task = asyncio.create_task(old._run_agent_queued_followup(
        ctx, adapter, event.text, event, "done", result, None))
    await asyncio.wait_for(entered.wait(), 2)
    old._draining = old._restart_requested = True
    release.set()
    assert await task is result
    old._run_agent.assert_not_awaited()
    assert phases == [("on_processing_start", ()),
                      ("on_processing_complete", (ProcessingOutcome.CANCELLED,))]
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    await notices.replay_pending(new)
    await drain(fresh)
    await notices.replay_pending(new)
    await drain(fresh)
    assert [item[0] for item in received] == ["late followup"]
