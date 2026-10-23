"""Each coalesced process occurrence keeps its identity through changed retry batches."""

import asyncio
from collections import OrderedDict
import threading

import pytest

from gateway import restart_notifications as notices
from tests.gateway.test_restart_internal_notifications import route
from tests.gateway.test_restart_notice_lifecycle import production_turns
from tests.gateway.test_kanban_wake_acceptance import drain


def producer(runner, adapter, source):
    runner._completion_delivery_lock = threading.Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    runner._build_process_event_source = lambda evt: source
    runner._resolve_injection_adapter = lambda platform, target: adapter
    runner._completion_notification_batch_window = 0.05


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_mode", ["same", "separate", "overlap"])
async def test_cancelled_batch_retains_each_occurrence_once(tmp_path, monkeypatch, retry_mode):
    old, adapter, source, key, _, _ = route(tmp_path)
    producer(old, adapter, source)
    a = {"type": "completion", "session_id": "process-A", "started_at": 1,
         "session_key": key, "exit_code": 0, "output": "A output"}
    b = {**a, "session_id": "process-B", "output": "B output"}
    c = {**a, "session_id": "process-C", "output": "C output"}
    saved, release = threading.Event(), threading.Event()
    original = notices._save

    def blocked_save(*args):
        result = original(*args)
        saved.set()
        assert release.wait(10)
        return result

    monkeypatch.setattr(notices, "_save", blocked_save)
    first = [asyncio.create_task(old._enqueue_process_completion_notification(name, evt))
             for name, evt in [("A completed", a), ("B completed", b)]]
    try:
        assert await asyncio.to_thread(saved.wait, 5)
        await old._cancel_process_completion_batch_tasks()
        assert await asyncio.gather(*first) == [False, False]
    finally:
        release.set()
    monkeypatch.setattr(notices, "_save", original)
    retry, retry_adapter, retry_source, _, _, _ = route(tmp_path)
    producer(retry, retry_adapter, retry_source)
    assert not adapter._background_tasks
    if retry_mode == "separate":
        assert await retry._enqueue_process_completion_notification("B output", dict(b)) is True
        assert await retry._enqueue_process_completion_notification("A output", dict(a)) is True
    else:
        entries = [("B output", b), ("C output", c)] if retry_mode == "overlap" else [("A output", a), ("B output", b)]
        assert await asyncio.gather(*[
            retry._enqueue_process_completion_notification(name, dict(evt)) for name, evt in entries]) == [True, True]
    new, fresh, source, key, sid, _ = route(tmp_path, restarting=False)
    worked = []

    async def pause(*args):
        pass

    async def worker(text):
        worked.append(text)

    production_turns(new, fresh, source, key, sid, pause, worker)
    for _ in range(4):
        await notices.replay_pending(new)
        await drain(fresh)
    assert sum("B output" in text for text in worked) == 1, worked
    assert len(worked) == (2 if retry_mode == "overlap" else 1), worked
    if retry_mode == "overlap":
        assert sum("C output" in text for text in worked) == 1, worked
