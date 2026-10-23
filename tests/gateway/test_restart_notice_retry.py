"""Regression checks for cancellation-safe notice identity and real worker completion."""
import asyncio
import threading

import pytest

from gateway import restart_notifications as notices
from gateway.platforms.event import MessageEvent
from gateway.wake import admit_internal_event
from tests.gateway.test_kanban_wake_acceptance import drain
from tests.gateway.test_restart_internal_notifications import route
from tests.gateway.test_restart_notice_lifecycle import production_turns
from types import SimpleNamespace


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel_after_commit', [False, True])
async def test_same_notice_retry_keeps_one_record(tmp_path, monkeypatch, cancel_after_commit):
    old, adapter, source, _, _, _ = route(tmp_path)
    event = MessageEvent(text='one admitted notice', source=source, internal=True)
    saved, release = threading.Event(), threading.Event()
    original = notices._save

    def blocked_save(*args):
        original(*args)
        saved.set()
        assert release.wait(10)

    monkeypatch.setattr(notices, '_save', blocked_save)
    intake = asyncio.create_task(admit_internal_event(adapter, event))
    try:
        assert await asyncio.to_thread(saved.wait, 5)
        if cancel_after_commit:
            intake.cancel()
            with pytest.raises(asyncio.CancelledError):
                await intake
    finally:
        release.set()
    if not cancel_after_commit:
        await intake
    monkeypatch.setattr(notices, '_save', original)
    # Retry this exact MessageEvent, not a new fabricated producer notice.
    await admit_internal_event(adapter, event)
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    for _ in range(4):
        await notices.replay_pending(new)
        await drain(fresh)
    assert [row[0] for row in received] == ['one admitted notice']


@pytest.mark.asyncio
@pytest.mark.parametrize('timed_out', [False, True])
async def test_timeout_is_not_a_completed_notice_turn(tmp_path, timed_out):
    old, adapter, source, key, sid, _ = route(tmp_path, restarting=False)

    async def pause(*args):
        pass

    async def worker(text):
        pass

    production_turns(old, adapter, source, key, sid, pause, worker)
    unfinished = []
    if timed_out:
        def start(ctx, run_sync):
            future = asyncio.create_task(asyncio.Event().wait())
            unfinished.append(future)
            timeout_fired = threading.Event()
            timeout_fired.set()
            return SimpleNamespace(executor_task=future, agent_timeout=60, timeout_fired=timeout_fired)

        old._run_agent_start_turn_worker = start
        # Keep the real executor-poll and synthetic timeout-result path intact.
        old._run_agent_await_turn_worker = type(old)._run_agent_await_turn_worker.__get__(old)
    await admit_internal_event(adapter, MessageEvent(text='unfinished notice', source=source, internal=True))
    try:
        await drain(adapter)
        assert all(not future.done() for future in unfinished)
    finally:
        for future in unfinished:
            future.cancel()
        await asyncio.gather(*unfinished, return_exceptions=True)
    old._draining = old._restart_requested = True
    await adapter.cancel_background_tasks()
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    for _ in range(3):
        await notices.replay_pending(new)
        await drain(fresh)
    assert [row[0] for row in received] == (['unfinished notice'] if timed_out else [])


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel_after_commit', [False, True])
async def test_process_producer_retry_does_not_duplicate_journal(tmp_path, monkeypatch, cancel_after_commit):
    from collections import OrderedDict
    old, adapter, source, key, _, _ = route(tmp_path)
    old._completion_delivery_lock = threading.Lock()
    old._completion_deliveries_inflight = set()
    old._completion_deliveries_delivered = OrderedDict()
    old._completion_delivery_retention = 2048
    old._build_process_event_source = lambda evt: source
    old._resolve_injection_adapter = lambda platform, target: adapter
    evt = {'type': 'completion', 'session_id': 'review-process', 'started_at': 1,
           'session_key': key, 'exit_code': 0}
    saved, release = threading.Event(), threading.Event()
    original = notices._save

    def blocked_save(*args):
        original(*args)
        saved.set()
        assert release.wait(10)

    monkeypatch.setattr(notices, '_save', blocked_save)
    intake = asyncio.create_task(old._deliver_completion_notification('process completed', evt))
    try:
        assert await asyncio.to_thread(saved.wait, 5)
        if cancel_after_commit:
            intake.cancel()
            with pytest.raises(asyncio.CancelledError):
                await intake
    finally:
        release.set()
    if not cancel_after_commit:
        assert await intake is True
    monkeypatch.setattr(notices, '_save', original)
    assert not old._completion_deliveries_inflight
    # The production producer releases an unacknowledged stable identity for retry.
    await old._deliver_completion_notification('process completed', dict(evt))
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
    assert len(worked) == 1


@pytest.mark.asyncio
async def test_producer_retry_after_completion_and_new_occurrence_are_distinct(tmp_path):
    from gateway.wake import deliver_wake

    old, adapter, source, _, _, received = route(tmp_path, restarting=False)
    identity = ("completion", "stable-process", 1)
    await deliver_wake(adapter, text="one occurrence", source=source, notice_identity=identity)
    await drain(adapter)
    assert len(received) == 1
    new, fresh, fresh_source, _, _, replayed = route(tmp_path, restarting=False)
    # A reconstructed producer retry in a fresh gateway still acknowledges the done row.
    await deliver_wake(fresh, text="one occurrence", source=fresh_source, notice_identity=identity)
    await drain(fresh)
    await notices.replay_pending(new)
    assert replayed == []
    await deliver_wake(fresh, text="new incarnation", source=fresh_source,
                       notice_identity=("completion", "stable-process", 2))
    await drain(fresh)
    assert [row[0] for row in replayed] == ["new incarnation"]

