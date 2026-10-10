"""Old inbound backlog stays on disk; transcript-spool recovery is unaffected."""
import asyncio
import json
import threading
import time
from unittest.mock import AsyncMock, MagicMock
import pytest
from gateway.shutdown_flush import recover_pending_to_db


@pytest.mark.parametrize('age,recovered', [(6 * 3600, 1), (6 * 3600 + 1, 0), (7 * 3600, 0)])
def test_pending_age_boundary_preserves_stale_files(tmp_path, monkeypatch, age, recovered):
    monkeypatch.setattr('gateway.shutdown_flush._get_flush_dir', lambda: tmp_path)
    monkeypatch.setattr('gateway.shutdown_flush.time.time', lambda: 100000)
    path = tmp_path / 'pending-old.json'
    path.write_text(json.dumps({'session_key': 'k', 'ts': 100000 - age,
                               'data': {'session_id': 's', 'text': 'old'}}))
    db = MagicMock()
    assert recover_pending_to_db(db) == recovered
    assert path.exists() is (not recovered)
    assert db.append_message.call_count == recovered


@pytest.mark.asyncio
async def test_startup_recovery_does_not_block_loop_or_overtake_resume(monkeypatch):
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner._start_post_connect_services = AsyncMock()
    runner._await_startup_boot_sends = AsyncMock()
    runner._finish_startup_restore = AsyncMock()
    runner._schedule_recovered_process_watchers = MagicMock()
    started, progressed = threading.Event(), threading.Event()
    blocked = []
    order = []

    def recover(_runner):
        started.set()
        if not progressed.wait(2):
            blocked.append(True)
        order.append('recover')

    async def witness():
        deadline = time.monotonic() + 5
        while not started.is_set():
            assert time.monotonic() < deadline
            await asyncio.sleep(0)
        progressed.set()

    monkeypatch.setattr('gateway.shutdown_flush.recover_gateway_pending', recover)
    monkeypatch.setattr('gateway.run._restart_notification_pending', lambda: False)
    monkeypatch.setattr('gateway.run._planned_restart_notification_pending', lambda: False)
    runner._schedule_resume_pending_sessions = lambda: order.append('resume')
    await asyncio.gather(runner._start_finish_wiring(0), witness())
    assert blocked == []
    assert order == ['recover', 'resume']
