"""Gateway boundary producer and profile scope for restart catch-up (#221)."""
import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from cron import jobs
from gateway import lifecycle_ledger as ledger
from gateway.run_shutdown import GatewayShutdownMixin
from hermes_constants import get_hermes_home


@pytest.fixture
def life(monkeypatch):
    root = get_hermes_home()
    current = [datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: current[0])
    monkeypatch.setattr(ledger.time, "time", lambda: current[0].timestamp())
    monkeypatch.setattr(ledger, "_process_hermes_home", lambda: root)
    monkeypatch.setattr(ledger, "detect_unclean_exit", lambda home=None: None)
    ledger.record_startup()
    return root, current


def read():
    return json.loads(ledger.get_lifecycle_sentinel_path().read_text())


@pytest.mark.asyncio
async def test_real_restart_gate_records_first_stop_before_wait(life):
    root, clock = life
    runner = SimpleNamespace(_restart_task_started=False, _draining=False,
                            _mark_api_runs_shutdown_requested=Mock(),
                            _await_active_work_before_restart=AsyncMock(), stop=AsyncMock())
    clock[0] = clock[0].replace(hour=9, minute=42)
    assert GatewayShutdownMixin.request_restart(runner)
    stopped = clock[0].timestamp()
    assert runner._draining and read()["dispatch_stopped_at"] == stopped
    clock[0] = clock[0].replace(hour=10, minute=12)
    from gateway.lifecycle_cron import stop_dispatch
    stop_dispatch(runner)  # stop() may run much later; don't replace the first boundary
    assert read()["dispatch_stopped_at"] == stopped
    await runner._restart_task
    ledger.mark_exited(75)
    clock[0] = clock[0].replace(minute=13)
    ledger.record_startup()
    assert read()["restart_windows"][0] == {
        "start": stopped, "end": clock[0].timestamp(), "pid": read()["pid"],
        "homes": [str(root.resolve())]}


def test_canceled_drain_does_not_cover_later_downtime(life):
    runner = SimpleNamespace(_external_drain_active=False, _draining=False, _running=True,
                             _active_work_count=lambda: 0, _update_runtime_status=Mock(),
                             _serving_state=lambda: "running")
    GatewayShutdownMixin._enter_external_drain(runner)
    assert read()["dispatch_stopped_at"]
    GatewayShutdownMixin._exit_external_drain(runner)
    assert "dispatch_stopped_at" not in read()
    ledger.mark_exited(0)
    ledger.record_startup()
    assert not read()["restart_windows"]


def test_served_profile_window_does_not_authorize_unserved_home(life, tmp_path):
    from cron.scheduler_ownership import register_ticked_homes
    from cron.scheduler_provider import _profile_cron_scope
    from gateway.lifecycle_cron import stop_dispatch
    root, clock = life
    served, unrelated = tmp_path / "served", tmp_path / "unrelated"
    due = clock[0].replace(hour=10).isoformat()
    for home in (served, unrelated):
        with _profile_cron_scope(home):
            jobs.create_job(prompt="check", schedule=due)
    register_ticked_homes([served])
    try:
        clock[0] = clock[0].replace(hour=9, minute=42)
        stop_dispatch(SimpleNamespace())
        ledger.mark_exited(75)
        clock[0] = clock[0].replace(hour=10, minute=13)
        ledger.record_startup()
        with _profile_cron_scope(served):
            assert len(jobs.get_due_jobs()) == 1
        with _profile_cron_scope(unrelated):
            assert jobs.get_due_jobs() == []
    finally:
        register_ticked_homes([])


@pytest.mark.parametrize("field,value", [("dispatch_stopped_at", None),
                                          ("dispatch_stopped_at", float("nan")),
                                          ("dispatch_stopped_at", 99999999999),
                                          ("dispatch_stopped_pid", -1),
                                          ("start_time", None)])
def test_malformed_boundary_never_authorizes_a_window(life, field, value):
    from gateway.lifecycle_cron import stop_dispatch
    stop_dispatch(SimpleNamespace())
    ledger.mark_exited(0)
    record = read()
    record[field] = value
    record["restart_windows"] = None
    ledger.get_lifecycle_sentinel_path().write_text(json.dumps(record))
    ledger.record_startup()
    assert read()["phase"] == "running"
    assert not read()["restart_windows"]


def test_window_history_is_bounded_without_changing_old_records(life):
    from gateway.lifecycle_cron import stop_dispatch
    root, clock = life
    prior = []
    for hour in range(9, 14):
        clock[0] = clock[0].replace(hour=hour, minute=1)
        stop_dispatch(SimpleNamespace())
        ledger.mark_exited(75)
        clock[0] = clock[0].replace(minute=4)
        ledger.record_startup()
        windows = read()["restart_windows"]
        assert windows[:-1] == prior[-2:]
        assert len(windows) <= 3
        prior = windows
