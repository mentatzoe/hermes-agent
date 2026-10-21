"""Independent review regressions for restart windows (aleph-vault#221)."""
import asyncio
import json
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from cron import jobs, delivery_queue
from cron.scheduler_provider import _profile_cron_scope
from cron.scheduler_ownership import register_ticked_homes
from gateway import lifecycle_ledger as ledger
from hermes_constants import get_hermes_home


def test_unset_home_restart_from_secondary_records_and_reads_launch_ledger(tmp_path, monkeypatch):
    import hermes_constants as constants
    from gateway.lifecycle_cron import stop_dispatch

    root, satellite = tmp_path / "root", tmp_path / "root/profiles/satellite"
    satellite.mkdir(parents=True)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr(constants, "_get_platform_default_hermes_home", lambda: root)
    clock = [datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: clock[0])
    monkeypatch.setattr(ledger.time, "time", lambda: clock[0].timestamp())
    ledger.record_startup()
    with _profile_cron_scope(satellite):
        job = jobs.create_job(prompt="check", schedule=clock[0].replace(hour=10).isoformat())
    register_ticked_homes([satellite])
    try:
        clock[0] = clock[0].replace(minute=42)
        with _profile_cron_scope(satellite):
            stop_dispatch(SimpleNamespace())  # /restart can arrive inside the secondary scope
        sentinel = json.loads(ledger.get_lifecycle_sentinel_path().read_text())
        assert sentinel["dispatch_stopped_at"] == clock[0].timestamp()
        assert not ledger.get_lifecycle_sentinel_path(satellite).exists()
        ledger.mark_exited(75)
        clock[0] = clock[0].replace(hour=10, minute=13)
        ledger.record_startup()
        with _profile_cron_scope(satellite):
            assert [j["id"] for j in jobs.get_due_jobs()] == [job["id"]]
    finally:
        register_ticked_homes([])


@pytest.mark.parametrize("named_launch", [False, True])
def test_skipped_secondary_notice_drains_with_adapter_multiplex_disabled(tmp_path, monkeypatch, named_launch):
    import hermes_yaml as yaml
    import gateway.run as run
    from gateway.config import Platform
    from cron.occurrences import scheduled_instant
    from cron import scheduler

    root = tmp_path / "root"
    satellite = root / "profiles/satellite"
    satellite.mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump({"gateway": {
        "multiplex_profiles": False, "profile_routes": [
            {"platform": "discord", "chat_id": "C1", "guild_id": "G1", "profile": "satellite"}]}}))
    (satellite / "config.yaml").write_text(yaml.safe_dump({"platforms": {
        "discord": {"home_channel": "C1"}}}))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    clock = [datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: clock[0])
    with _profile_cron_scope(satellite):
        job = jobs.create_job(prompt="check", schedule=clock[0].replace(hour=10).isoformat(), deliver="discord:C1")
        clock[0] = clock[0].replace(hour=10, minute=13)
        assert jobs.get_due_jobs() == []
        key = f"oneshot-missed:{job['id']}:{scheduled_instant(job['next_run_at'])}"
        assert delivery_queue.get_status(key)["status"] == "pending"
    launch_name = "custom" if named_launch else "default"
    monkeypatch.setattr(run, "_cron_tick_profile_homes", lambda cfg: [(launch_name, root), ("satellite", satellite)])
    monkeypatch.setattr(run, "_cron_profile_gate", lambda *a: True)
    monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"wrap_response": False}})
    sent = []

    class Primary:
        async def send(self, chat_id, content, metadata=None):
            sent.append((chat_id, content))
            return {"success": True, "message_id": "m1"}

    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False),
                             _primary_profile_name=launch_name, _profile_adapters={})
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        run._drain_restart_safe_cron_deliveries({Platform.DISCORD: Primary()}, loop, runner)
        with _profile_cron_scope(satellite):
            assert delivery_queue.get_status(key)["status"] == "delivered"
        assert len(sent) == 1 and sent[0][0] == "C1" and "skipped" in sent[0][1]
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()
