"""Stateless late one-shots use normal claims, execution and profile delivery."""
from datetime import datetime, timedelta, timezone

import pytest

from cron import jobs, scheduler


@pytest.fixture
def rig(tmp_path, monkeypatch):
    clock = [datetime(2026, 10, 21, 12, tzinfo=timezone.utc)]
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_REAL_HOME", str(tmp_path))
    monkeypatch.setattr(jobs, "_hermes_now", lambda: clock[0])
    monkeypatch.setattr(scheduler, "_hermes_now", lambda: clock[0])
    from cron import scheduler_prompt
    monkeypatch.setattr(scheduler_prompt, "_hermes_now", lambda: clock[0])
    (tmp_path / "config.yaml").write_text(
        "cron:\n  oneshot_max_lateness_seconds: 21600\n  notify_missed_oneshots: true\n")
    for name in ("_maybe_reap_dead_owners", "_maybe_run_worktree_maintenance", "_sweep_mcp_orphans"):
        monkeypatch.setattr(scheduler, name, lambda *a, **kw: None)
    monkeypatch.setattr(scheduler, "_should_yield_tick_to_fresh_gateway", lambda: None)
    from cron import bot_chat_delivery
    monkeypatch.setattr(bot_chat_delivery, "drain", lambda: None)
    runs, sent = [], []

    def run(job, **kwargs):
        if job.get("no_agent"):
            return scheduler._run_no_agent_job(job, job["id"], job["name"], None)
        prompt = scheduler._build_job_prompt(job)
        runs.append(prompt)
        return True, prompt, "reminder result", None

    monkeypatch.setattr(scheduler, "run_job", run)

    def deliver(job, content, adapters=None, loop=None, **kwargs):
        from hermes_constants import get_hermes_home
        from cron.scheduler_delivery import _resolve_delivery_targets
        sent.append((content, get_hermes_home(), adapters, _resolve_delivery_targets(job, **kwargs)))
        return None

    monkeypatch.setattr(scheduler, "_deliver_result", deliver)
    with jobs.use_cron_store(tmp_path):
        yield tmp_path, clock, runs, sent


def late_job(rig, seconds=1800, **kwargs):
    home, clock, _, _ = rig
    job = jobs.create_job(prompt="Check today's plan", schedule="in 1h", name="reminder",
                          model="fixture", deliver="discord:123", **kwargs)
    rows = jobs.load_jobs()
    row = next(r for r in rows if r["id"] == job["id"])
    due = (clock[0] - timedelta(seconds=seconds)).isoformat()
    row.update(schedule={"kind": "once", "run_at": due}, next_run_at=due)
    jobs.save_jobs(rows)
    return job, due


def test_restart_wait_runs_once_with_note(rig):
    _, due = late_job(rig)
    assert scheduler.tick(verbose=False) == 1
    assert len(rig[2]) == 1
    assert rig[2][0].startswith("This one-off job is running late")
    assert due in rig[2][0] and "1800" in rig[2][0]
    assert "time-bound" in rig[2][0] and "silent" in rig[2][0]
    assert scheduler.tick(verbose=False) == 0
    assert len(rig[2]) == 1


def test_seven_hours_late_is_skipped_once(rig):
    _, due = late_job(rig, 7 * 3600)
    assert scheduler.tick(verbose=False) == 0
    assert not rig[2]
    assert len(rig[3]) == 1
    content = rig[3][0][0]
    assert "Skipped" in content and "reminder" in content and due in content and "25200" in content
    assert scheduler.tick(verbose=False) == 0
    assert len(rig[3]) == 1
    assert not jobs.load_jobs()


@pytest.mark.parametrize("stdout,visible", [
    ("result", True), ("", False), ('{"wakeAgent": false}', False), ("[SILENT]", False),
])
def test_script_output_late_prefix(rig, monkeypatch, stdout, visible):
    _, due = late_job(rig, no_agent=True, script="fixture.py")
    monkeypatch.setattr(scheduler, "_run_job_script_with_claim_heartbeat", lambda *a, **kw: (True, stdout))
    assert scheduler.tick(verbose=False) == 1
    if visible:
        content = rig[3][0][0]
        assert content.startswith("This one-off job is running late")
        assert due in content and content.endswith("result")
    else:
        assert not rig[3]


def test_schedule_edited_before_worker_uses_current_time(rig, monkeypatch):
    job, old_due = late_job(rig)
    claim = scheduler.claim_job_for_fire
    edited_due = (rig[1][0] + timedelta(minutes=5)).isoformat()

    def edit_then_claim(*args, **kwargs):
        jobs.update_job(job["id"], {"schedule": edited_due})
        rig[1][0] += timedelta(minutes=10)
        return claim(*args, **kwargs)

    monkeypatch.setattr(scheduler, "claim_job_for_fire", edit_then_claim)
    assert scheduler.tick(verbose=False) == 1
    assert len(rig[2]) == 1
    assert edited_due in rig[2][0] and old_due not in rig[2][0]
    assert "300" in rig[2][0]
    assert scheduler.tick(verbose=False) == 0


def test_defaults_remain_diagnostic_only(rig):
    (rig[0] / "config.yaml").write_text("cron: {}\n")
    late_job(rig)
    assert scheduler.tick(verbose=False) == 0
    assert not rig[2] and not rig[3]
    assert not jobs.load_jobs()


@pytest.mark.parametrize("catch_up", [True, False])
def test_recurring_unchanged(rig, catch_up):
    with (rig[0] / "config.yaml").open("a") as f:
        f.write(f"  catch_up_missed: {str(catch_up).lower()}\n")
    job = jobs.create_job(prompt="recurring", schedule="every 1h", model="fixture", deliver="local")
    rows = jobs.load_jobs()
    rows[0]["next_run_at"] = (rig[1][0] - timedelta(hours=7)).isoformat()
    jobs.save_jobs(rows)
    assert scheduler.tick(verbose=False) == int(catch_up)
    assert all("one-off job is running late" not in prompt for prompt in rig[2])
    assert all("Skipped" not in item[0] for item in rig[3])
    assert jobs.get_job(job["id"])


@pytest.mark.parametrize("target_profile", ["default", "secondary"])
def test_named_launch_multiplex_uses_actual_profile_delivery(rig, monkeypatch, target_profile):
    import asyncio
    from concurrent.futures import Future
    import threading
    from unittest.mock import MagicMock
    from cron.scheduler_provider import InProcessCronScheduler, _profile_cron_scope
    from cron.scheduler_delivery import _deliver_result
    from gateway.config import Platform, PlatformConfig
    from hermes_constants import get_hermes_home

    root = rig[0]
    launch = root / "profiles" / "named"
    secondary = root / "profiles" / "secondary"
    for home in (launch, secondary):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text((root / "config.yaml").read_text())
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("HERMES_PROFILE", "named")
    target_home = root if target_profile == "default" else secondary
    with _profile_cron_scope(target_home):
        late_job(rig)
        late_job(rig, 7 * 3600)

    delivered = []

    def adapter(owner):
        result = MagicMock()

        async def send(chat_id, content, metadata=None):
            delivered.append((owner, chat_id, content, get_hermes_home()))
            return {"success": True, "message_id": "fixture"}

        result.send = send
        return result

    primary, named, satellite = adapter("default"), adapter("named"), adapter("secondary")
    config = MagicMock()
    config.platforms = {Platform.DISCORD: PlatformConfig(enabled=True)}
    config.get_home_channel = lambda p: None
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr(scheduler, "_deliver_result", _deliver_result)

    def schedule(coro, loop):
        future = Future()
        future.set_result(asyncio.run(coro))
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", schedule)
    loop = MagicMock()
    loop.is_running.return_value = True
    stop = threading.Event()
    real_tick = scheduler.tick

    def one_cycle(**kwargs):
        kwargs["sync"] = True
        result = real_tick(**kwargs)
        if get_hermes_home() == secondary:
            stop.set()
        return result

    monkeypatch.setattr(scheduler, "tick", one_cycle)
    provider = InProcessCronScheduler()
    provider.start(stop, interval=0, loop=loop,
                   profile_homes=[("default", root), ("named", launch), ("secondary", secondary)],
                   adapters={Platform.DISCORD: primary}, default_profile="default",
                   profile_adapters={"named": {Platform.DISCORD: named},
                                     "secondary": {Platform.DISCORD: satellite}})
    assert len(rig[2]) == 1
    assert len(delivered) == 2
    assert any("Skipped" in row[2] for row in delivered)
    for owner, chat, _, home in delivered:
        assert owner == target_profile and chat == "123" and home == target_home


@pytest.mark.parametrize("seconds,expected", [(120, 1), (121, 1), (21600, 1), (21601, 0)])
def test_cap_boundaries(rig, seconds, expected):
    late_job(rig, seconds)
    assert scheduler.tick(verbose=False) == expected
    assert len(rig[2]) == expected
    assert len(rig[3]) == 1


@pytest.mark.parametrize("value", ["0", "-1", "garbage", ".nan", ".inf"])
def test_invalid_or_disabled_cap_does_not_run(rig, value):
    (rig[0] / "config.yaml").write_text(
        f"cron:\n  oneshot_max_lateness_seconds: {value}\n  notify_missed_oneshots: true\n")
    late_job(rig)
    assert scheduler.tick(verbose=False) == 0
    assert not rig[2] and len(rig[3]) == 1


@pytest.mark.parametrize("manual", [False, True])
def test_forced_and_manual_fire_have_no_late_note(rig, manual):
    job, _ = late_job(rig, 7 * 3600)
    claimed = jobs.claim_job_for_fire(job["id"], force=not manual, manual=manual, return_job=True)
    assert scheduler.run_one_job(claimed)
    assert len(rig[2]) == 1
    assert "one-off job is running late" not in rig[2][0]
    assert all("Skipped" not in row[0] for row in rig[3])


def test_store_outage_recovery_stays_exempt(rig, monkeypatch):
    late_job(rig, 7 * 3600)
    monkeypatch.setattr(jobs.store_health, "outage_covers", lambda *a: True)
    assert scheduler.tick(verbose=False) == 1
    assert "one-off job is running late" not in rig[2][0]
    assert all("Skipped" not in row[0] for row in rig[3])


def test_skip_notice_waits_for_durable_retirement(rig, monkeypatch):
    import errno
    late_job(rig, 7 * 3600)
    save = jobs.save_jobs

    def denied(*args, **kwargs):
        raise OSError(errno.EACCES, "fixture denied write")

    monkeypatch.setattr(jobs, "save_jobs", denied)
    assert scheduler.tick(verbose=False) == 0
    assert not rig[3] and jobs.load_jobs()
    monkeypatch.setattr(jobs, "save_jobs", save)
    assert scheduler.tick(verbose=False) == 0
    assert len(rig[3]) == 1 and not jobs.load_jobs()


def test_skip_uses_failure_override_not_other_lane(rig):
    job, _ = late_job(rig, 7 * 3600)
    jobs.update_job(job["id"], {"failure_deliver": "discord:456"})
    assert scheduler.tick(verbose=False) == 0
    assert rig[3][0][3][0]["chat_id"] == "456"
