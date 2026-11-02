"""Post-update relaunch verification polls every paused target at once (#132338 review, kshitij).

Each relaunched profile / unmapped gateway is retired only on its own stable readiness, and before
this every target ran its own confirmation window in turn: N targets cost N windows (5 profiles,
10 s) where the fleet-wide wait had cost one. One loop keeps the per-target crediting.

Real ``_verify_relaunched_gateways_alive`` and its poll; the seam is the PID-file read that says
which gateway each profile home runs (``get_running_pid``), so no process is spawned or signalled.
"""

from __future__ import annotations

import pytest

from gateway import status
from hermes_cli import gateway_windows, update_cmd_windows
from hermes_cli.update_cmd_windows import _READY_CONFIRM_S, _verify_relaunched_gateways_alive


class _Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    # Measure logical polling work, not host scheduling or profile-lookup latency.
    # Keep the real time module (and the Linux real-child case) untouched.
    clock = _Clock()
    monkeypatch.setattr(update_cmd_windows, "_time", clock)
    return clock


@pytest.fixture
def running(monkeypatch):
    """PID-file callables by profile, plus the time/name of every real poll's read."""
    table: dict = {}
    probes = []

    def get_pid(path, cleanup_stale=False):
        name = path.parent.name
        probes.append((update_cmd_windows._time.monotonic(), name))
        return (table.get(name) or (lambda: None))()

    monkeypatch.setattr(status, "get_running_pid", get_pid)
    monkeypatch.setattr(status, "_pid_exists", lambda pid: False)  # every old gateway is gone
    monkeypatch.setattr(gateway_windows, "_write_start_attestation", lambda *a, **k: None)
    return table, probes


def test_many_ready_targets_cost_one_confirmation_window(running, clock):
    table, probes = running
    profiles = {f"p{i}": 40000 + i for i in range(6)}
    for i, name in enumerate(profiles):
        table[name] = lambda pid=50000 + i: pid
    token = {"profiles": dict(profiles), "unmapped": []}
    started = clock.monotonic()
    _verify_relaunched_gateways_alive(token, profiles, [])
    elapsed = clock.monotonic() - started
    assert token["relaunched_profiles"] == sorted(profiles) and token["profiles"] == {}
    assert elapsed >= _READY_CONFIRM_S, "targets were credited before surviving the confirmation window"
    assert elapsed < 2 * _READY_CONFIRM_S, f"6 ready targets took {elapsed:.1f}s: verified one after another"
    rounds = sorted({when for when, _ in probes})
    assert rounds[0] == started
    assert rounds[-1] - started >= _READY_CONFIRM_S
    for when in rounds:
        assert sorted(name for at, name in probes if at == when) == sorted(profiles)


def test_each_target_is_credited_only_on_its_own_stable_gateway(running, monkeypatch, clock):
    table, probes = running
    monkeypatch.setattr(update_cmd_windows, "_RELAUNCH_VERIFY_TIMEOUT_S", 3.0)
    born = clock.monotonic()
    table["up"] = lambda: 50001
    table["old"] = lambda: 40002  # only the process the relaunch replaces
    table["flaps"] = lambda: 50003 if clock.monotonic() - born < 1.0 else None  # died during confirmation
    profiles = {"up": 40001, "old": 40002, "flaps": 40003, "never": 40004}
    token = {"profiles": dict(profiles), "unmapped": []}
    with pytest.raises(RuntimeError, match="not verified alive: flaps, never, old"):
        _verify_relaunched_gateways_alive(token, profiles, [])
    assert token["relaunched_profiles"] == ["up"]
    assert sorted(token["profiles"]) == ["flaps", "never", "old"]
    assert max(when for when, name in probes if name == "up") - born >= _READY_CONFIRM_S


@pytest.mark.platforms("linux")
@pytest.mark.spawns_gateway_lookalike  # a stub child that sleeps; reaped below
def test_one_gateway_never_retires_two_unmapped_debts(running, monkeypatch, tmp_path):
    import os
    import subprocess
    import sys

    from hermes_cli import gateway
    monkeypatch.setattr(update_cmd_windows, "_RELAUNCH_VERIFY_TIMEOUT_S", 3.0)
    entry = tmp_path / "bin" / "hermes"
    entry.parent.mkdir()
    entry.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    argv = [sys.executable, str(entry), "gateway", "run"]
    # Pre-home entries (an older updater's token): argv is all that identifies them.
    debts = [{"pid": 40011, "argv": argv}, {"pid": 40012, "argv": argv}]
    child = subprocess.Popen(argv, env=dict(os.environ))
    try:
        monkeypatch.setattr(gateway, "find_gateway_pids", lambda all_profiles=False: [child.pid])
        token = {"profiles": {}, "unmapped": list(debts)}
        with pytest.raises(RuntimeError, match="pid 40012"):
            _verify_relaunched_gateways_alive(token, {}, debts)
        assert token["unmapped"] == [debts[1]], "one relaunched gateway retired both debts"
    finally:
        child.kill()
        child.wait(timeout=10)
