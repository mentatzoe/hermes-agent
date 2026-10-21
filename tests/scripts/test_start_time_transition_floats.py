"""The drained-transition gate includes float birth records (aleph-vault #233)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "transition_float_check", Path(__file__).resolve().parents[2] / "scripts/start_time_transition_check.py")
check = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check
SPEC.loader.exec_module(check)

BIRTH = 1792487588.691234
PID = 4242


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixtures(home, locks, birth):
    _write(home / "spawn-ledger.json", [{"pid": PID, "create_time": birth,
                                        "spawner_pid": PID, "spawner_create": round(birth, 3)}])
    _write(home / "runtime/active_sessions.json", {"entries": [{"pid": PID, "process_start_time": birth}]})
    _write(home / "runtimes/llamacpp/server.json", {"pid": PID, "create_time": birth,
                                                "owner_pid": PID, "owner_create_time": birth})
    _write(locks / "host-serve.json", {"pid": PID, "createTime": birth})
    _write(home / "state/gateway.lifecycle.json", {"phase": "running", "pid": PID,
                                                "start_time": birth + 10, "create_time": birth})
    _write(home / "logs/update_receipts/update_running.json",
           {"outcome": "running", "pid": PID, "pid_create_time": birth,
            "writer_pid": PID, "writer_create_time": birth})
    _write(home / ".hermes-update-paused-gateways.test.json.1.claim",
           {"owner": {"pid": PID, "ct": f"ct:{birth:.3f}"},
            "claimer": {"pid": PID, "ct": f"ct:{birth:.3f}"},
            "token": {"identities": {str(PID): f"ct:{birth:.3f}"},
                      "posix_units": [{"pid": PID, "ct": round(birth * 100)}]}})
    (home / ".hermes-update-in-progress").write_text(
        f"{PID}\n1792487590\nct:{birth:.3f}\ndelegate:{PID} ct:{birth:.3f}\n", encoding="utf-8")
    launcher = home / "bot-desktop/launcher.pid"
    launcher.parent.mkdir()
    launcher.write_text(f"{PID} {birth!r}", encoding="utf-8")


def _hooks(*, alive=True, birth=BIRTH):
    return {"alive": lambda pid: alive, "raw": lambda pid: int(round(birth * 100)),
            "birth": lambda pid: birth}


FAMILIES = ("spawn-ledger.json:create_time", "spawn-ledger.json:spawner_create",
            "active_sessions.json.entries[0]:process_start_time", "server.json:create_time",
            "server.json:owner_create_time", "host-serve.json:createTime", "gateway.lifecycle.json:create_time",
            "update_running.json:pid_create_time", "update_running.json:writer_create_time",
            "claim.owner:ct", "claim.claimer:ct", "claim.token.identities.4242", "posix_units[0]:ct",
            "update-in-progress:owner", "update-in-progress:delegate", "launcher.pid")


@pytest.mark.parametrize("shift", [0, 1, -1, 3])
def test_float_families_are_checked_in_their_own_units_and_never_written(tmp_path, shift):
    home, locks = tmp_path / "home", tmp_path / "locks"
    _fixtures(home, locks, BIRTH + shift)
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    lines = []
    assert check.run(home, locks, out=lines.append, **_hooks()) == (1 if shift else 0)
    records = [line for line in lines if line.strip().startswith("DIFFERS" if shift else "OK")]
    # A list row adds [0] to the ledger's source; compare a normalized locator, not a count alone.
    normalized = [line.replace("spawn-ledger.json[0]", "spawn-ledger.json") for line in records]
    for family in FAMILIES:
        assert any(family in line for line in normalized), family
    assert {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()} == before


def test_dead_float_owners_are_skipped_but_unreadable_live_owners_fail(tmp_path):
    home, locks = tmp_path / "home", tmp_path / "locks"
    _fixtures(home, locks, BIRTH + 3)
    assert check.run(home, locks, out=lambda _: None, **_hooks(alive=False)) == 0
    hooks = _hooks()
    hooks["birth"] = lambda pid: None
    assert check.run(home, locks, out=lambda _: None, **hooks) == 1
    assert check.run(home, locks, out=lambda _: None, restarting=frozenset({PID}), **hooks) == 0


def test_finished_receipts_and_stopped_lifecycle_do_not_hold_the_gate(tmp_path):
    home = tmp_path / "home"
    _write(home / "logs/update_receipts/update_done.json",
           {"outcome": "success", "pid": PID, "pid_create_time": BIRTH + 3})
    _write(home / "state/gateway.lifecycle.json",
           {"phase": "stopped", "pid": PID, "create_time": BIRTH + 3})
    assert check.collect(home, None) == []


def test_rounded_births_allow_only_the_writers_precision_not_clock_drift():
    for recorded in (BIRTH, round(BIRTH, 3), float(int(BIRTH))):
        rule = check.SECONDS if recorded == BIRTH else check.MILLIS
        record = check.Record("fixture", PID, recorded, rule)
        assert check.judge(record, **_hooks()).status == "ok"
        for shift in (-1, 1, 3):
            shifted = check.Record("fixture", PID, recorded + shift, rule)
            assert check.judge(shifted, **_hooks()).status == "differs"
    # Losing the sub-centisecond remainder is not equality for an exact float guard.
    record = check.Record("fixture", PID, round(BIRTH * 100) / 100, check.SECONDS)
    assert check.judge(record, **_hooks()).status == "differs"


@pytest.mark.parametrize("value", [True, "bad", float("nan"), float("inf"), -1, 10 ** 1000])
def test_invalid_float_births_never_pass_the_transition_gate(value):
    record = check.Record("fixture", PID, value, check.SECONDS)
    assert check.judge(record, **_hooks()).status == "differs"


@pytest.mark.platforms("any")
def test_register_child_full_precision_spawner_passes_and_shifted_birth_fails(tmp_path, monkeypatch):
    import subprocess

    from hermes_cli import process_identity

    home = tmp_path / "home"
    ledger = home / "spawn-ledger.json"
    monkeypatch.setattr(process_identity, "_ledger_path", lambda: ledger)
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    try:
        assert process_identity.register_child(child.pid, "mcp-helper")
        original = ledger.read_bytes()
        lines = []
        assert check.run(home, None, out=lines.append) == 0, "\n".join(lines)
        assert ledger.read_bytes() == original
        row = json.loads(original)[0]
        for shift in (1, -1, 3):
            ledger.write_text(json.dumps([dict(row, spawner_create=row["spawner_create"] + shift)]))
            assert check.run(home, None, out=lambda _: None) == 1
    finally:
        child.communicate(timeout=10)

