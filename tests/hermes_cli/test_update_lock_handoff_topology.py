"""Real handoff topology: update stages adopt, intervening runtime hosts refuse.

The marker-only regressions in test_update_lock used a sleeper child as their
orchestrator. Refs mentatzoe/aleph-vault#260: a real parent bounds the admission
walk without weakening the runtime-host fence or mocking process identity.
"""

import os
import sys
import time

import pytest

from hermes_cli.update_lock import process_create_time
from tests.hermes_cli.update_lock_handoff_support import run_handoff_child


@pytest.fixture
def relay(tmp_path):
    """A harmless process with canonical Hermes argv, never a running service."""
    script = tmp_path / "hermes_cli" / "main.py"
    script.parent.mkdir()
    script.write_text(
        "import subprocess, sys\n"
        "command = sys.argv[sys.argv.index('--') + 1:]\n"
        "raise SystemExit(subprocess.run(command, timeout=60).returncode)\n",
        encoding="utf-8",
    )
    return script


@pytest.mark.spawns_gateway_lookalike  # relay is a reaped stub, not a runtime
@pytest.mark.parametrize("handoff", [True, False], ids=["explicit-handoff", "legacy-ancestry"])
@pytest.mark.parametrize("stage", [("version",), ("gateway", "run"), ("serve",), ("dashboard",)])
def test_non_host_stages_adopt_but_runtime_hosts_refuse(tmp_path, relay, handoff, stage):
    pytest.importorskip("psutil")  # the production runtime-host fence uses psutil
    marker = tmp_path / "marker"
    owner = os.getpid()
    marker.write_text(f"{owner}\n{int(time.time())}\nct:{process_create_time():.3f}\n", encoding="utf-8")
    base = marker.read_bytes()
    runtime_host = stage != ("version",)
    run_handoff_child(marker, f"""
        from hermes_cli import update_lock
        before = marker.read_bytes()
        # Prove both the real relationship and classifier, rather than allowing
        # an identity failure to make the runtime-host refusal test pass.
        assert update_lock.UpdateLock._is_partner({owner})
        assert update_lock._is_ancestor_pid({owner})
        assert update_lock._runtime_host_below({owner}) is {runtime_host}
        lock = UpdateLock(path=marker)
        try:
            assert lock.acquire() is {not runtime_host}, lock.holder
            assert lock.acquired is False
            if {runtime_host}:
                assert lock.holder is not None and lock.holder.pid == {owner}
                assert marker.read_bytes() == before, "refusal must not delegate or rewrite"
            else:
                assert f"delegate:{{os.getpid()}} ct:".encode() in marker.read_bytes()
        finally:
            lock.release()
        assert marker.read_bytes() == before, "the owner retains its exact claim"
    """, via=[sys.executable, "-B", str(relay), *stage, "--"], handoff=handoff)
    assert marker.read_bytes() == base
