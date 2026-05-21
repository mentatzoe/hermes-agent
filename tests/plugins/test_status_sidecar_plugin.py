"""Tests for the status-sidecar plugin.

The plugin maintains a small durable status ledger ("HEARTBEAT" sidecar) and
appends a ``<system_status>`` block to the user message via the
``pre_llm_call`` hook. Verifies:

  * Library: schema CRUD, atomic writes, TTL staleness, corrupt-file safety.
  * Plugin ``pre_llm_call``: appends block to user-context return, skips when
    stale/missing, never raises.
  * ``status_update`` tool: short-annotation write path, bounded drift signals.
  * Bundled-plugin discovery via ``PluginManager`` (opt-in, not auto-loaded).

The plugin must NEVER mutate the system prompt; hook returns become
``_plugin_user_context`` in ``run_agent.py`` and are appended to the user
message only.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "runtime" / "status_sidecar"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    """Each test gets a fresh HERMES_HOME under tmp_path."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # Prevent accidental load of the developer's real config.yaml.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    yield hermes_home


def _load_lib():
    """Import the plugin's library module fresh."""
    lib_path = PLUGIN_DIR / "status_sidecar.py"
    spec = importlib.util.spec_from_file_location(
        "status_sidecar_under_test", lib_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_plugin_init():
    """Import the plugin __init__ fresh, using the same naming convention as
    ``PluginManager`` so the relative ``from . import status_sidecar`` works.
    """
    plugin_dir = PLUGIN_DIR
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.status_sidecar",
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.status_sidecar"
    mod.__path__ = [str(plugin_dir)]
    sys.modules["hermes_plugins.status_sidecar"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Manifest + layout
# ---------------------------------------------------------------------------

class TestManifest:
    def test_plugin_directory_exists(self):
        assert PLUGIN_DIR.is_dir(), f"plugin dir missing: {PLUGIN_DIR}"
        assert (PLUGIN_DIR / "plugin.yaml").exists()
        assert (PLUGIN_DIR / "__init__.py").exists()
        assert (PLUGIN_DIR / "status_sidecar.py").exists()

    def test_manifest_declares_pre_llm_call(self):
        data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
        assert data["name"] == "status-sidecar"
        assert "pre_llm_call" in data["hooks"]


# ---------------------------------------------------------------------------
# Library: ledger CRUD and atomic writes
# ---------------------------------------------------------------------------

class TestLedgerCrud:
    def test_initial_read_returns_empty(self, _isolate_env):
        ss = _load_lib()
        record = ss.read_status()
        assert record == {} or record is None or record.get("status_updated_at") is None

    def test_write_then_read_round_trip(self, _isolate_env):
        ss = _load_lib()
        ss.write_status(
            active_kanban_task_id="t_abc123",
            active_workspace="/tmp/work",
            last_tool_invocation="kanban_show",
        )
        record = ss.read_status()
        assert record["active_kanban_task_id"] == "t_abc123"
        assert record["active_workspace"] == "/tmp/work"
        assert record["last_tool_invocation"] == "kanban_show"
        assert "status_updated_at" in record
        # Must be a float epoch seconds value.
        assert isinstance(record["status_updated_at"], (int, float))

    def test_write_updates_timestamp_each_call(self, _isolate_env):
        ss = _load_lib()
        ss.write_status(active_kanban_task_id="t_1")
        t1 = ss.read_status()["status_updated_at"]
        time.sleep(0.02)
        ss.write_status(active_kanban_task_id="t_2")
        t2 = ss.read_status()["status_updated_at"]
        assert t2 > t1
        assert ss.read_status()["active_kanban_task_id"] == "t_2"

    def test_drift_signals_are_bounded(self, _isolate_env):
        """Drift-signal ring buffer keeps only the most recent N entries."""
        ss = _load_lib()
        for i in range(20):
            ss.append_drift_signal(f"signal {i}")
        record = ss.read_status()
        signals = record.get("recent_drift_signals") or []
        assert len(signals) <= ss.MAX_DRIFT_SIGNALS
        # Newest at the end (FIFO ring buffer): last one we appended is present.
        assert signals[-1] == f"signal 19"

    def test_drift_signal_length_capped(self, _isolate_env):
        ss = _load_lib()
        long_signal = "x" * 5000
        ss.append_drift_signal(long_signal)
        signals = ss.read_status().get("recent_drift_signals") or []
        assert len(signals[-1]) <= ss.MAX_DRIFT_SIGNAL_CHARS


class TestStaleness:
    def test_fresh_status_passes_ttl_check(self, _isolate_env):
        ss = _load_lib()
        ss.write_status(active_kanban_task_id="t_fresh")
        assert ss.is_fresh(ss.read_status(), ttl_seconds=3600) is True

    def test_stale_status_fails_ttl_check(self, _isolate_env):
        """Manually backdate the timestamp and confirm is_fresh is False."""
        ss = _load_lib()
        ss.write_status(active_kanban_task_id="t_old")
        # Backdate via internal helper / direct DB poke.
        ss._force_set_updated_at(time.time() - 24 * 3600)
        record = ss.read_status()
        assert ss.is_fresh(record, ttl_seconds=4 * 3600) is False

    def test_empty_ledger_is_not_fresh(self, _isolate_env):
        ss = _load_lib()
        assert ss.is_fresh(ss.read_status(), ttl_seconds=3600) is False


# ---------------------------------------------------------------------------
# Library: corrupt / missing files MUST NOT crash
# ---------------------------------------------------------------------------

class TestCorruptionSafety:
    def test_missing_ledger_returns_empty(self, _isolate_env):
        ss = _load_lib()
        # No file exists yet.
        assert (ss.get_ledger_path()).exists() is False
        record = ss.read_status()
        assert record == {} or record is None

    def test_corrupt_sqlite_returns_empty(self, _isolate_env):
        ss = _load_lib()
        ledger = ss.get_ledger_path()
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_bytes(b"this is not a sqlite file at all")
        # Must not raise.
        record = ss.read_status()
        assert record == {} or record is None

    def test_corrupt_does_not_crash_render(self, _isolate_env):
        ss = _load_lib()
        ledger = ss.get_ledger_path()
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_bytes(b"junk")
        block = ss.render_status_block(ttl_seconds=3600)
        # render_status_block is never allowed to raise; returns either
        # empty string or a clearly-marked "unavailable" hint.
        assert isinstance(block, str)
        assert "<system_status" not in block or "unavailable" in block.lower()


# ---------------------------------------------------------------------------
# Library: injection block formatting
# ---------------------------------------------------------------------------

class TestRenderBlock:
    def test_fresh_status_renders_block(self, _isolate_env):
        ss = _load_lib()
        ss.write_status(
            active_kanban_task_id="t_abc",
            active_workspace="/work/here",
            last_tool_invocation="kanban_show",
        )
        block = ss.render_status_block(ttl_seconds=3600)
        assert block.startswith("<system_status")
        assert block.rstrip().endswith("</system_status>")
        assert "t_abc" in block
        assert "/work/here" in block
        # Provenance: must include the updated_at age or ISO timestamp.
        assert "updated" in block.lower() or "age" in block.lower()

    def test_stale_status_does_NOT_render_live_block(self, _isolate_env):
        ss = _load_lib()
        ss.write_status(active_kanban_task_id="t_stale")
        ss._force_set_updated_at(time.time() - 24 * 3600)
        block = ss.render_status_block(ttl_seconds=4 * 3600)
        # Either empty, or explicitly marked stale — must NEVER look like
        # current truth.
        if block:
            assert "stale" in block.lower() or "expired" in block.lower()
            assert "current=true" not in block.lower()
        # Live-looking block forbidden.
        assert "<system_status live=" not in block or "stale" in block.lower()

    def test_block_under_token_budget(self, _isolate_env):
        ss = _load_lib()
        # Fill it with lots of drift signals.
        for i in range(50):
            ss.append_drift_signal(f"some longish drift signal number {i:03d}")
        ss.write_status(
            active_kanban_task_id="t_xyz",
            active_workspace="/some/path",
            last_tool_invocation="kanban_show",
        )
        block = ss.render_status_block(ttl_seconds=3600)
        # 300 tokens ~ 1200 chars worst-case for ascii-heavy English.
        assert len(block) <= ss.BLOCK_CHAR_BUDGET
        assert len(block) <= 1500  # Generous absolute cap

    def test_block_includes_freshness_provenance(self, _isolate_env):
        ss = _load_lib()
        ss.write_status(active_kanban_task_id="t_prov")
        block = ss.render_status_block(ttl_seconds=3600)
        # Must indicate how old the data is, not just present it as truth.
        # Either "updated " or "age=" or an explicit timestamp.
        assert any(
            marker in block.lower()
            for marker in ("updated", "age=", "age_seconds", "iso=")
        )


# ---------------------------------------------------------------------------
# Plugin __init__: pre_llm_call hook
# ---------------------------------------------------------------------------

class TestPreLlmCallHook:
    def test_hook_returns_dict_with_context_when_fresh(self, _isolate_env):
        ss = _load_lib()
        ss.write_status(active_kanban_task_id="t_hook")
        plugin = _load_plugin_init()
        result = plugin._on_pre_llm_call(
            session_id="sess1",
            user_message="hello",
            conversation_history=[],
            is_first_turn=True,
            model="claude",
            platform="cli",
            sender_id="zoe",
        )
        assert isinstance(result, dict)
        assert "context" in result
        assert "<system_status" in result["context"]
        assert "t_hook" in result["context"]

    def test_hook_returns_None_when_ledger_empty(self, _isolate_env):
        # Don't write anything to the ledger.
        plugin = _load_plugin_init()
        result = plugin._on_pre_llm_call(
            session_id="sess2",
            user_message="hello",
            conversation_history=[],
            is_first_turn=True,
            model="claude",
            platform="cli",
            sender_id="zoe",
        )
        # No content → don't inject anything (no laundering of empty state).
        assert result is None

    def test_hook_skips_stale_state(self, _isolate_env, monkeypatch):
        ss = _load_lib()
        ss.write_status(active_kanban_task_id="t_old")
        ss._force_set_updated_at(time.time() - 24 * 3600)
        # 4-hour TTL by default.
        monkeypatch.setenv("STATUS_SIDECAR_TTL_SECONDS", str(4 * 3600))
        plugin = _load_plugin_init()
        result = plugin._on_pre_llm_call(
            session_id="sess3",
            user_message="hi",
            conversation_history=[],
            is_first_turn=True,
            model="claude",
            platform="cli",
            sender_id="zoe",
        )
        # Either None (skipped) or context explicitly marked stale.
        if result is not None:
            assert "stale" in result["context"].lower() or "expired" in result["context"].lower()

    def test_hook_never_raises_on_corruption(self, _isolate_env):
        ss = _load_lib()
        ledger = ss.get_ledger_path()
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_bytes(b"corrupt!!")
        plugin = _load_plugin_init()
        # Must not raise; either returns None or a safe sentinel.
        result = plugin._on_pre_llm_call(
            session_id="sess4",
            user_message="hi",
            conversation_history=[],
            is_first_turn=True,
            model="claude",
            platform="cli",
            sender_id="zoe",
        )
        # Pass condition: no exception.
        assert result is None or isinstance(result, dict)

    def test_hook_does_not_mutate_system_prompt(self, _isolate_env):
        """The hook signature receives a user_message + conversation_history.
        It must NOT touch any system prompt. We assert by inspecting return
        shape: only "context" keys are honoured by run_agent.py.
        """
        ss = _load_lib()
        ss.write_status(active_kanban_task_id="t_x")
        plugin = _load_plugin_init()
        result = plugin._on_pre_llm_call(
            session_id="sess5",
            user_message="hi",
            conversation_history=[],
            is_first_turn=True,
            model="claude",
            platform="cli",
            sender_id="zoe",
        )
        assert isinstance(result, dict)
        # No system-prompt mutation keys.
        forbidden = {"system_prompt", "system", "system_message"}
        assert not (forbidden & set(result.keys()))


# ---------------------------------------------------------------------------
# Plugin __init__: status_update tool
# ---------------------------------------------------------------------------

class TestStatusUpdateTool:
    def test_tool_writes_drift_signal(self, _isolate_env):
        plugin = _load_plugin_init()
        result = plugin._status_update_handler(
            {"drift_signal": "user asked to simplify scope"},
        )
        # Tool handlers return JSON strings.
        payload = json.loads(result)
        assert payload.get("ok") is True
        ss = _load_lib()
        signals = ss.read_status().get("recent_drift_signals") or []
        assert any("simplify scope" in s for s in signals)

    def test_tool_writes_active_task(self, _isolate_env):
        plugin = _load_plugin_init()
        result = plugin._status_update_handler(
            {"active_kanban_task_id": "t_via_tool"},
        )
        payload = json.loads(result)
        assert payload.get("ok") is True
        ss = _load_lib()
        assert ss.read_status()["active_kanban_task_id"] == "t_via_tool"

    def test_tool_rejects_oversized_drift_signal(self, _isolate_env):
        """Drift signal text gets truncated, not stored unbounded."""
        plugin = _load_plugin_init()
        plugin._status_update_handler(
            {"drift_signal": "y" * 10000},
        )
        ss = _load_lib()
        signals = ss.read_status().get("recent_drift_signals") or []
        assert signals
        assert len(signals[-1]) <= ss.MAX_DRIFT_SIGNAL_CHARS


# ---------------------------------------------------------------------------
# Discovery: bundled plugin is found but not loaded unless enabled
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_plugin_discovered_as_standalone_opt_in(self, _isolate_env, monkeypatch):
        from hermes_cli import plugins as plugins_mod
        manager = plugins_mod.PluginManager()
        manager.discover_and_load()
        # Look for the plugin under its category key.
        loaded = manager._plugins.get("runtime/status_sidecar") or \
                 manager._plugins.get("runtime/status-sidecar") or \
                 manager._plugins.get("status-sidecar")
        assert loaded is not None, (
            f"plugin not discovered; available keys: "
            f"{sorted(manager._plugins.keys())}"
        )
        # Opt-in by default.
        assert loaded.enabled is False


# ---------------------------------------------------------------------------
# Smoke test: write status, invoke hook, show injected output
# ---------------------------------------------------------------------------

class TestSmokeInjection:
    def test_end_to_end_injection(self, _isolate_env, capsys):
        ss = _load_lib()
        ss.write_status(
            active_kanban_task_id="t_smoke",
            active_workspace="/tmp/smoke",
            last_tool_invocation="kanban_show",
        )
        ss.append_drift_signal("smoke test")
        plugin = _load_plugin_init()
        result = plugin._on_pre_llm_call(
            session_id="smoke",
            user_message="what's the status?",
            conversation_history=[],
            is_first_turn=True,
            model="claude",
            platform="cli",
            sender_id="zoe",
        )
        assert isinstance(result, dict)
        ctx = result["context"]
        # Print so a human running the test sees the actual injected block.
        print("\n----- INJECTED BLOCK -----")
        print(ctx)
        print("----- END -----")
        assert "t_smoke" in ctx
        assert "/tmp/smoke" in ctx
        assert "smoke test" in ctx
