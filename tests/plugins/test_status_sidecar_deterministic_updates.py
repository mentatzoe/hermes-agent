"""Tests for the status-sidecar deterministic post_tool_call updates,
on_session_finalize hook, and tool exposure on Discord.

Background: the first card t_8a4caa0e shipped the ledger + pre_llm_call
injection but registered ``status_update`` under ``toolset="memory"``.
Zoe's Discord ``platform_toolsets`` does NOT include ``memory``, so the
tool is invisible on Aleph's actual DM surface. Card t_3f1d4fe9 fixes
that by:

 1. Moving the tool to a toolset that IS in Discord's allowlist.
 2. Adding deterministic ``post_tool_call`` updates so status stays
    fresh without the agent having to remember to call a tool.
 3. Adding ``on_session_finalize`` for a compact end-of-session drift
    signal.

Hard invariants tested:

 * No tool ``result`` body, no file contents, and no Kanban prose ever
   land in the ledger. Only ids, paths, tool names.
 * Hook never raises.
 * The status_update tool is visible to ``platform_toolsets.discord``.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "runtime" / "status_sidecar"


# ---------------------------------------------------------------------------
# Fixtures (mirror the existing test file so we stay isolated)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    yield hermes_home


def _load_lib():
    lib_path = PLUGIN_DIR / "status_sidecar.py"
    spec = importlib.util.spec_from_file_location(
        "status_sidecar_under_test_det", lib_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_plugin_init():
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
    # Fresh import each call — drop any cached copy.
    sys.modules.pop("hermes_plugins.status_sidecar", None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.status_sidecar"
    mod.__path__ = [str(plugin_dir)]
    sys.modules["hermes_plugins.status_sidecar"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. Pure derivation: derive_status_from_tool_call
#    (lives in status_sidecar.py for unit-testability; pure function)
# ---------------------------------------------------------------------------

class TestDeriveStatusFromToolCall:
    def test_unknown_tool_returns_empty(self, _isolate_env):
        ss = _load_lib()
        out = ss.derive_status_from_tool_call("random_thing", {"x": 1}, '{"ok":true}')
        assert out == {}

    def test_kanban_show_captures_task_id_workspace_and_project(self, _isolate_env):
        ss = _load_lib()
        # Result mirrors what kanban_show actually returns: a JSON string.
        result = json.dumps(
            {
                "task": {
                    "id": "t_abc123",
                    "title": "Some title with body content not stored",
                    "workspace_path": "/Users/x/.hermes/kanban/workspaces/t_abc123/hermes-agent",
                    "body": "PRIVATE BODY MUST NEVER LAND IN LEDGER",
                }
            }
        )
        out = ss.derive_status_from_tool_call("kanban_show", {"task_id": "t_abc123"}, result)
        assert out.get("active_kanban_task_id") == "t_abc123"
        assert "active_workspace" in out
        assert out["active_workspace"].endswith("hermes-agent")
        assert out.get("active_project_slug") == "hermes-agent"
        # Hard: must NEVER carry body / title / prose anywhere in the derived dict.
        flat = json.dumps(out)
        assert "PRIVATE BODY" not in flat
        assert "title" not in flat.lower()

    def test_kanban_complete_records_last_action(self, _isolate_env):
        ss = _load_lib()
        out = ss.derive_status_from_tool_call(
            "kanban_complete",
            {"summary": "did the thing — SECRET PROSE"},
            '{"ok": true}',
        )
        # Records the action verb only — no summary content.
        assert "last_tool_invocation" in out
        assert out["last_tool_invocation"] == "kanban_complete"
        flat = json.dumps(out)
        assert "SECRET PROSE" not in flat
        assert "did the thing" not in flat

    def test_kanban_create_returns_id_only(self, _isolate_env):
        ss = _load_lib()
        result = json.dumps({"task_id": "t_new99", "ok": True})
        out = ss.derive_status_from_tool_call(
            "kanban_create",
            {"title": "PRIVATE TITLE", "body": "PRIVATE BODY", "assignee": "eng"},
            result,
        )
        assert out.get("last_tool_invocation") == "kanban_create"
        flat = json.dumps(out)
        assert "PRIVATE TITLE" not in flat
        assert "PRIVATE BODY" not in flat

    def test_cronjob_create_captures_job_id(self, _isolate_env):
        ss = _load_lib()
        result = json.dumps(
            {
                "job_id": "job_42",
                "prompt": "SECRET PROMPT — must not leak",
                "schedule": "30m",
            }
        )
        out = ss.derive_status_from_tool_call(
            "cronjob",
            {"action": "create", "prompt": "SECRET PROMPT — must not leak"},
            result,
        )
        assert out.get("last_cron_job_id") == "job_42"
        assert out.get("last_tool_invocation") == "cronjob"
        flat = json.dumps(out)
        assert "SECRET PROMPT" not in flat

    def test_cronjob_other_action_records_action_name(self, _isolate_env):
        ss = _load_lib()
        out = ss.derive_status_from_tool_call(
            "cronjob",
            {"action": "remove", "job_id": "job_42"},
            '{"ok": true}',
        )
        # Mutating cron actions remain structural events, but render as events
        # rather than top-level "current focus".
        assert out.get("last_tool_invocation") == "cronjob"
        assert out.get("last_cron_job_id") == "job_42"
        assert out.get("_drift") == "cronjob remove job_42"

    def test_cronjob_list_is_low_signal_and_ignored(self, _isolate_env):
        ss = _load_lib()
        out = ss.derive_status_from_tool_call(
            "cronjob",
            {"action": "list"},
            '{"jobs": [{"job_id": "SECRET_JOB_BODY_SHOULD_NOT_MATTER"}]}',
        )
        assert out == {}
    def test_write_file_captures_path_project_only_no_content(self, _isolate_env):
        ss = _load_lib()
        out = ss.derive_status_from_tool_call(
            "write_file",
            {
                "path": "/Users/zoe/github/aleph-vault/projects/aleph/hermes-harness/note.md",
                "content": "PRIVATE NOTE CONTENT that must never be stored",
            },
            '{"ok": true}',
        )
        # Path lands in last_tool_invocation as a brief basename breadcrumb.
        # Project slug is inferred from path. The full content MUST NOT appear.
        flat = json.dumps(out)
        assert "PRIVATE NOTE CONTENT" not in flat
        assert out.get("last_tool_invocation", "").startswith("write_file")
        assert "note.md" in out.get("last_tool_invocation", "")
        assert out.get("active_project_slug") == "github/aleph-vault/projects/aleph/hermes-harness"

    def test_patch_captures_path_only_no_diff(self, _isolate_env):
        ss = _load_lib()
        out = ss.derive_status_from_tool_call(
            "patch",
            {
                "path": "/tmp/file.py",
                "old_string": "SECRET old code",
                "new_string": "SECRET new code",
            },
            '{"ok": true, "diff": "--- a/file.py\\n+++ b/file.py\\nSECRET diff content"}',
        )
        flat = json.dumps(out)
        assert "SECRET" not in flat
        assert out.get("last_tool_invocation", "").startswith("patch")

    def test_status_update_itself_does_not_recurse(self, _isolate_env):
        """status_update should NOT trigger another post_tool_call write —
        otherwise every drift signal write turns into two ledger touches.
        """
        ss = _load_lib()
        out = ss.derive_status_from_tool_call(
            "status_update",
            {"drift_signal": "user asked to defer"},
            '{"ok": true}',
        )
        assert out == {}

    def test_terminal_tool_ignored(self, _isolate_env):
        """terminal is too noisy to track per-call. Verify it's deliberately
        ignored (no last_tool_invocation churn from every shell command)."""
        ss = _load_lib()
        out = ss.derive_status_from_tool_call(
            "terminal",
            {"command": "ls"},
            '{"output": "file1\\nfile2", "exit_code": 0}',
        )
        assert out == {}


# ---------------------------------------------------------------------------
# 2. Plugin hook: _on_post_tool_call applies derived updates
# ---------------------------------------------------------------------------

class TestPostToolCallHook:
    def test_kanban_show_updates_ledger_and_active_session(self, _isolate_env):
        plugin = _load_plugin_init()
        ss = _load_lib()
        result = json.dumps(
            {"task": {"id": "t_hookpost", "workspace_path": "/Users/x/.hermes/kanban/workspaces/t_hookpost/hermes-agent"}}
        )
        plugin._on_post_tool_call(
            tool_name="kanban_show",
            args={"task_id": "t_hookpost"},
            result=result,
            task_id="",
            session_id="s1",
            tool_call_id="c1",
            duration_ms=12,
        )
        rec = ss.read_status()
        assert rec.get("active_kanban_task_id") == "t_hookpost"
        assert rec.get("active_workspace").endswith("hermes-agent")
        assert rec.get("active_project_slug") == "hermes-agent"

        sessions = ss.read_active_sessions(ttl_seconds=3600)
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "s1"
        assert sessions[0]["active_kanban_task_id"] == "t_hookpost"
        assert sessions[0]["active_project_slug"] == "hermes-agent"

    def test_no_result_body_in_ledger(self, _isolate_env):
        """Hard guarantee: the SQLite ledger file must never contain any
        tool result body text or args content. Read it raw and grep."""
        plugin = _load_plugin_init()
        ss = _load_lib()
        result_str = (
            '{"task": {"id": "t_nokleak", '
            '"body": "TOPSECRET_PROSE_MARKER", '
            '"title": "TOPSECRET_TITLE_MARKER", '
            '"workspace_path": "/work/t_nokleak"}}'
        )
        plugin._on_post_tool_call(
            tool_name="kanban_show",
            args={"task_id": "t_nokleak"},
            result=result_str,
            task_id="",
            session_id="s2",
            tool_call_id="c2",
            duration_ms=8,
        )
        ledger = ss.get_ledger_path()
        assert ledger.exists()
        raw = ledger.read_bytes()
        assert b"TOPSECRET_PROSE_MARKER" not in raw
        assert b"TOPSECRET_TITLE_MARKER" not in raw

    def test_cronjob_create_updates_last_cron_job_id(self, _isolate_env):
        plugin = _load_plugin_init()
        ss = _load_lib()
        plugin._on_post_tool_call(
            tool_name="cronjob",
            args={"action": "create", "prompt": "SECRET_PROMPT_X"},
            result=json.dumps({"job_id": "job_xyz", "ok": True}),
            task_id="",
            session_id="s3",
            tool_call_id="c3",
            duration_ms=15,
        )
        rec = ss.read_status()
        assert rec.get("last_cron_job_id") == "job_xyz"
        raw = ss.get_ledger_path().read_bytes()
        assert b"SECRET_PROMPT_X" not in raw

    def test_write_file_records_tool_not_content(self, _isolate_env):
        plugin = _load_plugin_init()
        ss = _load_lib()
        plugin._on_post_tool_call(
            tool_name="write_file",
            args={
                "path": "/tmp/x.md",
                "content": "SECRET_FILE_BODY_should_never_appear",
            },
            result='{"ok": true}',
            task_id="",
            session_id="s4",
            tool_call_id="c4",
            duration_ms=4,
        )
        raw = ss.get_ledger_path().read_bytes()
        assert b"SECRET_FILE_BODY" not in raw
        rec = ss.read_status()
        assert (rec.get("last_tool_invocation") or "").startswith("write_file")

    def test_hook_never_raises_on_garbage_result(self, _isolate_env):
        plugin = _load_plugin_init()
        # Result that isn't a JSON string at all.
        plugin._on_post_tool_call(
            tool_name="kanban_show",
            args={"task_id": "t_garbage"},
            result="not even close to json {{",
            task_id="",
            session_id="s5",
            tool_call_id="c5",
            duration_ms=1,
        )

    def test_hook_never_raises_on_non_dict_args(self, _isolate_env):
        plugin = _load_plugin_init()
        plugin._on_post_tool_call(
            tool_name="kanban_show",
            args="not a dict",
            result='{"ok": true}',
            task_id="",
            session_id="s6",
            tool_call_id="c6",
            duration_ms=1,
        )

    def test_hook_ignores_unknown_tools(self, _isolate_env):
        plugin = _load_plugin_init()
        ss = _load_lib()
        plugin._on_post_tool_call(
            tool_name="some_random_tool",
            args={"x": 1},
            result='{"ok": true}',
            task_id="",
            session_id="s7",
            tool_call_id="c7",
            duration_ms=1,
        )
        # Unknown tool → no ledger touch at all (no row created from noise).
        rec = ss.read_status()
        # Either empty (no write happened) or the row exists from a prior
        # call. In this fresh-fixture test, must be empty.
        assert rec == {} or rec.get("status_updated_at") is None


# ---------------------------------------------------------------------------
# 3. on_session_finalize hook: compact drift signal
# ---------------------------------------------------------------------------

class TestOnSessionFinalize:
    def test_finalize_appends_drift_signal(self, _isolate_env):
        plugin = _load_plugin_init()
        ss = _load_lib()
        # Seed: a post_tool_call happened earlier this session.
        plugin._on_post_tool_call(
            tool_name="kanban_show",
            args={"task_id": "t_final"},
            result=json.dumps({"task": {"id": "t_final", "workspace_path": "/w"}}),
            task_id="",
            session_id="sess_final",
            tool_call_id="c1",
            duration_ms=1,
        )
        plugin._on_session_finalize(session_id="sess_final")
        signals = ss.read_status().get("recent_drift_signals") or []
        assert signals, "expected at least one drift signal after finalize"
        # The signal must NOT contain user prose. It's a structured breadcrumb.
        # We assert positively: contains "session" marker.
        assert any("session" in s.lower() for s in signals)

    def test_finalize_safe_on_empty_ledger(self, _isolate_env):
        plugin = _load_plugin_init()
        # No prior activity. Must not raise.
        plugin._on_session_finalize(session_id="empty_sess")

    def test_finalize_is_bounded(self, _isolate_env):
        """Calling finalize many times must not blow the drift signal ring."""
        plugin = _load_plugin_init()
        ss = _load_lib()
        for i in range(30):
            plugin._on_session_finalize(session_id=f"s{i}")
        signals = ss.read_status().get("recent_drift_signals") or []
        assert len(signals) <= ss.MAX_DRIFT_SIGNALS


# ---------------------------------------------------------------------------
# 4. Tool exposure: status_update must be visible on Discord
# ---------------------------------------------------------------------------

class TestToolExposure:
    def _resolve_toolset(self) -> str:
        """Resolve the toolset the plugin actually uses for status_update.

        Loads the plugin module and inspects ``STATUS_UPDATE_TOOLSET`` if
        exposed, falling back to a regex over the ``register_tool`` call.
        """
        plugin = _load_plugin_init()
        ts = getattr(plugin, "STATUS_UPDATE_TOOLSET", None)
        if isinstance(ts, str) and ts:
            return ts
        # Fallback: regex over the source.
        init_src = (PLUGIN_DIR / "__init__.py").read_text()
        import re
        m = re.search(
            r"register_tool\([^)]*toolset\s*=\s*[\"']([^\"']+)[\"']",
            init_src,
            re.DOTALL,
        )
        assert m, "could not resolve status_update toolset"
        return m.group(1)

    def test_status_update_toolset_is_in_discord_platform_toolsets(self):
        """The user's Discord platform_toolsets config does NOT include
        'memory'. The status_update tool must therefore live under a
        toolset that IS allowlisted on Discord (e.g. 'todo'), or be
        explicitly documented as relying on deterministic hooks only.
        """
        # Discord's allowlist as it actually exists in Zoe's config —
        # codified here so the test fails the moment we put status_update
        # under an invisible toolset.
        discord_toolsets = {
            "browser", "clarify", "code_execution", "cronjob",
            "delegation", "file", "homeassistant", "image_gen",
            "messaging", "moa", "rl", "session_search", "skills",
            "terminal", "todo", "tts", "vision", "web",
        }
        ts = self._resolve_toolset()
        assert ts in discord_toolsets, (
            f"status_update is under toolset={ts!r}, which is NOT in "
            f"Discord's platform_toolsets allowlist. Choose one of: "
            f"{sorted(discord_toolsets)}."
        )

    def test_status_update_not_in_memory_toolset(self):
        """Regression guard for the v1 issue: status_update was under
        'memory', which Discord drops."""
        ts = self._resolve_toolset()
        assert ts != "memory"


# ---------------------------------------------------------------------------
# 5. Manifest declares the new hooks
# ---------------------------------------------------------------------------

class TestManifestHooks:
    def test_manifest_declares_post_tool_call(self):
        data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
        assert "post_tool_call" in data["hooks"]

    def test_manifest_declares_on_session_finalize(self):
        data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
        assert "on_session_finalize" in data["hooks"]
