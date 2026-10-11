"""A mid-process lazy install of discord.py must reach every Discord adapter module.

A gateway that starts without discord.py imports the adapter with ``discord = None``;
``check_discord_requirements`` then lazy-installs discord.py and rebinds ``adapter.py``'s
module globals. The sibling mixin modules are imported along with the adapter, so a
module-level ``discord = None`` fallback in any of them is never rebound and native app
commands crash with ``'NoneType' object has no attribute 'DMChannel'``.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from tests.gateway.conftest import _ensure_discord_mock

_PKG = "plugins.platforms.discord"
_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "platforms" / "discord"


@pytest.fixture
def isolated_discord_plugin_modules():
    """Re-import the Discord plugin from scratch and restore the original modules afterwards."""
    saved = {name: mod for name, mod in sys.modules.items()
             if name == _PKG or name.startswith(_PKG + ".")}
    parent = sys.modules.get("plugins.platforms")
    parent_attr = getattr(parent, "discord", None) if parent is not None else None
    for name in saved:
        del sys.modules[name]
    try:
        yield
    finally:
        for name in [n for n in sys.modules if n == _PKG or n.startswith(_PKG + ".")]:
            del sys.modules[name]
        sys.modules.update(saved)
        if parent is not None and parent_attr is not None:
            parent.discord = parent_attr


def _lazy_bound_adapter_module(monkeypatch):
    """Import the adapter with discord.py missing, then simulate the lazy install."""
    original_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "discord" or name.startswith("discord."):
            raise ImportError("discord unavailable for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    adapter_mod = importlib.import_module(f"{_PKG}.adapter")
    # The sibling mixins were (re)loaded by the adapter import, with discord.py missing.
    assert f"{_PKG}.adapter_slash_auth" in sys.modules
    assert f"{_PKG}.adapter_thread_titles" in sys.modules
    assert adapter_mod.DISCORD_AVAILABLE is False
    assert adapter_mod.discord is None

    # "Install" discord.py: unblock imports and put the shared fake in place (forced even when
    # the real library is installed, so the fake DMChannel/Thread classes are the ones checked).
    monkeypatch.setattr(builtins, "__import__", original_import)
    for name in ("discord", "discord.ext", "discord.ext.commands"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    _ensure_discord_mock()

    # The real installer must not run. ``from pm import ensure_import`` resolves through the
    # pm facade, which caches the attribute on ``pm`` itself, so patch both homes.
    import pm
    import pm.extras

    def _no_install(extra):
        return None

    monkeypatch.setattr(pm.extras, "ensure_import", _no_install)
    monkeypatch.setattr(pm, "ensure_import", _no_install, raising=False)

    assert adapter_mod.check_discord_requirements() is True
    assert adapter_mod.DISCORD_AVAILABLE is True
    assert adapter_mod.discord is sys.modules["discord"]
    return adapter_mod, sys.modules["discord"]


def _make_adapter(adapter_mod):
    adapter = adapter_mod.DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=99999, name="HermesBot"), guilds=[])
    return adapter


@pytest.mark.usefixtures("isolated_discord_plugin_modules")
class TestLazyInstalledDiscordReachesSiblingMixins:
    def test_slash_authorization_sees_lazy_installed_dm_channel(self, monkeypatch):
        adapter_mod, discord_mod = _lazy_bound_adapter_module(monkeypatch)
        adapter = _make_adapter(adapter_mod)
        calls = []

        def record_is_allowed_user(user_id, author=None, *, guild=None, is_dm=False, channel_ids=None):
            calls.append({"user_id": user_id, "is_dm": is_dm, "channel_ids": channel_ids})
            return True

        adapter._is_allowed_user = record_is_allowed_user
        interaction = SimpleNamespace(
            channel=discord_mod.DMChannel(),
            channel_id=555,
            user=SimpleNamespace(id=42),
            guild=None,
        )

        assert adapter._evaluate_slash_authorization(interaction) == (True, None)
        # DM path: the user gate runs DM-scoped, with no channel keys.
        assert calls == [{"user_id": "42", "is_dm": True, "channel_ids": None}]

    def test_slash_authorization_sees_lazy_installed_thread(self, monkeypatch):
        adapter_mod, discord_mod = _lazy_bound_adapter_module(monkeypatch)
        adapter = _make_adapter(adapter_mod)
        monkeypatch.setattr(adapter, "_get_allowed_channels", lambda: {"777"})
        monkeypatch.setattr(adapter, "_get_ignored_channels", lambda: set())
        adapter._is_allowed_user = lambda *a, **k: True
        thread = discord_mod.Thread()
        thread.id = 888
        thread.name = "a thread"
        thread.parent = SimpleNamespace(id=777, name="general")
        thread.parent_id = 777
        interaction = SimpleNamespace(channel=thread, channel_id=888, user=SimpleNamespace(id=42), guild=None)

        # Allowed only via the thread's parent, which requires the isinstance(..., discord.Thread) branch.
        assert adapter._evaluate_slash_authorization(interaction) == (True, None)

    def test_guild_channel_labels_sees_lazy_installed_thread(self, monkeypatch):
        adapter_mod, discord_mod = _lazy_bound_adapter_module(monkeypatch)
        adapter = _make_adapter(adapter_mod)
        thread = discord_mod.Thread()
        thread.id = 888
        thread.name = "a thread"
        thread.topic = "thread topic"
        thread.parent = SimpleNamespace(id=777, name="general", guild=SimpleNamespace(name="Guild"))
        thread.guild = thread.parent.guild

        assert adapter._guild_channel_labels(thread) == ("Guild / #general / a thread", "thread topic")


# ── Static guard: only adapter.py may own a ``discord = None`` fallback ────────────────────────


def _binds_discord_none(node: ast.AST) -> bool:
    if isinstance(node, ast.Assign):
        targets, value = node.targets, node.value
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        targets, value = [node.target], node.value
    else:
        return False
    return (isinstance(value, ast.Constant) and value.value is None
            and any(isinstance(t, ast.Name) and t.id == "discord" for t in targets))


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(t, ast.Name) and t.id in {"ImportError", "ModuleNotFoundError"} for t in types)


def _module_level_statements(body):
    """Statements executed at import time: the module body plus nested if/try/with blocks."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for field in ("body", "orelse", "finalbody"):
            yield from _module_level_statements(getattr(stmt, field, []) or [])
        for handler in getattr(stmt, "handlers", []) or []:
            yield from _module_level_statements(handler.body)


def _discord_none_fallbacks(source: str) -> list[int]:
    tree = ast.parse(source)
    lines = {stmt.lineno for stmt in _module_level_statements(tree.body) if _binds_discord_none(stmt)}
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and _catches_import_error(node):
            lines.update(n.lineno for n in ast.walk(node) if _binds_discord_none(n))
    return sorted(lines)


def test_static_guard_detects_the_sibling_fallback_pattern():
    snippet = "try:\n    import discord\nexcept ImportError:\n    discord = None\n"
    assert _discord_none_fallbacks(snippet) == [4]
    assert _discord_none_fallbacks("def f():\n    try:\n        import discord\n"
                                   "    except ImportError:\n        discord = None\n") == [5]
    assert _discord_none_fallbacks("discord: object = None\n") == [1]
    assert _discord_none_fallbacks("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n"
                                   "    import discord\n") == []


def test_only_adapter_binds_a_module_level_discord_none_fallback():
    """adapter.py rebinds its fallback in check_discord_requirements; nothing else can."""
    modules = sorted(_PLUGIN_DIR.rglob("*.py"))
    assert _PLUGIN_DIR / "adapter.py" in modules
    offenders = {
        str(path.relative_to(_PLUGIN_DIR)): lines
        for path in modules
        if path.name != "adapter.py" or path.parent != _PLUGIN_DIR
        if (lines := _discord_none_fallbacks(path.read_text(encoding="utf-8")))
    }
    assert offenders == {}, (
        "These Discord plugin modules bind `discord = None` as an import fallback, which a "
        "mid-process lazy install never rebinds. Import discord inside the functions that use it "
        f"(and under TYPE_CHECKING for annotations): {offenders}"
    )
