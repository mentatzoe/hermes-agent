import pytest
import model_tools
from hermes_cli import plugins
from hermes_cli.plugins import PluginManager


@pytest.mark.parametrize("grant", [None, [], ["skill_view"], ["skill_view", "honcho_context"]])
def test_dispatch_passes_only_call_scoped_grants(tmp_path, monkeypatch, grant):
    plugin = tmp_path / "plugins" / "grant_probe"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: grant_probe\nversion: 0.1.0\n")
    (plugin / "__init__.py").write_text("""
SEEN = []
def register(ctx):
    def observe(*, next_call, enabled_tools=None, **kw):
        SEEN.append(enabled_tools)
        return next_call()
    ctx.register_middleware('tool_execution', observe)
""")
    (tmp_path / "config.yaml").write_text("plugins:\n  enabled: [grant_probe]\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager = PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(model_tools.registry, "dispatch", lambda *a, **kw: '{"success":true}')
    monkeypatch.setattr(model_tools, "_last_resolved_tool_names", ["unrelated", "honcho_context"])
    result = model_tools.handle_function_call("skill_view", {"name": "fixture"}, enabled_tools=grant)
    assert result == '{"success":true}'
    assert manager._plugins['grant_probe'].module.SEEN == [tuple(grant or ())]
