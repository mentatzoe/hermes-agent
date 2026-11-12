import pytest
import model_tools
from hermes_cli import plugins
from hermes_cli.plugins import PluginManager


@pytest.mark.parametrize("grant", [None, [], ["skill_view"], ["skill_view", "honcho_context"]])
@pytest.mark.parametrize("route", ["direct", "agent"])
def test_dispatch_passes_only_call_scoped_grants(tmp_path, monkeypatch, grant, route):
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
    if route == "direct":
        result = model_tools.handle_function_call("skill_view", {"name": "fixture"}, enabled_tools=grant)
        assert result == '{"success":true}'
    else:
        from unittest.mock import patch
        from run_agent import AIAgent
        from agent.tool_executor import _run_agent_tool_execution_middleware
        with patch("model_tools.get_tool_definitions", return_value=[]), patch("agent.process_bootstrap.OpenAI"):
            agent = AIAgent(api_key="fixture", base_url="https://fixture.invalid/v1", model="fixture",
                            quiet_mode=True, skip_context_files=True, skip_memory=True)
        agent.valid_tool_names = set(grant or ())
        _run_agent_tool_execution_middleware(agent, function_name="skill_view", function_args={"name": "fixture"},
            effective_task_id="task", tool_call_id="call", execute=lambda args: '{"success":true}')
    assert len(manager._plugins['grant_probe'].module.SEEN) == 1
    assert set(manager._plugins['grant_probe'].module.SEEN[0] or ()) == set(grant or ())
    assert manager._plugins['grant_probe'].module.SEEN[0] is not None
