"""Aleph row 2: delegation keeps identity without personal memory inheritance."""
from unittest.mock import patch

import pytest

from run_agent import AIAgent
from tools.delegate_tool import _build_child_agent


@pytest.mark.parametrize("workspace", [False, True])
@pytest.mark.parametrize("depth", [0, 1])
def test_child_prompt_starts_with_own_soul(tmp_path, monkeypatch, workspace, depth):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    soul = "I am the test companion. Respect our agreed boundaries."
    (home / "SOUL.md").write_text(soul)
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("hermes_cli.plugins.discover_plugins"),
        patch("tools.delegate_tool._resolve_workspace_hint", return_value=str(tmp_path) if workspace else None),
        patch("tools.delegate_tool._resolve_child_credential_pool", return_value=None),
    ):
        parent = AIAgent(model="test-model", provider="openai", api_key="offline-test",
                         base_url="http://localhost:1/v1", skip_memory=True,
                         skip_context_files=True, quiet_mode=True)
        parent._delegate_depth = depth
        child = _build_child_agent(0, "Inspect synthetic data", None, [], None, 3, 1, parent)
        try:
            prompt = child._build_system_prompt()
            assert prompt.startswith(soul)
            assert prompt.count(soul) == 1
            assert child._memory_manager is None
            assert child.skip_context_files is True
        finally:
            child.close()
            parent.close()
