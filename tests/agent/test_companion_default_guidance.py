"""Row 4: useful substance and follow-through survive every no-SOUL fallback."""
from types import SimpleNamespace

import pytest

from agent.system_prompt import _identity_parts
from agent.codex_responses_adapter import _preflight_codex_api_kwargs
from agent.transports.codex import ResponsesApiTransport


def assert_companion_guidance(text):
    assert "Understand the purpose" in text
    assert "evidence and uncertainty" in text
    assert "Carry agreed work through" in text
    assert "respecting explicit stops" in text
    assert "one-line answer" not in text


@pytest.mark.parametrize("skip", [False, True])
def test_missing_or_skipped_soul_uses_companion_guidance(tmp_path, monkeypatch, skip):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Prevent stock ensure_hermes_home from installing its bundled SOUL.
    (tmp_path / "SOUL.md").write_text("")
    agent = SimpleNamespace(load_soul_identity=False, skip_context_files=skip, _session_db=None)
    parts, loaded = _identity_parts(agent, None)
    assert not loaded
    assert_companion_guidance(parts[0])


def test_codex_empty_instructions_use_companion_guidance():
    transport = ResponsesApiTransport()
    payload = transport.build_kwargs("test-model", [{"role": "user", "content": "Hello"}])
    assert_companion_guidance(payload["instructions"])
    payload = _preflight_codex_api_kwargs({"model": "test-model", "instructions": "", "input": []})
    assert_companion_guidance(payload["instructions"])


def test_soul_still_replaces_the_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "SOUL.md").write_text("My own identity, not the generic one.")
    agent = SimpleNamespace(load_soul_identity=True, skip_context_files=True, _session_db=None)
    parts, loaded = _identity_parts(agent, None)
    assert loaded
    assert parts == ["My own identity, not the generic one."]
