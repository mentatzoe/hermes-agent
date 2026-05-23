"""Integration regression for Hindsight stale prefetch on topic switch."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from agent.memory_manager import MemoryManager
from plugins.memory.hindsight import HindsightMemoryProvider


def _provider_with_mock_client(tmp_path, monkeypatch):
    config_path = tmp_path / "hindsight" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        '{"mode":"cloud","apiKey":"test-key","api_url":"http://localhost:9999",'
        '"bank_id":"test-bank","budget":"mid","memory_mode":"hybrid"}'
    )
    monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)

    provider = HindsightMemoryProvider()
    provider.initialize(session_id="session-A", hermes_home=str(tmp_path), platform="cli")
    client = MagicMock()
    client.arecall = AsyncMock(
        return_value=SimpleNamespace(
            results=[SimpleNamespace(text="K3 stale Barcelona marker should not leak")]
        )
    )
    client.areflect = AsyncMock(return_value=SimpleNamespace(text=""))
    client.aretain_batch = AsyncMock()
    client.aclose = AsyncMock()
    provider._client = client
    return provider


def test_topic_switch_prefetch_pass_no_stale_prefetch(tmp_path, monkeypatch):
    provider = _provider_with_mock_client(tmp_path, monkeypatch)
    manager = MemoryManager()
    manager.add_provider(provider)

    manager.queue_prefetch_all("hotel recommendations in Barcelona", session_id="session-A")
    if provider._prefetch_thread:
        provider._prefetch_thread.join(timeout=5.0)

    context = manager.prefetch_all("what time did this session start", session_id="session-A")

    verdict = (
        "fail_stale_prefetch_reproduced"
        if "K3 stale Barcelona marker" in context
        else "pass_no_stale_prefetch"
    )
    assert verdict == "pass_no_stale_prefetch"
    assert context == ""
