"""A webhook agent-mode turn has no human reader: a bare silence marker must post nothing.

Regression (15 Oct 2026): a repair route tells its agent to answer ``[SILENT]`` when there is
nothing for the operator. The webhook adapter built its
``MessageEvent`` without ``reply_expected``, so the human-turn rule (5ea8fb2b78, v0.21.6)
replaced the marker with the "unexpected silence" warning and, with ``show_reasoning`` on,
prepended the reasoning block, and the route's ``deliver: discord`` posted both into the target chat.

The webhook event now carries ``reply_expected=False`` (not ``internal``: an internal event
skips authorization and the emergency stop). It is persisted on the user row, so crash
recovery decides the same way the live turn did.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.platforms.webhook import WebhookAdapter
from gateway.response_filters import is_intentional_silence_response
from gateway.session import SessionEntry, SessionSource

_WARNING = ("⚠️ The model returned only a silence marker for a message that needed a reply. "
            "Try again or rephrase.")
_REASONING = "The other repair agent already owns this fault, so there is nothing to report."


def _webhook_adapter() -> WebhookAdapter:
    routes = {"repair": {"prompt": "Repair: {payload.what}", "deliver": "discord",
                         "deliver_extra": {"chat_id": "home-dm"}}}
    return WebhookAdapter(PlatformConfig(enabled=True, extra={
        "host": "127.0.0.1", "port": 0, "routes": routes, "rate_limit": 30,
        "max_body_bytes": 1_048_576, "secret": "INSECURE_NO_AUTH"}))


async def _webhook_event() -> MessageEvent:
    """The MessageEvent the adapter really builds for one agent-mode POST."""
    adapter = _webhook_adapter()
    captured = []

    async def _capture(event):
        captured.append(event)

    adapter.handle_message = _capture
    app = web.Application(client_max_size=adapter._max_body_bytes)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/webhooks/repair", json={"what": "home sync"},
                              headers={"X-GitHub-Delivery": "silence-1"})
        assert resp.status == 202
    await asyncio.sleep(0.05)
    assert len(captured) == 1
    return captured[0]


def _discord_event() -> MessageEvent:
    source = SessionSource(platform=Platform.DISCORD, chat_id="home-dm", chat_type="dm", user_id="zoe")
    return MessageEvent(text="are you there?", source=source, message_id="msg-1")


def _runner(monkeypatch, tmp_path, source: SessionSource):
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner._show_reasoning = True  # display.show_reasoning is on for Discord in the live config
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:test", session_id="sess-silent", created_at=datetime.now(),
        updated_at=datetime.now(), platform=source.platform, chat_type=source.chat_type)
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._run_agent = AsyncMock(return_value={
        "final_response": "[SILENT]", "last_reasoning": _REASONING,
        "messages": [{"role": "user", "content": "prompt"}, {"role": "assistant", "content": "[SILENT]"}],
        "tools": [], "history_offset": 0, "last_prompt_tokens": 0, "api_calls": 1, "failed": False,
    })
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *_a, **_kw: 100_000)
    return runner


@pytest.mark.asyncio
async def test_webhook_agent_event_is_not_awaiting_a_reply_and_stays_non_internal():
    event = await _webhook_event()
    assert event.reply_expected is False
    # Authorization and the emergency stop still apply to it.
    assert not event.internal


@pytest.mark.asyncio
async def test_webhook_turn_silence_posts_nothing_and_persists_reply_expected_false(monkeypatch, tmp_path):
    event = await _webhook_event()
    runner = _runner(monkeypatch, tmp_path, event.source)

    response = await runner._handle_message_with_agent(event, event.source, "agent:main:test", 1)

    assert response == ""  # no warning, no reasoning block
    persisted = runner._run_agent.await_args.kwargs["persist_user_display_metadata"]
    assert persisted["reply_expected"] is False


@pytest.mark.asyncio
async def test_human_discord_turn_silence_still_gets_the_visible_fallback(monkeypatch, tmp_path):
    event = _discord_event()
    runner = _runner(monkeypatch, tmp_path, event.source)

    response = await runner._handle_message_with_agent(event, event.source, "agent:main:test", 1)

    assert response and not is_intentional_silence_response(response)
    assert _WARNING in response
    assert "reply_expected" not in runner._run_agent.await_args.kwargs["persist_user_display_metadata"]


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["webhook", "discord"])
async def test_crash_recovery_agrees_with_the_live_turn(monkeypatch, tmp_path, origin):
    """The user row the live turn persisted decides what a crash-left [SILENT] reply owes."""
    event = await _webhook_event() if origin == "webhook" else _discord_event()
    runner = _runner(monkeypatch, tmp_path, event.source)
    live = await runner._handle_message_with_agent(event, event.source, "agent:main:test", 1)
    kwargs = runner._run_agent.await_args.kwargs
    user_row = {"role": "user", "content": "prompt", "timestamp": 100.0,
                "display_metadata": kwargs["persist_user_display_metadata"]}
    if kwargs.get("persist_user_display_kind"):
        user_row["display_kind"] = kwargs["persist_user_display_kind"]
    history = [user_row, {"role": "assistant", "content": "[SILENT]", "timestamp": 101.0}]

    owed = runner._crash_left_reply(history, 50.0, event.source)

    if origin == "webhook":
        assert live == "" and owed == ""
    else:
        assert _WARNING in live and owed == _WARNING
