"""display.channel_overrides.<platform>.<id> must reach the gateway (Zoe, 2026-09-30).

Main joined #aleph-seat with per-channel quiet entries, but the display
resolver only read ``display.platforms.<platform>``, so the room stayed loud
unless the whole platform was silenced (which also silenced Home).  The room
thread, every other thread under the room (including ones created later) and
the room channel itself must resolve quiet from those entries, while channels
without an entry keep the platform values.
"""

import asyncio
import copy
import importlib
import sys
import time
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from gateway.config import Platform, PlatformConfig
from gateway.display_config import resolve_display_setting
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource, build_session_key

ROOM = "1554601512842301531"          # #aleph-seat
ROOM_THREAD = "1554661535073443841"   # the thread main joined
LATER_THREAD = "1554799999999999999"  # any thread created under the room later
HOME = "1496360045733085244"
OTHER = "1500000000000000001"
ZOE = "362335298983559169"

QUIET_ENTRY = {
    "tool_progress": "off",
    "interim_assistant_messages": False,
    "long_running_notifications": False,
    "busy_ack_detail": False,
    "busy_steer_ack_enabled": False,
    "show_reasoning": False,
    "thinking_progress": False,
    "streaming": False,
    "live_status": "off",
    "fail_open": False,  # stray non-display key present in the real entries
}

QUIET_EXPECTED = {
    "tool_progress": "off",
    "interim_assistant_messages": False,
    "streaming": False,
    "long_running_notifications": False,
    "busy_ack_detail": False,
    "busy_steer_ack_enabled": False,
    "show_reasoning": False,
    "thinking_progress": False,
    "live_status": "off",
}

LOUD_PLATFORM = {
    "streaming": True,
    "tool_progress": "all",
    "interim_assistant_messages": True,
    "long_running_notifications": True,
    "busy_ack_detail": True,
    "busy_steer_ack_enabled": True,
    "show_reasoning": True,
    "thinking_progress": True,
    "live_status": "full",
}


def _config(platform_discord):
    return {
        "display": {
            "tool_progress": "new",
            "interim_assistant_messages": True,
            "show_reasoning": True,
            "platforms": {"discord": dict(platform_discord)},
            "channel_overrides": {
                "discord": {
                    ROOM: dict(QUIET_ENTRY),
                    # YAML bare ``off`` arrives as False; the resolver normalises it.
                    ROOM_THREAD: {**QUIET_ENTRY, "tool_progress": False, "live_status": False},
                }
            },
        },
        "streaming": {"enabled": False},
    }


def _thread(thread_id, parent=ROOM):
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=thread_id,
        chat_type="thread",
        thread_id=thread_id,
        parent_chat_id=parent,
        user_id=ZOE,
    )


def _channel(chat_id):
    return SessionSource(
        platform=Platform.DISCORD, chat_id=chat_id, chat_type="group", user_id=ZOE
    )


def _resolved(config, source, platform_key="discord"):
    from gateway.run import _display_config_for_source

    view = _display_config_for_source(config, source)
    return {key: resolve_display_setting(view, platform_key, key) for key in QUIET_EXPECTED}


# ---------------------------------------------------------------------------
# Resolution through the per-source view
# ---------------------------------------------------------------------------

def test_room_thread_is_quiet_even_when_platform_is_loud():
    assert _resolved(_config(LOUD_PLATFORM), _thread(ROOM_THREAD)) == QUIET_EXPECTED


def test_any_other_thread_under_the_room_inherits_the_room_entry():
    assert _resolved(_config(LOUD_PLATFORM), _thread(LATER_THREAD)) == QUIET_EXPECTED


def test_room_channel_itself_is_quiet():
    assert _resolved(_config(LOUD_PLATFORM), _channel(ROOM)) == QUIET_EXPECTED


def test_channel_without_entry_gets_platform_values():
    from gateway.run import _display_config_for_source

    loud = _config(LOUD_PLATFORM)
    view = _display_config_for_source(loud, _channel(HOME))
    for key in QUIET_EXPECTED:
        assert resolve_display_setting(view, "discord", key) == resolve_display_setting(
            loud, "discord", key
        ), key
    home = _resolved(loud, _channel(HOME))
    assert home["tool_progress"] == "all"
    assert home["interim_assistant_messages"] is True
    assert home["streaming"] is True

    # Main's pre-04:56 platform block: Home falls back to the global display keys.
    restored = _resolved(_config({"streaming": True}), _channel(HOME))
    assert restored["tool_progress"] == "new"
    assert restored["interim_assistant_messages"] is True
    assert restored["streaming"] is True
    assert restored["show_reasoning"] is True


def test_per_channel_value_beats_per_platform_value():
    cfg = _config({"tool_progress": "all", "streaming": True})
    cfg["display"]["channel_overrides"]["discord"][OTHER] = {
        "tool_progress": "new",
        "streaming": "off",
    }
    got = _resolved(cfg, _channel(OTHER))
    assert got["tool_progress"] == "new"
    assert got["streaming"] is False


def test_thread_entry_refines_the_parent_entry_key_by_key():
    cfg = _config(LOUD_PLATFORM)
    cfg["display"]["channel_overrides"]["discord"][LATER_THREAD] = {"tool_progress": "new"}
    got = _resolved(cfg, _thread(LATER_THREAD))
    assert got["tool_progress"] == "new"  # the thread's own value wins
    assert got["interim_assistant_messages"] is False  # the rest still comes from the room
    assert got["streaming"] is False


def test_non_display_keys_and_nulls_are_ignored():
    from gateway.run import _display_config_for_source

    cfg = _config({"streaming": True})
    cfg["display"]["channel_overrides"]["discord"][OTHER] = {
        "streaming": None,
        "fail_open": False,
        "tool_progress": "off",
    }
    view = _display_config_for_source(cfg, _channel(OTHER))
    assert "fail_open" not in view["display"]["platforms"]["discord"]
    assert resolve_display_setting(view, "discord", "streaming") is True
    assert resolve_display_setting(view, "discord", "tool_progress") == "off"


def test_other_platforms_ignore_discord_entries():
    cfg = _config(LOUD_PLATFORM)
    cfg["display"]["platforms"]["telegram"] = {"tool_progress": "all"}
    src = SessionSource(platform=Platform.TELEGRAM, chat_id=ROOM, chat_type="group")
    assert _resolved(cfg, src, platform_key="telegram")["tool_progress"] == "all"


def test_caller_config_is_not_mutated():
    from gateway.run import _display_config_for_source

    cfg = _config(LOUD_PLATFORM)
    before = copy.deepcopy(cfg)
    _display_config_for_source(cfg, _thread(ROOM_THREAD))
    _display_config_for_source(cfg, _channel(HOME))
    assert cfg == before


# ---------------------------------------------------------------------------
# The real agent-run path: tool progress and interim commentary
# ---------------------------------------------------------------------------

class _CaptureAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.DISCORD):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(content)
        return SendResult(success=True, message_id=f"m-{len(self.sent)}")

    async def edit_message(self, chat_id, message_id, content, **_kwargs) -> SendResult:
        self.edits.append(content)
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class _ChattyAgent:
    COMMENTARY = "Checking the repo first."

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.interim_assistant_callback = kwargs.get("interim_assistant_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        if self.tool_progress_callback:
            self.tool_progress_callback("tool.started", "terminal", "pwd", {})
            time.sleep(0.35)
        if self.interim_assistant_callback:
            self.interim_assistant_callback(self.COMMENTARY, already_streamed=False)
            time.sleep(0.2)
        return {"final_response": "done", "messages": [], "api_calls": 1}


def _agent_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.session_store = SimpleNamespace(_entries={}, _save=lambda: None)
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


def _run_turn(monkeypatch, tmp_path, source):
    import yaml

    loud_without_streaming = {**LOUD_PLATFORM, "streaming": False}
    (tmp_path / "config.yaml").write_text(
        yaml.dump(_config(loud_without_streaming)), encoding="utf-8"
    )
    monkeypatch.delenv("HERMES_TOOL_PROGRESS_MODE", raising=False)
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _ChattyAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = _CaptureAdapter()
    runner = _agent_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    result = asyncio.run(
        runner._run_agent(
            message="hello",
            context_prompt="",
            history=[],
            source=source,
            session_id=f"sess-{source.chat_id}",
            session_key=build_session_key(source),
        )
    )
    return adapter, result


def test_agent_run_in_room_thread_posts_no_progress_or_commentary(monkeypatch, tmp_path):
    adapter, result = _run_turn(monkeypatch, tmp_path, _thread(LATER_THREAD))
    assert result["final_response"] == "done"
    posted = "\n".join(adapter.sent + adapter.edits)
    assert _ChattyAgent.COMMENTARY not in posted
    assert "pwd" not in posted  # tool progress renders the command preview


def test_agent_run_in_home_keeps_progress_and_commentary(monkeypatch, tmp_path):
    adapter, result = _run_turn(monkeypatch, tmp_path, _channel(HOME))
    assert result["final_response"] == "done"
    posted = "\n".join(adapter.sent + adapter.edits)
    assert _ChattyAgent.COMMENTARY in posted
    assert "pwd" in posted


# ---------------------------------------------------------------------------
# Busy-session steer acknowledgement
# ---------------------------------------------------------------------------

def _busy_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner._busy_text_mode = "interrupt"
    runner._busy_input_mode = "steer"
    runner.adapters = {}
    runner.config = MagicMock()
    runner.config.group_sessions_per_user = True
    runner.config.thread_sessions_per_user = False
    runner.session_store = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    return runner


def _steer_while_busy(monkeypatch, tmp_path, source):
    import gateway.run as gateway_run

    monkeypatch.delenv("HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", raising=False)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: _config(LOUD_PLATFORM))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner = _busy_runner()
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = Platform.DISCORD
    adapter._text_debounce = {}
    adapter._busy_text_debounce_seconds = 0.6
    runner.adapters[Platform.DISCORD] = adapter

    event = MessageEvent(
        text="also check the tests",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-1",
    )
    session_key = build_session_key(source)
    agent = MagicMock()
    agent.steer = MagicMock(return_value=True)
    runner._running_agents[session_key] = agent
    asyncio.run(runner._handle_active_session_busy_message(event, session_key))
    return agent, adapter


def test_steer_in_room_thread_sends_no_ack(monkeypatch, tmp_path):
    agent, adapter = _steer_while_busy(monkeypatch, tmp_path, _thread(ROOM_THREAD))
    agent.steer.assert_called_once()
    adapter._send_with_retry.assert_not_called()


def test_steer_in_home_still_acks(monkeypatch, tmp_path):
    agent, adapter = _steer_while_busy(monkeypatch, tmp_path, _channel(HOME))
    agent.steer.assert_called_once()
    adapter._send_with_retry.assert_called_once()
    content = adapter._send_with_retry.call_args.kwargs.get("content", "")
    assert "Steered" in content
