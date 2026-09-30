"""The gateway hands the thread's parent channel to the agent it builds.

Honcho's ``excludeChats`` (Zoe, 2026-09-30) excludes a channel together with
every thread under it. A Discord thread has its own chat id, so the rule only
holds if the gateway passes the thread's parent channel through to the agent,
which hands it to the memory provider. Covers the real ``_run_agent`` path
used for every chat turn.
"""

import asyncio
import importlib
import sys
import types
from types import SimpleNamespace

from gateway.config import Platform
from gateway.session import SessionSource, build_session_key

ROOM = "1554601512842301531"
ROOM_THREAD = "1554661535073443841"
OTHER_CHANNEL = "1500000000000000001"
ZOE = "362335298983559169"


class _RecordingAgent:
    constructed = []

    def __init__(self, **kwargs):
        type(self).constructed.append(dict(kwargs))
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        return {"final_response": "done", "messages": [], "api_calls": 1}


def _runner():
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
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


def _run(monkeypatch, tmp_path, source):
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _RecordingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    _RecordingAgent.constructed = []

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    runner = _runner()
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
    assert result["final_response"] == "done"
    assert _RecordingAgent.constructed, "gateway never constructed an agent"
    return _RecordingAgent.constructed[-1]


def test_thread_turn_passes_parent_channel_to_agent(monkeypatch, tmp_path):
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id=ROOM_THREAD,
        chat_type="thread",
        thread_id=ROOM_THREAD,
        parent_chat_id=ROOM,
        user_id=ZOE,
    )
    kwargs = _run(monkeypatch, tmp_path, source)
    assert kwargs["chat_id"] == ROOM_THREAD
    assert kwargs["thread_id"] == ROOM_THREAD
    assert kwargs["parent_chat_id"] == ROOM


def test_channel_turn_has_no_parent(monkeypatch, tmp_path):
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id=OTHER_CHANNEL,
        chat_type="group",
        user_id=ZOE,
    )
    kwargs = _run(monkeypatch, tmp_path, source)
    assert kwargs["chat_id"] == OTHER_CHANNEL
    assert kwargs.get("parent_chat_id") is None
