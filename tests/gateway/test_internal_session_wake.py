import asyncio
import json

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
import hermes_state


class WakeAdapter:
    def __init__(self, runner):
        self.runner = runner
        self.events = []
        self._session_tasks = {}
        self._pending_messages = {}

    async def handle_message(self, event: MessageEvent):
        self.events.append(event)
        return await self.runner._handle_message(event)

    async def send(self, chat_id, text, **kwargs):
        return None


def _runner(monkeypatch, tmp_path):
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    runner = GatewayRunner(
        GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="test-token")}
        )
    )
    runner.adapters = {Platform.TELEGRAM: WakeAdapter(runner)}
    calls = []

    async def fake_run_agent(**kwargs):
        calls.append(kwargs)
        history = kwargs.get("history") or []
        session_id = kwargs["session_id"]
        message = kwargs["message"]
        runner._session_db.append_message(session_id, "user", message)
        runner._session_db.append_message(
            session_id,
            "assistant",
            f"acknowledged wake: {message}",
        )
        return {
            "final_response": f"acknowledged wake: {message}",
            "messages": [
                *history,
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"acknowledged wake: {message}"},
            ],
            "history_offset": len(history),
            "api_calls": 1,
            "last_prompt_tokens": 12,
            "tools": [],
        }

    monkeypatch.setattr(runner, "_run_agent", fake_run_agent)
    runner._is_user_authorized = lambda _source: False
    return runner, calls


def _origin(chat_id="lane-42"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="dm",
        user_id="zoe",
        user_name="Zoe",
        is_bot=False,
    )


def _receipt_rows(runner, session_key):
    with runner._session_db._lock:
        return [
            dict(row)
            for row in runner._session_db._conn.execute(
                "SELECT * FROM session_wake_receipts WHERE target_session_key = ? ORDER BY id",
                (session_key,),
            ).fetchall()
        ]


def test_internal_wake_reuses_existing_session_origin_and_records_agent_response(monkeypatch, tmp_path):
    runner, calls = _runner(monkeypatch, tmp_path)
    entry = runner.session_store.get_or_create_session(_origin("lane-origin"))

    result = asyncio.run(
        runner.wake_session(
            session_key=entry.session_key,
            payload="INTERNAL_WAKE_TEST_ORIGIN",
            source_kind="kanban",
            dedupe_key="origin-once",
        )
    )

    assert result["status"] == "agent_responded"
    assert len(calls) == 1
    assert calls[0]["session_id"] == entry.session_id
    assert calls[0]["source"].user_id == "zoe"
    assert calls[0]["source"].user_name == "Zoe"
    assert calls[0]["source"].is_bot is False
    assert runner.adapters[Platform.TELEGRAM].events[0].internal is True
    assert runner.adapters[Platform.TELEGRAM].events[0].source == entry.origin

    messages = runner._session_db.get_messages(entry.session_id)
    conversational = [m for m in messages if m["role"] in {"user", "assistant"}]
    assert [m["role"] for m in conversational[-2:]] == ["user", "assistant"]
    assert conversational[-2]["content"] == "INTERNAL_WAKE_TEST_ORIGIN"
    assert conversational[-1]["content"] == "acknowledged wake: INTERNAL_WAKE_TEST_ORIGIN"

    rows = _receipt_rows(runner, entry.session_key)
    assert len(rows) == 1
    receipt = rows[0]
    assert receipt["status"] == "agent_responded"
    assert receipt["source_kind"] == "kanban"
    assert receipt["target_session_key"] == entry.session_key
    assert receipt["target_session_id"] == entry.session_id
    assert receipt["injected_message_id"] == conversational[-2]["id"]
    assert receipt["assistant_message_id"] == conversational[-1]["id"]
    assert json.loads(receipt["origin_snapshot"])["user_id"] == "zoe"
    assert receipt["payload_hash"]
    assert receipt["payload_preview"] == "INTERNAL_WAKE_TEST_ORIGIN"


def test_internal_wake_dedupe_key_prevents_duplicate_injection(monkeypatch, tmp_path):
    runner, calls = _runner(monkeypatch, tmp_path)
    entry = runner.session_store.get_or_create_session(_origin("lane-dedupe"))

    first = asyncio.run(
        runner.wake_session(
            session_key=entry.session_key,
            payload="INTERNAL_WAKE_TEST_DEDUPE",
            source_kind="cron",
            dedupe_key="same-marker",
        )
    )
    second = asyncio.run(
        runner.wake_session(
            session_key=entry.session_key,
            payload="INTERNAL_WAKE_TEST_DEDUPE",
            source_kind="cron",
            dedupe_key="same-marker",
        )
    )

    assert first["status"] == "agent_responded"
    assert second["status"] == "deduped"
    assert len(calls) == 1
    assert len(runner.adapters[Platform.TELEGRAM].events) == 1
    assert [m["content"] for m in runner._session_db.get_messages(entry.session_id)].count(
        "INTERNAL_WAKE_TEST_DEDUPE"
    ) == 1

    rows = _receipt_rows(runner, entry.session_key)
    assert len(rows) == 1
    assert rows[0]["status"] == "agent_responded"
    assert rows[0]["dedupe_key"] == "same-marker"


def test_internal_wake_receipt_records_failure_when_adapter_unavailable(monkeypatch, tmp_path):
    runner, calls = _runner(monkeypatch, tmp_path)
    entry = runner.session_store.get_or_create_session(_origin("lane-failure"))
    adapter = runner.adapters[Platform.TELEGRAM]
    runner.adapters = {}

    result = asyncio.run(
        runner.wake_session(
            session_key=entry.session_key,
            payload="INTERNAL_WAKE_TEST_FAILURE",
            source_kind="cron",
            dedupe_key="failure-marker",
        )
    )

    assert result["status"] == "failure"
    assert "adapter unavailable" in result["error"]
    assert calls == []
    rows = _receipt_rows(runner, entry.session_key)
    assert len(rows) == 1
    assert rows[0]["status"] == "failure"
    assert rows[0]["error"] == result["error"]
    assert rows[0]["dispatched_at"] is None
    assert rows[0]["responded_at"] is None

    runner.adapters = {Platform.TELEGRAM: adapter}
    retry = asyncio.run(
        runner.wake_session(
            session_key=entry.session_key,
            payload="INTERNAL_WAKE_TEST_FAILURE",
            source_kind="cron",
            dedupe_key="failure-marker",
        )
    )

    assert retry["status"] == "agent_responded"
    assert len(calls) == 1
    rows = _receipt_rows(runner, entry.session_key)
    assert len(rows) == 1
    assert rows[0]["status"] == "agent_responded"
    assert rows[0]["assistant_message_id"] is not None


def test_internal_wake_can_target_by_session_id(monkeypatch, tmp_path):
    runner, calls = _runner(monkeypatch, tmp_path)
    entry = runner.session_store.get_or_create_session(_origin("lane-session-id"))

    result = asyncio.run(
        runner.wake_session(
            session_id=entry.session_id,
            payload="INTERNAL_WAKE_TEST_SESSION_ID",
            source_kind="kanban",
        )
    )

    assert result["status"] == "agent_responded"
    assert calls[0]["session_id"] == entry.session_id
    assert calls[0]["source"] == entry.origin
