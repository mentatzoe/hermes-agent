import asyncio
import json
from unittest.mock import AsyncMock

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, merge_pending_message_event
from gateway.run import GatewayRunner
from gateway.session import SessionSource
import hermes_state

from tests.gateway.restart_test_helpers import RestartTestAdapter


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


class ActiveWakeAdapter(WakeAdapter):
    def __init__(self, runner):
        super().__init__(runner)
        self._active_sessions = {}

    async def handle_message(self, event: MessageEvent):
        self.events.append(event)
        session_key = self.runner.session_store.get_or_create_session(event.source).session_key
        if session_key in self._active_sessions:
            self._pending_messages[session_key] = event
            return None
        return await self.runner._handle_message(event)


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


def test_internal_wake_queues_for_active_session_without_awaiting_running_task(monkeypatch, tmp_path):
    runner, calls = _runner(monkeypatch, tmp_path)
    adapter = ActiveWakeAdapter(runner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    entry = runner.session_store.get_or_create_session(_origin("lane-active"))

    async def scenario():
        never_done = asyncio.Event()
        running_task = asyncio.create_task(never_done.wait())
        adapter._active_sessions[entry.session_key] = asyncio.Event()
        adapter._session_tasks[entry.session_key] = running_task
        try:
            return await asyncio.wait_for(
                runner.wake_session(
                    session_key=entry.session_key,
                    payload="INTERNAL_WAKE_TEST_ACTIVE_QUEUE",
                    source_kind="send_message",
                    dedupe_key="active-queue-marker",
                ),
                timeout=0.2,
            )
        finally:
            running_task.cancel()
            await asyncio.gather(running_task, return_exceptions=True)

    result = asyncio.run(scenario())

    assert result["status"] == "queued"
    assert calls == []
    assert adapter.events[0].internal is True
    assert adapter._pending_messages[entry.session_key].text == "INTERNAL_WAKE_TEST_ACTIVE_QUEUE"

    rows = _receipt_rows(runner, entry.session_key)
    assert len(rows) == 1
    assert rows[0]["status"] == "queued"
    assert rows[0]["dispatched_at"] is not None
    assert rows[0]["responded_at"] is None


def test_internal_wake_queues_active_base_adapter_without_interrupting_running_agent(monkeypatch, tmp_path):
    runner, calls = _runner(monkeypatch, tmp_path)
    adapter = RestartTestAdapter()
    adapter.set_message_handler(AsyncMock(return_value=None))
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._is_user_authorized = lambda _source: True
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    entry = runner.session_store.get_or_create_session(_origin("lane-base-active"))

    interrupt_calls = []

    class RunningAgent:
        def interrupt(self, reason):
            interrupt_calls.append(reason)

        def get_activity_summary(self):
            return {}

    async def scenario():
        never_done = asyncio.Event()
        running_task = asyncio.create_task(never_done.wait())
        adapter._active_sessions[entry.session_key] = asyncio.Event()
        adapter._session_tasks[entry.session_key] = running_task
        runner._running_agents[entry.session_key] = RunningAgent()
        try:
            return await asyncio.wait_for(
                runner.wake_session(
                    session_key=entry.session_key,
                    payload="INTERNAL_WAKE_TEST_BASE_ACTIVE_QUEUE",
                    source_kind="send_message",
                    dedupe_key="base-active-queue-marker",
                ),
                timeout=0.2,
            )
        finally:
            running_task.cancel()
            await asyncio.gather(running_task, return_exceptions=True)

    result = asyncio.run(scenario())

    assert result["status"] == "queued"
    assert calls == []
    assert interrupt_calls == []
    pending = adapter._pending_messages[entry.session_key]
    assert pending.internal is True
    assert pending.text == "INTERNAL_WAKE_TEST_BASE_ACTIVE_QUEUE"
    assert adapter.sent == []

    rows = _receipt_rows(runner, entry.session_key)
    assert len(rows) == 1
    assert rows[0]["status"] == "queued"
    assert rows[0]["dispatched_at"] is not None
    assert rows[0]["responded_at"] is None


def test_internal_wake_queues_active_base_adapter_without_resolving_pending_clarify(monkeypatch, tmp_path):
    runner, calls = _runner(monkeypatch, tmp_path)
    adapter = RestartTestAdapter()
    adapter.set_message_handler(runner._handle_message)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._is_user_authorized = lambda source: True
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    entry = runner.session_store.get_or_create_session(_origin("lane-clarify-active"))

    interrupt_calls = []

    class RunningAgent:
        def interrupt(self, reason):
            interrupt_calls.append(reason)

        def get_activity_summary(self):
            return {}

    from tools import clarify_gateway as clarify_mod

    clarify_id = "internal-wake-clarify-active"
    clarify_entry = clarify_mod.register(
        clarify_id,
        entry.session_key,
        "Human answer?",
        None,
    )

    async def scenario():
        never_done = asyncio.Event()
        running_task = asyncio.create_task(never_done.wait())
        adapter._active_sessions[entry.session_key] = asyncio.Event()
        adapter._session_tasks[entry.session_key] = running_task
        runner._running_agents[entry.session_key] = RunningAgent()
        try:
            return await asyncio.wait_for(
                runner.wake_session(
                    session_key=entry.session_key,
                    payload="INTERNAL_WAKE_TEST_CLARIFY_ACTIVE_QUEUE",
                    source_kind="send_message",
                    dedupe_key="clarify-active-queue-marker",
                ),
                timeout=0.2,
            )
        finally:
            running_task.cancel()
            await asyncio.gather(running_task, return_exceptions=True)

    try:
        result = asyncio.run(scenario())

        assert result["status"] == "queued"
        assert calls == []
        assert interrupt_calls == []
        assert clarify_entry.response is None
        pending = adapter._pending_messages[entry.session_key]
        assert pending.internal is True
        assert pending.text == "INTERNAL_WAKE_TEST_CLARIFY_ACTIVE_QUEUE"
        assert adapter.sent == []

        rows = _receipt_rows(runner, entry.session_key)
        assert len(rows) == 1
        assert rows[0]["status"] == "queued"
        assert rows[0]["dispatched_at"] is not None
        assert rows[0]["responded_at"] is None
    finally:
        clarify_mod.clear_session(entry.session_key)


def test_internal_wake_queues_active_base_adapter_even_with_slash_command_payload(monkeypatch, tmp_path):
    runner, calls = _runner(monkeypatch, tmp_path)
    adapter = RestartTestAdapter()
    message_handler = AsyncMock(return_value=None)
    adapter.set_message_handler(message_handler)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._is_user_authorized = lambda source: True
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    entry = runner.session_store.get_or_create_session(_origin("lane-command-active"))

    interrupt_calls = []

    class RunningAgent:
        def interrupt(self, reason):
            interrupt_calls.append(reason)

        def get_activity_summary(self):
            return {}

    async def scenario():
        never_done = asyncio.Event()
        running_task = asyncio.create_task(never_done.wait())
        adapter._active_sessions[entry.session_key] = asyncio.Event()
        adapter._session_tasks[entry.session_key] = running_task
        runner._running_agents[entry.session_key] = RunningAgent()
        try:
            return await asyncio.wait_for(
                runner.wake_session(
                    session_key=entry.session_key,
                    payload="/status INTERNAL_WAKE_TEST_COMMAND_ACTIVE_QUEUE",
                    source_kind="send_message",
                    dedupe_key="command-active-queue-marker",
                ),
                timeout=0.2,
            )
        finally:
            running_task.cancel()
            await asyncio.gather(running_task, return_exceptions=True)

    result = asyncio.run(scenario())

    assert result["status"] == "queued"
    assert calls == []
    assert interrupt_calls == []
    message_handler.assert_not_awaited()
    pending = adapter._pending_messages[entry.session_key]
    assert pending.internal is True
    assert pending.text == "/status INTERNAL_WAKE_TEST_COMMAND_ACTIVE_QUEUE"
    assert adapter.sent == []

    rows = _receipt_rows(runner, entry.session_key)
    assert len(rows) == 1
    assert rows[0]["status"] == "queued"
    assert rows[0]["dispatched_at"] is not None
    assert rows[0]["responded_at"] is None


def _make_internal_wake_event(text, source, receipt_id):
    event = MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
        message_id=f"internal-wake:{receipt_id}",
    )
    setattr(event, "_hermes_internal_wake_receipt_id", receipt_id)
    return event


def test_merging_pending_internal_wakes_preserves_all_receipt_ids():
    source = _origin("lane-merged-receipt-ids")
    pending = {}
    first = _make_internal_wake_event("WAKE_ONE", source, 101)
    second = _make_internal_wake_event("WAKE_TWO", source, 102)

    merge_pending_message_event(pending, "session-key", first, merge_text=True)
    merge_pending_message_event(pending, "session-key", second, merge_text=True)

    merged = pending["session-key"]
    assert merged.text == "WAKE_ONE\nWAKE_TWO"
    assert getattr(merged, "_hermes_internal_wake_receipt_ids") == [101, 102]


def test_finalize_queued_internal_wake_receipts_updates_all_merged_receipts(monkeypatch, tmp_path):
    runner, _calls = _runner(monkeypatch, tmp_path)
    entry = runner.session_store.get_or_create_session(_origin("lane-finalize-queued"))

    def create_queued_receipt(payload, source_kind, dedupe_key):
        receipt, created = runner._session_db.create_session_wake_receipt(
            source_kind=source_kind,
            target_session_key=entry.session_key,
            target_session_id=entry.session_id,
            origin_snapshot=entry.origin.to_dict(),
            payload_hash="hash-" + dedupe_key,
            payload_preview=payload,
            payload_bytes=len(payload),
            dedupe_key=dedupe_key,
        )
        assert created is True
        runner._session_db.update_session_wake_receipt(
            int(receipt["id"]),
            status="queued",
            dispatched=True,
        )
        return int(receipt["id"])

    first_id = create_queued_receipt("WAKE_ONE", "send_message", "wake-one")
    second_id = create_queued_receipt("WAKE_TWO", "cron_delivery", "wake-two")

    runner._session_db.append_message(entry.session_id, "assistant", "original turn response")
    before_followup = max(m["id"] for m in runner._session_db.get_messages(entry.session_id))
    injected_id = runner._session_db.append_message(entry.session_id, "user", "WAKE_ONE\nWAKE_TWO")
    assistant_id = runner._session_db.append_message(entry.session_id, "assistant", "follow-up ack")

    event = _make_internal_wake_event("WAKE_ONE", entry.origin, first_id)
    merge_pending_message_event({entry.session_key: event}, entry.session_key, _make_internal_wake_event("WAKE_TWO", entry.origin, second_id), merge_text=True)

    runner._finalize_queued_internal_wake_receipts(
        event,
        session_id=entry.session_id,
        previous_max_id=before_followup,
        payload="WAKE_ONE\nWAKE_TWO",
    )

    rows = _receipt_rows(runner, entry.session_key)
    assert [row["id"] for row in rows] == [first_id, second_id]
    assert {row["status"] for row in rows} == {"agent_responded"}
    assert {row["injected_message_id"] for row in rows} == {injected_id}
    assert {row["assistant_message_id"] for row in rows} == {assistant_id}
    assert all(row["responded_at"] is not None for row in rows)
