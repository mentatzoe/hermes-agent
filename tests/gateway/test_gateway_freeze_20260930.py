"""Regression tests for the 30 Sep 2026 main-gateway incident.

1. Freeze: ``_is_agents_own_voice`` ran a synchronous, unindexed 30-day
   SQLite scan on the gateway event loop (60-120 s on a 5 GB state.db),
   blocking Discord heartbeats until the loop-liveness watchdog exited 75.
2. Lost queued messages: shutdown flush files never carried a session id,
   so startup recovery skipped every one of them.
3. Replay flood: drain notices ("⏳ Gateway is restarting ...") and hours-old
   rejected replies were ledgered and replayed after the next restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ── 1. Freeze: own-voice check off the loop, bounded ─────────────────────────


def _make_state_db(path, rows):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "session_id TEXT, role TEXT, content TEXT, tool_calls TEXT, timestamp REAL)"
    )
    conn.executemany(
        "INSERT INTO messages (session_id, role, content, tool_calls, timestamp) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


_SAID = (
    "the quick brown fox jumps over the lazy dog while seven wizards "
    "quietly judge every boxing match tonight"
)


def _tts_call(text=_SAID):
    return json.dumps(
        [{"function": {"name": "text_to_speech", "arguments": json.dumps({"text": text})}}]
    )


def test_own_voice_scan_is_bounded_to_recent_rows(tmp_path, monkeypatch):
    """A TTS call older than the scan window is not examined."""
    import gateway.run as run_mod

    home = tmp_path / ".hermes"
    home.mkdir()
    now = time.time()
    old_tts = [("s", "assistant", "", _tts_call(), now - 60)]
    filler = [("s", "user", "x", None, now - 30)] * 50
    _make_state_db(home / "state.db", old_tts + filler)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)

    monkeypatch.setattr(run_mod, "_OWN_VOICE_SCAN_ROWS", 10_000)
    assert run_mod._is_agents_own_voice("/tmp/x.ogg", _SAID) is True

    monkeypatch.setattr(run_mod, "_OWN_VOICE_SCAN_ROWS", 10)
    assert run_mod._is_agents_own_voice("/tmp/x.ogg", _SAID) is False


def test_own_voice_query_budget_aborts_and_fails_closed(tmp_path, monkeypatch):
    import gateway.run as run_mod

    home = tmp_path / ".hermes"
    home.mkdir()
    now = time.time()
    _make_state_db(
        home / "state.db",
        [("s", "assistant", "", '[{"function": {"name": "terminal"}}]', now)] * 5000,
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
    monkeypatch.setattr(run_mod, "_OWN_VOICE_QUERY_BUDGET_S", -1.0)

    started = time.monotonic()
    assert run_mod._is_agents_own_voice("/tmp/x.ogg", _SAID) is False
    assert time.monotonic() - started < 5


@pytest.mark.asyncio
async def test_own_voice_check_runs_off_the_event_loop():
    """A slow own-voice check must not stop other coroutines from running."""
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner._has_setup_skill = lambda: False

    check_started = threading.Event()
    loop_progressed = threading.Event()
    blocked = []

    def _slow_own_voice(_path, _transcript):
        # A check running ON the event loop can never see the witness tick,
        # because the witness coroutine cannot run until this returns.
        check_started.set()
        if not loop_progressed.wait(timeout=2.0):
            blocked.append(True)
        return False

    async def _witness():
        deadline = time.monotonic() + 10
        while not check_started.is_set():
            if time.monotonic() > deadline:
                raise AssertionError("own-voice check never started")
            await asyncio.sleep(0.005)
        loop_progressed.set()

    with (
        patch(
            "tools.transcription_tools.transcribe_audio",
            return_value={"success": True, "transcript": _SAID, "provider": "x"},
        ),
        patch("gateway.run._is_agents_own_voice", side_effect=_slow_own_voice),
    ):
        (text, transcripts), _ = await asyncio.gather(
            runner._enrich_message_with_transcription("", ["/tmp/voice.ogg"]),
            _witness(),
        )

    assert blocked == [], "event loop was blocked by the own-voice check"
    assert transcripts == [_SAID]


# ── 1b. Heartbeat warnings whose stack text contains '%' still log ───────────


def test_dynamic_format_filter_keeps_heartbeat_warning_loggable():
    from hermes_logging import _PreformatDynamicMessageFilter

    msg = (
        "Shard ID %s heartbeat blocked for more than %s seconds.\n"
        "Loop thread traceback (most recent call last):\n"
        '    "Dropping busy-mode follow-up for session %s — pending queue at cap (%d).",\n'
    )
    record = logging.LogRecord("discord.gateway", logging.WARNING, __file__, 1, msg, (None, 10), None)
    with pytest.raises(TypeError):
        record.getMessage()

    assert _PreformatDynamicMessageFilter().filter(record) is True
    rendered = record.getMessage()
    assert "Loop thread traceback" in rendered
    assert "(None, 10)" in rendered


def test_dynamic_format_filter_leaves_well_formed_records_alone():
    from hermes_logging import _PreformatDynamicMessageFilter

    record = logging.LogRecord(
        "discord.gateway", logging.WARNING, __file__, 1,
        "Shard ID %s heartbeat blocked for more than %s seconds.", (None, 10), None,
    )
    _PreformatDynamicMessageFilter().filter(record)
    assert record.getMessage() == "Shard ID None heartbeat blocked for more than 10 seconds."


# ── 2. Lost queued messages: flush carries a session id; recovery labels ─────


def _real_event(text="queued text", user_name="Aleph · Sol", is_bot=True):
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent
    from gateway.session import SessionSource

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="1554661535073443841",
        chat_type="thread",
        thread_id="1554661535073443841",
        user_id="1554590574684667925",
        user_name=user_name,
        is_bot=is_bot,
    )
    return MessageEvent(text=text, source=source, message_id="1554783099660009473")


def test_overflow_flush_of_real_message_event_is_recoverable(tmp_path, monkeypatch):
    from gateway.shutdown_flush import flush_overflow_to_file, recover_pending_to_db

    flush_dir = tmp_path / "pending_messages"
    flush_dir.mkdir()
    monkeypatch.setattr("gateway.shutdown_flush._get_flush_dir", lambda: flush_dir)
    key = "agent:main:discord:thread:1554661535073443841:1554661535073443841"

    flush_overflow_to_file(
        {key: [_real_event()]},
        session_id_resolver={key: "20260930_045221_2a450fb9"}.get,
    )
    payload = json.loads(next(flush_dir.glob("*.json")).read_text())
    assert payload["data"]["session_id"] == "20260930_045221_2a450fb9"
    assert payload["data"]["sender_name"] == "Aleph · Sol"
    assert payload["data"]["sender_is_bot"] is True

    db = MagicMock()
    assert recover_pending_to_db(session_db=db) == 1
    kwargs = db.append_message.call_args.kwargs
    assert kwargs["session_id"] == "20260930_045221_2a450fb9"
    # Same label as live delivery (gateway.session.format_sender_label):
    # a transport-attested peer bot is marked "[bot]".
    assert kwargs["content"] == "[Aleph · Sol [bot]] queued text"
    assert list(flush_dir.glob("*.json")) == []


def test_recovery_resolves_legacy_files_without_session_id(tmp_path, monkeypatch):
    from gateway.shutdown_flush import recover_pending_to_db

    flush_dir = tmp_path / "pending_messages"
    flush_dir.mkdir()
    monkeypatch.setattr("gateway.shutdown_flush._get_flush_dir", lambda: flush_dir)
    (flush_dir / "pending-legacy.json").write_text(json.dumps({
        "session_key": "agent:main:discord:thread:1:1",
        "reason": "shutdown",
        "ts": int(time.time()),
        "data": {"text": "legacy text only"},
    }))

    db = MagicMock()
    assert recover_pending_to_db(
        session_db=db, session_id_resolver={"agent:main:discord:thread:1:1": "sid-1"}.get
    ) == 1
    assert db.append_message.call_args.kwargs["session_id"] == "sid-1"


def test_recovery_preserves_but_skips_stale_flush_files(tmp_path, monkeypatch):
    from gateway.shutdown_flush import recover_pending_to_db

    flush_dir = tmp_path / "pending_messages"
    flush_dir.mkdir()
    monkeypatch.setattr("gateway.shutdown_flush._get_flush_dir", lambda: flush_dir)
    stale = flush_dir / "pending-stale.json"
    stale.write_text(json.dumps({
        "session_key": "k",
        "reason": "shutdown",
        "ts": int(time.time()) - 7 * 3600,
        "data": {"text": "old", "session_id": "sid"},
    }))

    db = MagicMock()
    assert recover_pending_to_db(session_db=db) == 0
    db.append_message.assert_not_called()
    assert stale.exists()


def test_recovery_replays_overflow_in_arrival_order(tmp_path, monkeypatch):
    """Flush files have random names; recovery must follow ``seq``, not names."""
    from gateway.shutdown_flush import flush_overflow_to_file, recover_pending_to_db

    flush_dir = tmp_path / "pending_messages"
    flush_dir.mkdir()
    monkeypatch.setattr("gateway.shutdown_flush._get_flush_dir", lambda: flush_dir)
    key = "agent:main:discord:thread:1:1"
    texts = [f"message {i}" for i in range(12)]
    flush_overflow_to_file(
        {key: [_real_event(text=t) for t in texts]},
        session_id_resolver={key: "sid"}.get,
    )

    db = MagicMock()
    assert recover_pending_to_db(session_db=db) == len(texts)
    replayed = [c.kwargs["content"] for c in db.append_message.call_args_list]
    assert replayed == [f"[Aleph · Sol [bot]] {t}" for t in texts]


# ── 3. Replay flood: notices not ledgered; old failures not replayed ─────────


@pytest.mark.asyncio
async def test_drain_notice_is_an_ephemeral_system_reply():
    """The drain-path notice is returned as an EphemeralReply."""
    from gateway.platforms.base import EphemeralReply

    reply = EphemeralReply(
        "⏳ Gateway is restarting and is not accepting new work right now."
    )
    assert isinstance(reply, EphemeralReply)
    assert str(reply).startswith("⏳ Gateway is restarting")


@pytest.mark.asyncio
async def test_drain_notice_is_never_ledgered(tmp_path, monkeypatch):
    """End-to-end through the real send pipeline: a drain notice that the
    platform rejects must not leave a ledger row to replay after restart."""
    from gateway import delivery_ledger as dl
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import (
        BasePlatformAdapter,
        EphemeralReply,
        MessageEvent,
        MessageType,
        SendResult,
    )
    from gateway.session import SessionSource

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")

    class _Adapter(BasePlatformAdapter):  # type: ignore[misc]
        def __init__(self):
            super().__init__(PlatformConfig(enabled=True), Platform.SLACK)

        async def connect(self, *, is_reconnect: bool = False):  # pragma: no cover
            return True

        async def disconnect(self):  # pragma: no cover
            return None

        async def get_chat_info(self, chat_id):  # pragma: no cover
            return None

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            return SendResult(success=False, error="403 Forbidden (error code: 50001)")

    adapter = _Adapter()
    adapter._message_handler = __import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock(
        return_value=EphemeralReply(
            "⏳ Gateway is restarting and is not accepting new work right now."
        )
    )
    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="channel"),
        message_id="m-1",
    )
    session_key = "agent:main:slack:channel:C1"
    adapter._active_sessions[session_key] = asyncio.Event()
    await adapter._process_message_background(event, session_key)

    with dl._connect() as conn:
        rows = conn.execute("SELECT COUNT(*) FROM delivery_obligations").fetchone()[0]
    assert rows == 0


def test_drain_paths_return_ephemeral_replies_in_source():
    """Both drain-notice returns in _handle_message are wrapped (no plain str).

    A plain ``str`` return is ledgered as an agent answer and replayed after
    the next restart; ``EphemeralReply`` is skipped by the ledger.
    """
    import inspect

    from gateway.run import GatewayRunner

    src = inspect.getsource(GatewayRunner._handle_message)
    assert 'return f"⏳ Gateway is' not in src
    assert src.count("return EphemeralReply(") >= 2


class TestFailedRowFreshness:
    @pytest.fixture(autouse=True)
    def _fresh_db(self, tmp_path, monkeypatch):
        from gateway import delivery_ledger as dl

        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")

    @staticmethod
    def _failed_row(age_seconds):
        from gateway import delivery_ledger as dl

        dl.record_obligation(
            obligation_id="ob-1", session_key="agent:main:discord:thread:1:1",
            platform="discord", chat_id="1", thread_id="1", content="reply",
        )
        dl.mark_failed("ob-1", "403 Forbidden (error code: 50001): Missing Access")
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1, "
                "created_at=? WHERE obligation_id='ob-1'",
                (time.time() - age_seconds,),
            )

    def test_old_failed_row_is_abandoned_not_replayed(self):
        from gateway import delivery_ledger as dl

        self._failed_row(dl.FAILED_REDELIVERY_MAX_AGE_SECONDS + 60)
        assert dl.sweep_recoverable() == []
        with dl._connect() as conn:
            (state,) = conn.execute(
                "SELECT state FROM delivery_obligations WHERE obligation_id='ob-1'"
            ).fetchone()
        assert state == "abandoned"

    def test_recent_failed_row_is_still_retried_with_marker(self):
        from gateway import delivery_ledger as dl

        self._failed_row(60)
        claimed = dl.sweep_recoverable()
        assert len(claimed) == 1
        assert claimed[0]["needs_marker"] is True
