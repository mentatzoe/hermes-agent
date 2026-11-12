"""Real core admissions/SQLite/compaction consumed by the independent monitor.

Set AUTHORED_MONITOR_SOURCE to a monitor checkout's zoe_profile_card.py, or
place monitor-integration beside this checkout. No provider/network calls.
"""

from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import threading
from unittest.mock import patch

import pytest

from agent.agent_runtime_helpers import _merge_consecutive_users
from agent.authored_turn_context import block, snapshots, check_request, text_of
from agent.interrupt_control import InterruptControlMixin
from agent.mid_turn_input import mid_turn_human_row
from agent.turn_context import build_api_messages
from hermes_state import SessionDB
from tests.agent.test_authored_admission import configure
from tests.agent.test_api_content_row_addressed_backfill import _RealPersistenceAgent
from tests.agent.test_turn_context import _build

HUMAN = {"id": "fixture-human", "name": "Pat", "is_bot": False}
BOT = {"id": "fixture-bot", "name": "Helper", "is_bot": True}
T0 = 1_800_000_000.0


@pytest.fixture
def monitor():
    path = Path(
        os.environ.get(
            "AUTHORED_MONITOR_SOURCE",
            str(
                Path(__file__).resolve().parents[3]
                / "monitor-integration"
                / "zoe_profile_card.py"
            ),
        )
    )
    if not path.is_file():
        pytest.skip("requires the independent authored monitor source")
    spec = importlib.util.spec_from_file_location("authored_monitor_integration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Producer(InterruptControlMixin, _RealPersistenceAgent):
    def __init__(self, db):
        super().__init__(db)
        self._model_request_active = threading.Event()
        self._model_request_active.set()
        self._current_turn_timestamp = T0
        self.ephemeral_system_prompt = ""

    def _copy_reasoning_content_for_api(self, source, target):
        pass  # fixture carries no provider reasoning

    def _should_sanitize_tool_calls(self):
        return False


@pytest.fixture(autouse=True)
def no_auxiliary_requests(monkeypatch):
    monkeypatch.setattr(
        "agent.turn_context._maybe_title_session_at_turn_start", lambda *a: None
    )
    monkeypatch.setattr("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None)
    monkeypatch.setattr(
        "agent.context_compressor.get_model_context_length", lambda *a, **k: 100_000
    )


@pytest.fixture
def producer(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="cli")
    agent = configure(Producer(db), tmp_path)
    yield agent, db, tmp_path / "state.db"
    db.close()


def admit(
    agent, *, at=T0, author=HUMAN, text="same", early=False, history=None, metadata=None
):
    if early:
        staged = {"role": "user", "content": text, "timestamp": at}
        agent._pending_cli_user_message = staged
        agent._flush_messages_to_session_db([staged], None)
    with (
        patch("time.time", return_value=at),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        from agent.message_content import flatten_message_text

        ctx = _build(
            agent,
            user_message=text,
            turn_author=author,
            summarize_user_message_for_log=flatten_message_text,
            conversation_history=history,
            persist_user_timestamp=at,
            persist_user_display_metadata=metadata,
        )
        agent._flush_messages_to_session_db(ctx.messages, history)
    return ctx.messages[ctx.current_turn_user_idx]


def report(monitor, path, start=T0 - 1, end=T0 + 100):
    return monitor.scan(path, since=start, until=end)


@pytest.mark.parametrize("early", [False, True])
def test_opener_author_survives_admission_and_early_flush(producer, monitor, early):
    agent, db, path = producer
    row = admit(agent, early=early)
    saved = db.get_messages("s1")[0]
    assert saved["display_metadata"]["turn_author"] == HUMAN
    assert saved["content"] == "same"
    assert snapshots(saved) == snapshots(row)
    assert report(monitor, path)["carried"] == 1


def test_explicit_idless_local_human_is_not_a_platform_guess(producer, monitor):
    agent, db, path = producer
    admit(agent, author=None, metadata={"input_provenance": "local_human"})
    admit(agent, at=T0 + 1, author=None)  # anonymous programmatic CLI-shaped call
    admit(agent, at=T0 + 2, author=BOT)
    result = report(monitor, path)
    assert (result["total"], result["carried"], result["unknown"]) == (2, 1, 1)


def test_nested_entries_have_their_own_window_not_carrier_or_verification_time(
    producer, monitor
):
    agent, db, path = producer
    opener = admit(agent, at=T0 - 10)
    facts = {
        "sender_id": "fixture-human",
        "is_bot": False,
        "typed_user_message": "same",
    }
    # A source fallback's verification date is older than this admission.
    agent._authored_turn_source.path.unlink()
    with patch("time.time", return_value=T0):
        assert agent.steer("same", facts=facts)
    with patch("time.time", return_value=T0 + 10):
        assert agent.steer("same", facts=facts)
    entries = agent._pending_steer
    row = mid_turn_human_row(
        agent, entries, content="framed same and same", display_kind="steer"
    )
    merged, _ = _merge_consecutive_users([opener, row])
    db.archive_and_compact("s1", merged)
    result = report(monitor, path, T0, T0 + 10)
    assert (result["total"], result["carried"], result["degraded"]) == (1, 1, 1)
    result = report(monitor, path, T0 + 10, T0 + 11)
    assert (result["total"], result["carried"]) == (1, 1)


def test_absorbed_uid_snapshots_are_distinct_after_merge_and_resume(producer, monitor):
    agent, db, path = producer
    first = admit(agent)
    second = admit(agent, at=T0 + 1)
    expected = [s["id"] for row in (first, second) for s in snapshots(row)]
    merged, _ = _merge_consecutive_users(deepcopy([first, second]))
    db.archive_and_compact("s1", merged)
    restored = db.get_messages_as_conversation("s1")
    assert set(expected) == {s["id"] for row in restored for s in snapshots(row)}
    result = report(monitor, path)
    assert (result["total"], result["carried"], result["unknown"]) == (2, 2, 0)
    # A new runtime uses stored bytes even after the source changes.
    resumed = configure(Producer(db), path.parent)
    wire, _ = build_api_messages(
        resumed,
        restored,
        current_turn_user_idx=-1,
        ext_prefetch_cache="",
        plugin_user_context="",
        moa_config=None,
        active_system_prompt="SYSTEM",
    )
    assert all(
        block(s) in "\n".join(text_of(r.get("content")) for r in wire)
        for row in restored
        for s in snapshots(row)
    )


def test_multiple_active_admissions_survive_real_full_compaction(producer, monitor):
    from tests.agent.test_cron_inflight_prompt_reappend import (
        _make_compressor,
        _tool_pairs,
    )
    from agent.conversation_compression import (
        _resolve_compress_call,
        _run_summary_dispatch,
    )
    from unittest.mock import MagicMock

    agent, db, path = producer
    opener = admit(agent)
    expected = list(snapshots(opener))
    rows = [{"role": "system", "content": "Fixture"}, opener, *_tool_pairs(20)]
    for i in range(2):
        with patch("time.time", return_value=T0 + i + 1):
            assert agent.steer(
                "same", facts={"sender_id": HUMAN["id"], "is_bot": False}
            )
        entry = agent._pending_steer.pop(0)
        row = mid_turn_human_row(agent, [entry], content="same", display_kind="steer")
        expected.extend(snapshots(row))
        rows.extend([row, *_tool_pairs(20, start=(i + 1) * 20)])
    agent._flush_messages_to_session_db(rows, None)
    agent.context_compressor = _make_compressor()
    compress_fn, kwargs = _resolve_compress_call(
        agent,
        approx_tokens=200_000,
        focus_topic=None,
        force=True,
        memory_context="",
        bypass_cooldown=False,
    )
    reply = MagicMock()
    reply.choices[0].message.content = "## Summary\nTool work continued."
    with patch("agent.context_compressor.call_llm", return_value=reply) as summary_call:
        compressed = _run_summary_dispatch(
            agent,
            rows,
            compress_fn,
            kwargs,
            commit_fence=None,
            attempt_generation=None,
            hard_cancel_event=None,
        )
    assert summary_call.call_count > 0
    db.archive_and_compact("s1", compressed)
    restored = db.get_messages_as_conversation("s1")
    # Repeat against the actual persisted/resumed generation, with enough new
    # tool work to force another summary rather than a no-op replay.
    with patch("agent.context_compressor.call_llm", return_value=reply) as summary_call:
        compressed = _run_summary_dispatch(
            agent,
            restored + _tool_pairs(40, start=100),
            compress_fn,
            kwargs,
            commit_fence=None,
            attempt_generation=None,
            hard_cancel_event=None,
        )
    assert summary_call.call_count > 0
    db.archive_and_compact("s1", compressed)
    restored = db.get_messages_as_conversation("s1")
    wire, _ = build_api_messages(
        agent,
        restored,
        current_turn_user_idx=-1,
        ext_prefetch_cache="",
        plugin_user_context="",
        moa_config=None,
        active_system_prompt="SYSTEM",
    )
    check_request(agent, {"messages": wire})
    result = report(monitor, path)
    assert (result["total"], result["carried"], result["unknown"]) == (3, 3, 0)
    assert all(
        block(s) in "\n".join(text_of(r.get("content")) for r in wire) for s in expected
    )


@pytest.mark.parametrize(
    "author,kind,carried,total",
    [
        (HUMAN, None, 1, 1),
        (BOT, None, 0, 0),
        (None, None, 1, 1),
        (None, "hidden", 0, 0),
    ],
)
def test_real_tui_correction_preserves_local_or_relay_author(
    producer, monitor, monkeypatch, author, kind, carried, total
):
    from tui_gateway import server as srv
    from tests.tui_gateway.test_relay_live_author import _session

    agent, db, path = producer
    session = _session(agent=agent, running=True)
    monkeypatch.setattr(srv, "_load_busy_input_mode", lambda: "steer")
    monkeypatch.setattr(srv, "_session_compression_in_flight", lambda s: False)
    with patch("time.time", return_value=T0):
        response = srv._handle_busy_submit(
            "r", "sid", session, "same", None, turn_author=author, display_kind=kind
        )
    assert response["result"]["status"] == "steered"
    row = mid_turn_human_row(
        agent, agent._pending_steer, content="same", display_kind="steer"
    )
    agent._flush_messages_to_session_db([row], None)
    result = report(monitor, path)
    assert (result["total"], result["carried"]) == (total, carried)


def test_gateway_bot_steer_does_not_become_human(producer, monitor):
    from types import SimpleNamespace
    from gateway.typed_text import authored_steer_kwargs

    agent, db, path = producer
    event = SimpleNamespace(
        text="same",
        internal=False,
        source=SimpleNamespace(user_id=BOT["id"], is_bot=True),
    )
    with patch("time.time", return_value=T0):
        assert agent.steer("same", **authored_steer_kwargs(agent.steer, event))
    row = mid_turn_human_row(
        agent, agent._pending_steer, content="same", display_kind="steer"
    )
    agent._flush_messages_to_session_db([row], None)
    assert report(monitor, path)["total"] == 0


def test_native_codex_inline_text_queue_preserves_occurrences_through_drain(
    producer, monitor, monkeypatch
):
    from contextlib import contextmanager
    from tui_gateway import server as srv
    from tests.tui_gateway.test_relay_live_author import _session

    agent, db, path = producer
    from tests.agent.test_authored_wire import make_agent
    from tests.agent.transports.test_codex_app_server_session import FakeClient
    from agent.transports.codex_app_server_session import CodexAppServerSession

    runtime_home = path.parent / "runtime"
    runtime_home.mkdir()
    agent = make_agent(runtime_home, monkeypatch, "codex_app_server")
    agent._session_db.close()
    agent._session_db, agent.session_id, agent._session_db_created = db, "s1", True
    agent._model_request_active.set()
    client = FakeClient()
    agent._codex_session = CodexAppServerSession(
        cwd=str(runtime_home), client_factory=lambda **kw: client
    )
    session = _session(agent=agent, running=True, session_key="s1")

    @contextmanager
    def owning_db(_session):
        yield db

    monkeypatch.setattr(srv, "_session_db", owning_db)
    monkeypatch.setattr(srv, "_ensure_session_db_row", lambda s: None)

    def surface_write_error(exc, *args):
        raise exc

    monkeypatch.setattr(srv, "_workdir_reraise_disk_full", surface_write_error)
    monkeypatch.setattr(srv, "_load_busy_input_mode", lambda: "steer")
    monkeypatch.setattr(srv, "_session_compression_in_flight", lambda s: False)
    monkeypatch.setattr(srv, "_session_uses_compute_host", lambda *a: False)
    for at in (T0, T0 + 1):
        with patch("time.time", return_value=at):
            response = srv._handle_busy_submit("r", "sid", session, "same", None)
        assert response["result"]["status"] == "queued"
    assert len([session["queued_prompt"], *session.get("queued_prompts", [])]) == 2
    before = report(monitor, path)
    assert (before["total"], before["unknown"]) == (2, 2)
    delivered = []

    def dispatch(rid, sid, session, text, **kwargs):
        srv._adopt_submit_user_row(session, agent, text, text)
        client.queue_notification(
            "item/completed",
            threadId="t",
            turnId="tu1",
            item={"type": "agentMessage", "id": "m1", "text": "done"},
        )
        client.queue_notification(
            "turn/completed",
            threadId="t",
            turn={"id": "turn-fake-001", "status": "completed", "error": None},
        )
        with patch("time.time", return_value=T0 + 20):
            result = agent.run_conversation(text, turn_author=kwargs.get("turn_author"))
        assert result["final_response"] == "done"
        delivered.append(next(r for r in result["messages"] if r["role"] == "user"))

    monkeypatch.setattr(srv, "_run_prompt_submit", dispatch)
    for _ in range(2):
        session["running"] = False
        assert srv._drain_queued_prompt("d", "sid", session)
    assert len(delivered) == 2
    requests = [params for method, params in client.requests if method == "turn/start"]
    assert len(requests) == 2
    for row, request in zip(delivered, requests):
        assert all(block(s) in text_of(request["input"]) for s in snapshots(row))
    result = report(monitor, path, T0, T0 + 2)
    assert (result["total"], result["carried"], result["unknown"]) == (2, 2, 0)
    assert report(monitor, path, T0 + 10, T0 + 30)["total"] == 0


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello",
        "/status",
        "\u0000json:literal",
        [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fixture"}}],
    ],
)
def test_actual_storage_shapes_keep_complete_body(producer, monitor, text):
    agent, db, path = producer
    admit(agent, text=text)
    result = report(monitor, path)
    assert (result["total"], result["carried"], result["unknown"]) == (1, 1, 0)


@pytest.mark.parametrize("damage", ["body", "snapshot", "both"])
def test_monitor_detects_damaged_real_admission_and_is_read_only(
    producer, monitor, damage
):
    import json
    from hashlib import sha256

    agent, db, path = producer
    admit(agent)
    original = db.get_messages("s1")[0]
    if damage in {"body", "both"}:
        db._write_rowcount(
            "UPDATE messages SET api_content=? WHERE id=?", ("same", original["id"])
        )
    if damage in {"snapshot", "both"}:
        meta = original["display_metadata"]
        del meta["authored_turn_context"]
        db._write_rowcount(
            "UPDATE messages SET display_metadata=? WHERE id=?",
            (json.dumps(meta), original["id"]),
        )
    db.close()
    before = sha256(path.read_bytes()).hexdigest()
    result = report(monitor, path)
    assert (result["total"], result["missing"]) == (1, 1)
    assert not result["provider_wire_verified"]
    assert sha256(path.read_bytes()).hexdigest() == before


def test_corrupt_nested_body_on_old_carrier_stays_unknown_in_its_window(
    producer, monitor
):
    import json

    agent, db, path = producer
    opener = admit(agent, at=T0 - 100)
    with patch("time.time", return_value=T0):
        assert agent.steer("later", facts={"sender_id": HUMAN["id"], "is_bot": False})
    steer = mid_turn_human_row(
        agent, agent._pending_steer, content="later", display_kind="steer"
    )
    merged, _ = _merge_consecutive_users([opener, steer])
    db.archive_and_compact("s1", merged)
    latest = db.get_messages("s1")[-1]
    metadata = latest["display_metadata"]
    metadata["mid_turn_entries"][0]["display_metadata"]["authored_turn_context"][0][
        "body"
    ] += "corrupt"
    db._write_rowcount(
        "UPDATE messages SET display_metadata=? WHERE id=?",
        (json.dumps(metadata), latest["id"]),
    )
    result = report(monitor, path, T0, T0 + 1)
    assert (
        result["total"],
        result["carried"],
        result["missing"],
        result["unknown"],
    ) == (1, 0, 0, 1)


def test_conflicting_admission_times_cannot_hide_at_window_boundary(producer, monitor):
    import json

    agent, db, path = producer
    row = admit(agent, at=T0 - 1)
    db.archive_and_compact("s1", [deepcopy(row)])
    latest = db.get_messages("s1")[-1]
    meta = latest["display_metadata"]
    meta["authored_turn_context"][0]["admission"]["timestamp"] = T0
    db._write_rowcount(
        "UPDATE messages SET display_metadata=? WHERE id=?",
        (json.dumps(meta), latest["id"]),
    )
    result = report(monitor, path, T0, T0 + 1)
    assert (result["total"], result["unknown"]) == (1, 1)


def test_unbound_legacy_opener_is_not_hidden_by_new_nested_snapshot(producer, monitor):
    agent, db, path = producer
    source = agent._authored_turn_source
    agent._authored_turn_source = None
    opener = admit(agent)  # established human occurrence, genuinely missing context
    agent._authored_turn_source = source
    with patch("time.time", return_value=T0 + 1):
        assert agent.steer("same", facts={"sender_id": HUMAN["id"], "is_bot": False})
    steer = mid_turn_human_row(
        agent, agent._pending_steer, content="same", display_kind="steer"
    )
    merged, _ = _merge_consecutive_users([opener, steer])
    db.archive_and_compact("s1", merged)
    result = report(monitor, path)
    assert (result["total"], result["missing"], result["carried"]) == (2, 1, 1)


@pytest.mark.parametrize(
    "kind,author,total,carried",
    [(None, None, 1, 1), ("hidden", None, 0, 0), (None, BOT, 0, 0)],
)
def test_local_prompt_submit_boundary_stamps_only_actual_local_inputs(
    producer, monitor, monkeypatch, kind, author, total, carried
):
    from tui_gateway import server as srv
    from tests.tui_gateway.test_relay_live_author import _session, _InlineThread
    from tools.bot_relay import DeliveryAuthor

    agent, db, path = producer
    session = _session(agent=agent, session_key="s1")
    monkeypatch.setitem(srv._sessions, "fixture-local-submit", session)
    monkeypatch.setattr(srv.threading, "Thread", _InlineThread)
    monkeypatch.setattr(srv, "_ensure_active_session_slot", lambda *a: None)
    monkeypatch.setattr(srv, "_session_uses_compute_host", lambda *a: False)
    monkeypatch.setattr(srv, "_persist_session_row_for_submit", lambda *a: None)
    monkeypatch.setattr(srv, "_start_agent_build", lambda *a: None)

    def run(rid, sid, session, text, display_kind, metadata, callback, turn_author):
        with (
            patch("time.time", return_value=T0),
            patch("hermes_cli.plugins.invoke_hook", return_value=[]),
        ):
            ctx = _build(
                agent,
                user_message=text,
                turn_author=turn_author,
                persist_user_display_kind=display_kind,
                persist_user_display_metadata=metadata,
                persist_user_timestamp=T0,
            )
            agent._flush_messages_to_session_db(ctx.messages, None)

    monkeypatch.setattr(srv, "_run_after_agent_ready", run)
    params = {"session_id": "fixture-local-submit", "text": "same"}
    if kind:
        params["display_kind"] = kind
    if author:
        params["_turn_author"] = DeliveryAuthor(author)
    result = srv._methods["prompt.submit"]("r", params)
    assert result["result"]["status"] == "streaming"
    result = report(monitor, path)
    assert (result["total"], result["carried"]) == (total, carried)
