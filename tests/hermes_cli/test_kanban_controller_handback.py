"""Controller hand-backs must not create replacement work (aleph-vault#267)."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect_closing() as connection:
        yield connection


def _start(conn, creator):
    tid = kb.create_task(conn, title="Issue work", assignee="worker", created_by=creator)
    kb.recompute_ready(conn)
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    return tid


def _block(conn, tid, kind, reason):
    run_id = kb.get_task(conn, tid).current_run_id
    assert kb.block_task(conn, tid, kind=kind, reason=reason, expected_run_id=run_id)
    assert kb.get_task(conn, tid).current_run_id is None


def _resume(conn, tid):
    assert kb.unblock_task(conn, tid)
    assert kb.get_task(conn, tid).status == "ready"
    assert kb.claim_task(conn, tid, claimer="worker") is not None


@pytest.mark.parametrize("creator", ["aleph-vault-controller", "research-vault-controller"])
@pytest.mark.parametrize("reasons", [
    pytest.param(["Question for Zoe", "Merged; Home installs"], id="question-then-setup"),
    pytest.param(["Merged; Home installs", "More work merged; Home installs"], id="setup-twice"),
    pytest.param(["First question for Zoe", "Different question for Zoe"], id="two-questions"),
])
def test_issue_handback_sequences_keep_the_same_blocked_card(conn, creator, reasons, monkeypatch):
    tid = _start(conn, creator)
    decompose = Mock(side_effect=AssertionError("A hand-back is not new work"))
    monkeypatch.setattr(decomp, "decompose_task", decompose)
    for reason in reasons:
        _block(conn, tid, "needs_input", reason)
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.recompute_ready(conn) == 0
        for candidate in decomp.list_triage_ids():
            decomp.decompose_task(candidate)
        assert [task.id for task in kb.list_tasks(conn)] == [tid]
        _resume(conn, tid)
    assert decompose.call_count == 0
    assert not [ev for ev in kb.list_events(conn, tid) if ev.kind == "block_loop_detected"]


@pytest.mark.parametrize("kind", [None, "needs_input", "capability", "transient", "dependency"])
def test_controller_handbacks_stay_sticky_beyond_loop_limit(conn, kind, all_assignees_spawnable):
    tid = _start(conn, "another-controller")
    for _ in range(kb.BLOCK_RECURRENCE_LIMIT + 2):
        _block(conn, tid, kind, "Same hand-back")
        assert kb.get_task(conn, tid).status == "blocked"
        spawn = Mock(side_effect=AssertionError("Blocked work cannot respawn"))
        assert not kbd.dispatch_once(conn, spawn_fn=spawn).spawned
        spawn.assert_not_called()
        _resume(conn, tid)
    assert kb.get_task(conn, tid).block_recurrences == kb.BLOCK_RECURRENCE_LIMIT + 2


@pytest.mark.parametrize("creator", [None, "worker", "controller", "-controller", "my-controller-helper"])
def test_ordinary_cards_still_trip_the_loop_breaker(conn, creator):
    tid = _start(conn, creator)
    for attempt in range(kb.BLOCK_RECURRENCE_LIMIT):
        _block(conn, tid, "needs_input", "Missing capability")
        if attempt + 1 < kb.BLOCK_RECURRENCE_LIMIT:
            _resume(conn, tid)
    assert kb.get_task(conn, tid).status == "triage"
    assert tid in decomp.list_triage_ids()
    assert [ev for ev in kb.list_events(conn, tid) if ev.kind == "block_loop_detected"]


def test_existing_controller_triage_is_not_automatically_decomposed(conn, monkeypatch):
    from gateway.kanban_watchers_dispatcher import _KanbanDispatcher

    controlled = kb.create_task(conn, title="Old hand-back", created_by="aleph-vault-controller", triage=True)
    ordinary = kb.create_task(conn, title="New work", created_by="worker", triage=True)
    assert decomp.list_triage_ids() == [ordinary]
    dispatcher = _KanbanDispatcher(kb, settings=Mock())
    monkeypatch.setattr(dispatcher, "_board_slugs", lambda: ["default"])
    decompose = Mock(return_value=0)
    monkeypatch.setattr(dispatcher, "_decompose_one", decompose)
    assert dispatcher.auto_decompose_tick(3) == 0
    decompose.assert_called_once_with(decomp, "default", ordinary)
    assert kb.get_task(conn, controlled).status == "triage"


def test_controller_dependency_wait_still_resumes_after_parent(conn):
    parent = _start(conn, "worker")
    tid = _start(conn, "aleph-vault-controller")
    kb.link_tasks(conn, parent_id=parent, child_id=tid,
                  expected_child_run_id=kb.get_task(conn, tid).current_run_id)
    _block(conn, tid, "dependency", "Waiting for parent")
    assert kb.get_task(conn, tid).status == "todo"
    assert kb.recompute_ready(conn) == 0
    assert kb.complete_task(conn, parent, result="Parent finished")
    kb.recompute_ready(conn)
    assert kb.get_task(conn, tid).status == "ready"


def test_stale_controller_worker_cannot_block_a_new_run(conn):
    tid = _start(conn, "aleph-vault-controller")
    stale = kb.get_task(conn, tid).current_run_id
    _block(conn, tid, "needs_input", "Question")
    _resume(conn, tid)
    current = kb.get_task(conn, tid).current_run_id
    assert current != stale
    assert not kb.block_task(conn, tid, kind="needs_input", reason="Stale", expected_run_id=stale)
    assert kb.get_task(conn, tid).current_run_id == current
    assert kb.get_task(conn, tid).status == "running"


def test_controller_triage_cannot_hide_ordinary_candidates_behind_limit(conn):
    with kb.write_txn(conn):
        conn.executemany(
            "INSERT INTO tasks (id, title, status, priority, created_at, created_by) "
            "VALUES (?, 'Old hand-back', 'triage', 10, 0, 'aleph-vault-controller')",
            [(f"controller-{index}",) for index in range(1000)],
        )
    other_tenant = kb.create_task(conn, title="Other tenant", triage=True, tenant="other")
    ordinary = kb.create_task(conn, title="Ordinary work", triage=True, tenant="selected")
    assert set(decomp.list_triage_ids()) == {other_tenant, ordinary}
    assert decomp.list_triage_ids(tenant="selected") == [ordinary]
