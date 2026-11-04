"""Populated sibling corruption must not authorize workspace removal.

Regression for mentatzoe/aleph-vault#286. The 401-row, small-page fixture and
five corruptions reproduce the issue's supplied fixture, not production data.
These checks do not detect internally consistent row deletion or later owners.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import kanban_ops


DAMAGE = ("header-100", "header-104", "header-16-32", "header-28-32", "empty-root")


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(home / ".hermes"))
    for name in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_WORKSPACES_ROOT"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _query(db, sql, params=()):
    with contextlib.closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as conn:
        return conn.execute(sql, params).fetchall()


def _populated_sibling(workspace, *, shared=True):
    db = kb.boards_root() / "populated" / "kanban.db"
    db.parent.mkdir(parents=True)
    with contextlib.closing(sqlite3.connect(db)) as conn:
        conn.execute("PRAGMA page_size=512")
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, status TEXT, workspace_path TEXT)")
        conn.executemany(
            "INSERT INTO tasks VALUES (?, ?, ?)",
            [(f"t_{n}", "ready", f"/elsewhere/{n}" * 20) for n in range(400)]
            + [("t_live", "running" if shared else "done", str(workspace))],
        )
        conn.commit()
    assert _query(db, "PRAGMA quick_check(1)") == [("ok",)]
    return db


def _damage(db, case):
    # Close every SQLite handle before rewriting fixture bytes. The donor and
    # page construction are the issue's repro, preserved in its assessment.
    original = db.read_bytes()
    data = bytearray(original)
    if case == "empty-root":
        root = _query(db, "SELECT rootpage FROM sqlite_master WHERE name='tasks'")[0][0]
        data[(root - 1) * 512:root * 512] = bytes([13, 0, 0, 0, 0, 2, 0, 0]).ljust(512, b"\0")
    else:
        donor = db.parent / "donor.db"
        with contextlib.closing(sqlite3.connect(donor, isolation_level=None)) as conn:
            conn.execute("PRAGMA page_size=4096")
            conn.execute("CREATE TABLE junk (b BLOB)")
            conn.execute("INSERT INTO junk VALUES (zeroblob(12000))")
            conn.execute("DROP TABLE junk")
        start, stop = {"header-100": (0, 100), "header-104": (0, 104),
                       "header-16-32": (16, 32), "header-28-32": (28, 32)}[case]
        data[start:stop] = donor.read_bytes()[start:stop]
    db.write_bytes(data)
    return original


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


def _workspace(conn, root, kind):
    task = kb.create_task(conn, title="fixture cleanup", workspace_kind=kind)
    if kind == "scratch":
        ws = kb.workspaces_root() / task
        ws.mkdir(parents=True)
        (ws / "kept.txt").write_text("still needed", encoding="utf-8")
    else:
        repo = root / "repo"
        repo.mkdir()
        _git(repo, "init")
        (repo / "kept.txt").write_text("still needed", encoding="utf-8")
        _git(repo, "add", "kept.txt")
        _git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
             "-c", "commit.gpgsign=false", "commit", "-m", "fixture")
        # The positive cleanup control must be genuinely clean and pushed.
        _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
        ws = root / "checkout"
        _git(repo, "worktree", "add", "-b", "fixture/cleanup", str(ws))
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready', workspace_path=? WHERE id=?", (str(ws), task))
    return task, ws


def _exercise(conn, task, flow):
    if flow == "gc":
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='archived' WHERE id=?", (task,))
        assert kanban_ops._cmd_gc(argparse.Namespace(
            event_retention_days=0, log_retention_days=0,
        )) == 0
    elif flow == "parent":
        child = kb.create_task(conn, title="last child", parents=[task], workspace_kind="dir")
        assert kb.complete_task(conn, task, summary="fixture parent", fire_lifecycle_hook=False)
        assert kb.complete_task(conn, child, summary="fixture child", fire_lifecycle_hook=False)
    else:
        assert kb.complete_task(conn, task, summary="fixture done", fire_lifecycle_hook=False)


def _deferrals(conn, task):
    return [json.loads(row[0]) for row in conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind=? ORDER BY id",
        (task, "workspace_cleanup_deferred_shared"),
    )]


@pytest.mark.parametrize("case", DAMAGE)
@pytest.mark.parametrize("flow", ["complete", "gc", "parent"])
@pytest.mark.parametrize("kind", ["scratch", "worktree"])
def test_damaged_populated_sibling_retains_workspace(board, case, flow, kind):
    with kbc.connect_closing() as conn:
        task, ws = _workspace(conn, board, kind)
        db = _populated_sibling(ws)
        original = _damage(db, case)
        # SQLite still names the table. A missing-table test would not expose
        # the false-free path this issue concerns.
        assert _query(db, "SELECT name FROM sqlite_master WHERE name='tasks'") == [("tasks",)]
        assert _query(db, "PRAGMA quick_check(1)") != [("ok",)]
        reason = kbw._workspace_in_use_by_other(conn, task, ws)
        _exercise(conn, task, flow)
        assert (ws / "kept.txt").read_text(encoding="utf-8") == "still needed"
        assert _deferrals(conn, task) == [{"path": str(ws), "reason": "unknown"}]
        assert reason == "unknown"
        db.write_bytes(original)
        assert _query(db, "SELECT workspace_path FROM tasks WHERE id='t_live'") == [(str(ws),)]
        assert kbw._workspace_in_use_by_other(conn, task, ws) == "shared"


@pytest.mark.parametrize("shared", [True, False], ids=["shared", "free"])
@pytest.mark.parametrize("flow", ["complete", "gc", "parent"])
@pytest.mark.parametrize("kind", ["scratch", "worktree"])
def test_healthy_populated_sibling_allows_only_unowned_cleanup(board, shared, flow, kind):
    with kbc.connect_closing() as conn:
        task, ws = _workspace(conn, board, kind)
        _populated_sibling(ws, shared=shared)
        assert kbw._workspace_in_use_by_other(conn, task, ws) == ("shared" if shared else None)
        _exercise(conn, task, flow)
        assert ws.exists() is shared
        expected = [{"path": str(ws), "reason": "shared"}] if shared else []
        assert _deferrals(conn, task) == expected


@pytest.mark.parametrize("flow", ["complete", "gc"])
def test_failed_integrity_read_defers_then_rechecks(board, monkeypatch, flow):
    with kbc.connect_closing() as conn:
        task, ws = _workspace(conn, board, "scratch")
        db = _populated_sibling(ws, shared=False)
        connect = sqlite3.connect

        def deny_check(*args, **kwargs):
            other = connect(*args, **kwargs)
            if args[0] == db.as_uri() + "?mode=ro":
                other.set_authorizer(lambda op, *rest: (
                    sqlite3.SQLITE_DENY if op == sqlite3.SQLITE_PRAGMA else sqlite3.SQLITE_OK
                ))
            return other

        with monkeypatch.context() as patch:
            patch.setattr(kbw.sqlite3, "connect", deny_check)
            _exercise(conn, task, flow)
        assert (ws / "kept.txt").exists()
        assert _deferrals(conn, task) == [{"path": str(ws), "reason": "unknown"}]
        # The failed result is not cached; ordinary GC after the fault clears
        # can collect this same archived candidate.
        _exercise(conn, task, "gc")
        assert not ws.exists()


@pytest.mark.parametrize("shared", [True, False])
def test_damage_elsewhere_is_checked_only_without_an_owner(board, shared):
    with kbc.connect_closing() as conn:
        task, ws = _workspace(conn, board, "scratch")
        db = _populated_sibling(ws, shared=shared)
        with contextlib.closing(sqlite3.connect(db)) as other:
            other.execute("CREATE TABLE events (body TEXT)")
            other.executemany("INSERT INTO events VALUES (?)", [("x" * 300,)] * 400)
            other.commit()
        root = _query(db, "SELECT rootpage FROM sqlite_master WHERE name='events'")[0][0]
        data = bytearray(db.read_bytes())
        data[(root - 1) * 512:root * 512] = bytes([13, 0, 0, 0, 0, 2, 0, 0]).ljust(512, b"\0")
        db.write_bytes(data)
        assert _query(db, "PRAGMA quick_check(1)") != [("ok",)]
        _exercise(conn, task, "complete")
        assert (ws / "kept.txt").exists()
        reason = "shared" if shared else "unknown"
        assert _deferrals(conn, task) == [{"path": str(ws), "reason": reason}]


@pytest.mark.parametrize("values", [[], ["OK"], [None], ["ok", "ok"], ["ok", "damage"]])
def test_only_a_single_exact_ok_can_authorize_cleanup(board, monkeypatch, values):
    """Inject only the pragma response; corruption/cleanup tests use real checks."""
    with kbc.connect_closing() as conn:
        task, ws = _workspace(conn, board, "scratch")
        db = _populated_sibling(ws, shared=False)
        connect = sqlite3.connect

        class CheckResponse(sqlite3.Connection):
            def execute(self, sql, *args, **kwargs):
                if sql == "PRAGMA quick_check(1)":
                    if not values:
                        return super().execute("SELECT 'ok' WHERE 0")
                    return super().execute(" UNION ALL ".join("SELECT ?" for _ in values), values)
                return super().execute(sql, *args, **kwargs)

        def patched_connect(*args, **kwargs):
            if args[0] == db.as_uri() + "?mode=ro":
                kwargs["factory"] = CheckResponse
            return connect(*args, **kwargs)

        monkeypatch.setattr(kbw.sqlite3, "connect", patched_connect)
        _exercise(conn, task, "complete")
        assert (ws / "kept.txt").exists()
        assert _deferrals(conn, task) == [{"path": str(ws), "reason": "unknown"}]


def test_shared_owner_needs_no_integrity_read(board, monkeypatch):
    ws = board / "shared"
    db = _populated_sibling(ws)
    connect = sqlite3.connect
    checks = []

    def patched_connect(*args, **kwargs):
        conn = connect(*args, **kwargs)

        def authorize(op, arg, *rest):
            if op == sqlite3.SQLITE_PRAGMA:
                checks.append(arg)
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        conn.set_authorizer(authorize)
        return conn

    monkeypatch.setattr(kbw.sqlite3, "connect", patched_connect)
    assert kbw._other_board_uses_path(db, "t_live", str(ws)) is True
    assert checks == []


@pytest.mark.parametrize("journal", ["DELETE", "WAL"])
def test_scan_and_check_share_one_read_only_snapshot(board, monkeypatch, journal):
    ws = board / "unused"
    db = _populated_sibling(ws, shared=False)
    connect = sqlite3.connect
    with contextlib.closing(connect(db, isolation_level=None)) as writer:
        assert writer.execute(f"PRAGMA journal_mode={journal}").fetchone()[0] == journal.lower()
    readers = []
    checked = []

    class ObservedConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == "PRAGMA quick_check(1)":
                assert self.in_transaction
                checked.append(self)
                # The scan is read-only even while it holds a transaction.
                with pytest.raises(sqlite3.OperationalError, match="readonly"):
                    super().execute("UPDATE tasks SET status='done'")
                with contextlib.closing(connect(db, timeout=0.05, isolation_level=None)) as writer:
                    if journal == "DELETE":
                        with pytest.raises(sqlite3.OperationalError, match="locked"):
                            writer.execute("BEGIN EXCLUSIVE")
                    else:
                        writer.execute("UPDATE tasks SET status='running' WHERE id='t_live'")
                        assert super().execute(
                            "SELECT status FROM tasks WHERE id='t_live'"
                        ).fetchone()[0] == "done"
                        # A reader can delay checkpoint completion; this isn't
                        # evidence of damage and must not turn free into unknown.
                        checkpoint = writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                        assert checkpoint[0] == 1
            return super().execute(sql, *args, **kwargs)

    def patched_connect(*args, **kwargs):
        assert args[0] == db.as_uri() + "?mode=ro"
        reader = connect(*args, factory=ObservedConnection, **kwargs)
        readers.append(reader)
        return reader

    with monkeypatch.context() as patch:
        patch.setattr(kbw.sqlite3, "connect", patched_connect)
        assert kbw._other_board_uses_path(db, "candidate", str(ws)) is False
    assert len(readers) == 1
    assert checked == readers
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        readers[0].execute("SELECT 1")
    with contextlib.closing(connect(db, timeout=0.05, isolation_level=None)) as writer:
        if journal == "DELETE":
            writer.execute("BEGIN EXCLUSIVE")
        else:
            assert writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
    # A later owner is outside the old snapshot but must appear on a fresh scan.
    assert kbw._other_board_uses_path(db, "candidate", str(ws)) is (journal == "WAL")


def test_exclusive_lock_retains_then_releases_workspace(board):
    with kbc.connect_closing() as conn:
        task, ws = _workspace(conn, board, "scratch")
        db = _populated_sibling(ws, shared=False)
        with contextlib.closing(sqlite3.connect(db, isolation_level=None)) as writer:
            writer.execute("BEGIN EXCLUSIVE")
            _exercise(conn, task, "complete")
        assert (ws / "kept.txt").exists()
        assert _deferrals(conn, task) == [{"path": str(ws), "reason": "unknown"}]
        _exercise(conn, task, "gc")
        assert not ws.exists()
