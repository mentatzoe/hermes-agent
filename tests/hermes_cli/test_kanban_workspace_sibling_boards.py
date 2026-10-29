"""A sibling board database with no tables must not stop scratch cleanup.

Regression for mentatzoe/aleph-vault#283. On 29 Oct 2026 the Mac mini had a 4096-byte
``kanban/boards/default/kanban.db`` with a valid SQLite header and no tables. Before removing a
finished card's workspace, completion, the parent sweep and ``kanban gc`` read every other
board's database for a live task on the same path. Reading the stub raised ``no such table:
tasks``, which counted as "cannot tell", so every finished card kept its directory and recorded
``workspace_cleanup_deferred_shared`` with reason ``unknown``.

A file that holds no tables cannot hold a task, so the scan skips it. A file that cannot be read
at all (damaged, not a database, locked) or that holds unfamiliar tables still blocks removal.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import kanban_ops

_PAGE = 4096

# The file found at ~/.hermes/kanban/boards/default/kanban.db on 29 Oct 2026: the header of an
# empty one-page database written by SQLite 3.50.4, then zeros up to 4096 bytes.
_INCIDENT_STUB_HEADER = bytes.fromhex(
    "53514c69746520666f726d617420330010000101004020200000000100000001"
    "0000000000000000000000010000000000000000000000000000000000000000"
    "0000000000000000000000000000000000000000000000000000000000000001"
    "002e8a140d0000000010"
)


# ---------------------------------------------------------------------------
# Fixtures and sibling-file builders
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _git(*args: str, cwd: str | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=60,
    )
    assert result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr}"
    return result.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A project repo whose history is fully pushed, so its linked worktrees are removable."""
    origin = tmp_path / "origin.git"
    _git("init", "--bare", str(origin))
    project = tmp_path / "project"
    _git("clone", str(origin), str(project))
    _git("-C", str(project), "config", "user.email", "t@example.com")
    _git("-C", str(project), "config", "user.name", "t")
    (project / "README.md").write_text("hello\n", encoding="utf-8")
    _git("-C", str(project), "add", "README.md")
    _git("-C", str(project), "commit", "-m", "init")
    _git("-C", str(project), "push", "origin", "HEAD")
    return project


def _write_board(path: Path, rows: list[tuple[str, str, str]] | None = None) -> int:
    """A sibling board with the columns the scan reads; returns the root page of ``tasks``."""
    if rows is None:
        rows = [(f"t_{n}", "ready", f"/elsewhere/{n}" * 20) for n in range(400)]
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, status TEXT, workspace_path TEXT)")
        conn.executemany("INSERT INTO tasks VALUES (?, ?, ?)", rows)
        conn.commit()
        return conn.execute("SELECT rootpage FROM sqlite_master WHERE name = 'tasks'").fetchone()[0]


def _overwrite_page(path: Path, page_number: int, fill: bytes, *, tail: int = _PAGE) -> None:
    data = bytearray(path.read_bytes())
    end = page_number * _PAGE
    data[end - tail:end] = fill * tail
    path.write_bytes(bytes(data))


# Files that hold no tables. None of them can hold a task.

def _write_empty_file(path: Path) -> None:
    path.write_bytes(b"")


def _write_incident_stub(path: Path) -> None:
    path.write_bytes(_INCIDENT_STUB_HEADER.ljust(_PAGE, b"\0"))


def _write_dropped_tables(path: Path) -> None:
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE tasks (id TEXT)")
        conn.execute("DROP TABLE tasks")
        conn.commit()


def _write_empty_wal_database(path: Path) -> None:
    with contextlib.closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    assert path.read_bytes()[18:20] == b"\x02\x02", "header should say WAL"


# Files the scan cannot read, or does not recognize. Each one must keep blocking removal.

def _write_garbage(path: Path) -> None:
    path.write_bytes(b"this is not a database\n" * 300)


def _write_magic_only(path: Path) -> None:
    path.write_bytes(b"SQLite format 3\0")


def _write_truncated_header(path: Path) -> None:
    path.write_bytes(_INCIDENT_STUB_HEADER[:100])


def _write_board_cut_to_first_page(path: Path) -> None:
    _write_board(path)
    path.write_bytes(path.read_bytes()[:_PAGE])


def _write_board_with_damaged_schema(path: Path) -> None:
    _write_board(path)
    _overwrite_page(path, 1, b"\xaa", tail=200)


def _write_board_with_damaged_tasks_page(path: Path) -> None:
    """The schema reads fine and names ``tasks``; the table's own pages do not read."""
    root = _write_board(path)
    _overwrite_page(path, root, b"\0")


def _write_unfamiliar_tables(path: Path) -> None:
    """Readable, but not a kanban board: there is no way to tell what it tracks."""
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)")
        conn.commit()


def _plant(write, slug: str = "default") -> Path:
    """Create ``<kanban home>/kanban/boards/<slug>/kanban.db`` the way the incident left it."""
    db = kb.boards_root() / slug / "kanban.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    write(db)
    return db


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------


def _bind_scratch_workspace(conn, task_id: str) -> Path:
    ws = Path(kbw.resolve_workspace(kb.get_task(conn, task_id)))
    kbw.set_workspace_path(conn, task_id, ws)
    (ws / "work.txt").write_text("in progress", encoding="utf-8")
    return ws


def _running_scratch_task(conn, title: str = "task") -> tuple[str, Path]:
    tid = kb.create_task(conn, title=title)
    assert kb.claim_task(conn, tid) is not None
    return tid, _bind_scratch_workspace(conn, tid)


def _finished_scratch_task(conn) -> tuple[str, Path]:
    """An archived scratch task whose directory is still on disk."""
    tid = kb.create_task(conn, title="finished")
    ws = kb.workspaces_root() / tid
    ws.mkdir(parents=True)
    (ws / "work.txt").write_text("in progress", encoding="utf-8")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='archived', workspace_kind='scratch', workspace_path=? WHERE id=?",
            (str(ws), tid),
        )
    return tid, ws


def _worktree_task(conn, repo: Path) -> tuple[str, Path]:
    tid = kb.create_task(conn, title="worktree task", assignee="worker")
    wt = repo / ".worktrees" / tid
    kbw._ensure_git_worktree(repo, wt, f"wt/{tid}")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET workspace_kind='worktree', workspace_path=?, branch_name=?, "
            "status='ready' WHERE id=?",
            (str(wt), f"wt/{tid}", tid),
        )
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    return tid, wt


def _deferred_reasons(conn, task_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind=? ORDER BY id",
        (task_id, "workspace_cleanup_deferred_shared"),
    ).fetchall()
    return [json.loads(r["payload"])["reason"] for r in rows]


def _assert_outcome(conn, task_id: str, path: Path, removed: bool) -> None:
    """Removed with no deferral, or kept intact with one deferral of reason ``unknown``."""
    if removed:
        assert not path.exists()
        assert _deferred_reasons(conn, task_id) == []
    else:
        assert path.is_dir()
        assert _deferred_reasons(conn, task_id) == ["unknown"]


# ---------------------------------------------------------------------------
# The scan: which sibling files count as "cannot tell"
# ---------------------------------------------------------------------------

_NO_TABLES = [
    pytest.param(_write_empty_file, id="zero-byte-file"),
    pytest.param(_write_incident_stub, id="header-only-4096-byte-stub-from-the-incident"),
    pytest.param(_write_dropped_tables, id="every-table-dropped"),
    pytest.param(_write_empty_wal_database, id="empty-database-in-wal-mode"),
]

_CANNOT_TELL = [
    pytest.param(_write_garbage, id="not-a-database"),
    pytest.param(_write_magic_only, id="sqlite-magic-string-only"),
    pytest.param(_write_truncated_header, id="header-cut-short"),
    pytest.param(_write_board_cut_to_first_page, id="board-cut-to-its-first-page"),
    pytest.param(_write_board_with_damaged_schema, id="damaged-schema-page"),
    pytest.param(_write_board_with_damaged_tasks_page, id="schema-intact-tasks-pages-damaged"),
    pytest.param(_write_unfamiliar_tables, id="tables-but-no-tasks-table"),
]


@pytest.mark.parametrize("write", _NO_TABLES)
def test_sibling_board_file_without_tables_is_not_a_sharer(kanban_home, write):
    _plant(write)
    with kbc.connect_closing() as conn:
        tid, ws = _finished_scratch_task(conn)
        assert kbw._workspace_in_use_by_other(conn, tid, ws) is None


@pytest.mark.parametrize("write", _CANNOT_TELL)
def test_sibling_board_file_that_cannot_be_read_still_blocks_removal(kanban_home, write):
    _plant(write)
    with kbc.connect_closing() as conn:
        tid, ws = _finished_scratch_task(conn)
        assert kbw._workspace_in_use_by_other(conn, tid, ws) == "unknown"


def test_locked_sibling_board_still_blocks_removal(kanban_home):
    """A board another process holds exclusively can't be read now, so removal waits."""
    db = _plant(lambda path: _write_board(path, rows=[]))
    holder = sqlite3.connect(db, isolation_level=None)
    try:
        holder.execute("BEGIN EXCLUSIVE")
        with kbc.connect_closing() as conn:
            tid, ws = _finished_scratch_task(conn)
            assert kbw._workspace_in_use_by_other(conn, tid, ws) == "unknown"
    finally:
        holder.close()


@pytest.mark.platforms("posix")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores file modes")
def test_sibling_board_the_process_may_not_read_still_blocks_removal(kanban_home):
    db = _plant(lambda path: _write_board(path, rows=[]))
    db.chmod(0)
    try:
        with kbc.connect_closing() as conn:
            tid, ws = _finished_scratch_task(conn)
            assert kbw._workspace_in_use_by_other(conn, tid, ws) == "unknown"
    finally:
        db.chmod(0o600)


@pytest.mark.parametrize("empty_last", [False, True], ids=["empty-board-scanned-first", "empty-board-scanned-last"])
def test_empty_sibling_does_not_hide_a_live_sharer_on_another_board(kanban_home, monkeypatch, empty_last):
    """Skipping a file with no tables must not end the scan, whichever order the boards come in."""
    _plant(_write_incident_stub, "empty-board")
    kb.create_board("live-board")
    real_scan = kbw._sibling_board_db_files

    def ordered_scan(conn):
        found = real_scan(conn)
        assert {p.parent.name for p in found} == {"empty-board", "live-board"}
        # iterdir() order is up to the filesystem; fix it so both orders are exercised.
        return sorted(found, key=lambda p: p.parent.name != "empty-board", reverse=empty_last)

    monkeypatch.setattr(kbw, "_sibling_board_db_files", ordered_scan)
    with kbc.connect_closing() as conn:
        tid, ws = _finished_scratch_task(conn)
    with kbc.connect_closing(board="live-board") as other:
        live = kb.create_task(other, title="still working")
        with kb.write_txn(other):
            other.execute(
                "UPDATE tasks SET status='ready', workspace_kind='scratch', workspace_path=? WHERE id=?",
                (str(ws), live),
            )
    with kbc.connect_closing() as conn:
        assert kbw._workspace_in_use_by_other(conn, tid, ws) == "shared"
    with kbc.connect_closing(board="live-board") as other, kb.write_txn(other):
        other.execute("UPDATE tasks SET status='done' WHERE id=?", (live,))
    with kbc.connect_closing() as conn:
        assert kbw._workspace_in_use_by_other(conn, tid, ws) is None


def test_board_whose_tables_are_only_in_the_wal_is_not_mistaken_for_an_empty_file(kanban_home):
    """With nothing checkpointed, the main file looks empty; only the WAL knows the board has tasks."""
    db = kb.boards_root() / "wal-board" / "kanban.db"
    db.parent.mkdir(parents=True)
    writer = sqlite3.connect(db, isolation_level=None)
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, status TEXT, workspace_path TEXT)")
        with kbc.connect_closing() as conn:
            tid, ws = _finished_scratch_task(conn)
            writer.execute("INSERT INTO tasks VALUES ('t_live', 'running', ?)", (str(ws),))
            with contextlib.closing(sqlite3.connect(db.as_uri() + "?immutable=1", uri=True)) as main_only:
                assert main_only.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] == 0
            assert kbw._workspace_in_use_by_other(conn, tid, ws) == "shared"
    finally:
        writer.close()


# ---------------------------------------------------------------------------
# The real cleanup paths, with a sibling file planted where the incident had it
# ---------------------------------------------------------------------------

_SIBLINGS = [
    pytest.param(_write_incident_stub, True, id="no-tables-removed"),
    pytest.param(_write_garbage, False, id="unreadable-kept"),
]


@pytest.mark.parametrize(("write", "removed"), _SIBLINGS)
def test_completing_a_scratch_task_removes_its_workspace(kanban_home, write, removed):
    _plant(write)
    with kbc.connect_closing() as conn:
        tid, ws = _running_scratch_task(conn)
        assert kb.complete_task(conn, tid, summary="done")
        _assert_outcome(conn, tid, ws, removed)


@pytest.mark.parametrize(("write", "removed"), _SIBLINGS)
def test_last_child_completing_sweeps_the_parents_scratch_workspace(kanban_home, write, removed):
    _plant(write)
    with kbc.connect_closing() as conn:
        parent, parent_ws = _running_scratch_task(conn, "parent")
        child = kb.create_task(conn, title="child")
        kb.link_tasks(conn, parent, child)
        assert kb.complete_task(conn, parent, summary="parent done")
        assert parent_ws.is_dir()  # the child may still need the hand-off files
        assert kb.claim_task(conn, child) is not None
        child_ws = _bind_scratch_workspace(conn, child)
        assert kb.complete_task(conn, child, summary="child done")
        _assert_outcome(conn, child, child_ws, removed)
        _assert_outcome(conn, parent, parent_ws, removed)


@pytest.mark.parametrize(("write", "removed"), _SIBLINGS)
def test_kanban_gc_removes_an_archived_scratch_workspace(kanban_home, write, removed):
    _plant(write)
    with kbc.connect_closing() as conn:
        tid, ws = _finished_scratch_task(conn)
    args = argparse.Namespace(event_retention_days=30, log_retention_days=30)
    assert kanban_ops._cmd_gc(args) == 0
    with kbc.connect_closing() as conn:
        _assert_outcome(conn, tid, ws, removed)


@pytest.mark.parametrize(("write", "removed"), _SIBLINGS)
def test_completing_a_worktree_task_removes_its_clean_worktree(kanban_home, repo, write, removed):
    _plant(write)
    with kbc.connect_closing() as conn:
        tid, wt = _worktree_task(conn, repo)
        assert kb.complete_task(conn, tid, summary="done")
        _assert_outcome(conn, tid, wt, removed)


@pytest.mark.parametrize(("write", "removed"), _SIBLINGS)
def test_kanban_gc_removes_an_archived_worktree(kanban_home, repo, write, removed):
    """``kanban gc`` backstops worktrees that escaped the completion hook, with the same scan."""
    _plant(write)
    with kbc.connect_closing() as conn:
        tid, wt = _worktree_task(conn, repo)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='archived' WHERE id=?", (tid,))
    args = argparse.Namespace(event_retention_days=30, log_retention_days=30)
    assert kanban_ops._cmd_gc(args) == 0
    with kbc.connect_closing() as conn:
        _assert_outcome(conn, tid, wt, removed)


@pytest.mark.parametrize(("write", "removed"), _SIBLINGS)
def test_last_child_completing_sweeps_the_parents_worktree(kanban_home, repo, write, removed):
    """The parent sweep takes its own branch for a worktree parent."""
    _plant(write)
    with kbc.connect_closing() as conn:
        parent, parent_wt = _worktree_task(conn, repo)
        child = kb.create_task(conn, title="child")
        kb.link_tasks(conn, parent, child)
        assert kb.complete_task(conn, parent, summary="parent done")
        assert parent_wt.is_dir()  # the child may still need the hand-off files
        assert kb.claim_task(conn, child) is not None
        child_ws = _bind_scratch_workspace(conn, child)
        assert kb.complete_task(conn, child, summary="child done")
        _assert_outcome(conn, child, child_ws, removed)
        _assert_outcome(conn, parent, parent_wt, removed)
