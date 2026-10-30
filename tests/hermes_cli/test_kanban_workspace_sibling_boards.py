"""A sibling board database with no tables must not stop scratch cleanup.

Regression for mentatzoe/aleph-vault#283. On 29 Oct 2026 the Mac mini had a 4096-byte
``kanban/boards/default/kanban.db`` with a valid SQLite header and no tables. Before removing a
finished card's workspace, completion, the parent sweep and ``kanban gc`` read every other
board's database for a live task on the same path. Reading the stub raised ``no such table:
tasks``, which counted as "cannot tell", so every finished card kept its directory and recorded
``workspace_cleanup_deferred_shared`` with reason ``unknown``.

A file that holds no tables cannot hold a task, so the scan skips it. The skip needs proof that
the file is intact: a damaged schema page can make a live board read as empty, so a file with no
tables must also pass SQLite's own integrity check, and it must be exactly as long as its own
header says (an old, empty first page written back over a used board passes the integrity check,
whether the board's pages then sit behind the pages the header counts or inside the larger page
it declares). A file that cannot be read at all (damaged, not a database, locked) or that holds
unfamiliar tables still blocks removal.

The scan must also leave SQLite's own locks alone. A plain ``open()`` and ``close()`` of a database
file cancels every lock this process holds on it, so a header read that runs while a connection of
ours is in the middle of a read lets a writer in another process commit under that read. The scan
reads the header before it opens its own connection and opens that connection tracked, so a second
scan of the same file refuses its header read instead of running it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import kanban_ops
from hermes_cli.sqlite_safe_read import connect_tracked, has_live_connection

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


def _write_board(
    path: Path, rows: list[tuple[str, str, str]] | None = None, *, page_size: int | None = None
) -> int:
    """A sibling board with the columns the scan reads; returns the root page of ``tasks``."""
    if rows is None:
        rows = [(f"t_{n}", "ready", f"/elsewhere/{n}" * 20) for n in range(400)]
    with contextlib.closing(sqlite3.connect(path)) as conn:
        if page_size is not None:
            conn.execute(f"PRAGMA page_size={page_size}")
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
    """A board that was used and then emptied: the dropped pages sit on the free list."""
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE tasks (id TEXT)")
        conn.executemany("INSERT INTO tasks VALUES (?)", [(f"t_{n}" * 20,) for n in range(500)])
        conn.commit()
        conn.execute("DROP TABLE tasks")
        conn.commit()


def _write_emptied_auto_vacuum_database(path: Path) -> None:
    """Used, then emptied, with the file shrinking as it goes (pointer-map pages in play)."""
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA auto_vacuum=FULL")
        conn.execute("CREATE TABLE tasks (id TEXT)")
        conn.executemany("INSERT INTO tasks VALUES (?)", [(f"t_{n}" * 20,) for n in range(500)])
        conn.commit()
        conn.execute("DROP TABLE tasks")
        conn.commit()


def _write_emptied_database_with_64k_pages(path: Path) -> None:
    """Pages of 65536 bytes, which the file header stores as 1 (the largest size does not fit in two bytes)."""
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA page_size=65536")
        conn.execute("CREATE TABLE tasks (id TEXT)")
        conn.executemany("INSERT INTO tasks VALUES (?)", [(f"t_{n}" * 20,) for n in range(500)])
        conn.commit()
        conn.execute("DROP TABLE tasks")
        conn.commit()
    assert path.read_bytes()[16:18] == b"\x00\x01", "the header should store 65536 as 1"
    assert path.stat().st_size > 65536, "the emptied file should still hold its old pages"


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


def _damage_schema_page(path: Path, how: str) -> None:
    """Corrupt page 1 (the schema) so SQLite reads the database as having no tables.

    SQLite raises no error for either shape, and every other page, including the one that holds
    the tasks, stays in the file. ``PRAGMA quick_check`` is the thing that notices.
    """
    data = bytearray(path.read_bytes())
    page_size = int.from_bytes(data[16:18], "big")
    if page_size == 1:  # the header stores 65536 as 1
        page_size = 65536
    if how == "zeroed-cell-count":
        assert data[100] == 13 and int.from_bytes(data[103:105], "big") > 0, "page 1 is not a populated leaf"
        data[103:105] = b"\0\0"
    else:
        assert how == "replaced-by-empty-leaf"
        data[100:page_size] = bytes(page_size - 100)
        data[100] = 13
        data[105:107] = (page_size % 65536).to_bytes(2, "big")
    path.write_bytes(bytes(data))


def _write_board_with_zeroed_schema_cell_count(path: Path) -> None:
    _write_board(path)
    _damage_schema_page(path, "zeroed-cell-count")


def _write_board_with_schema_replaced_by_empty_leaf(path: Path) -> None:
    _write_board(path)
    _damage_schema_page(path, "replaced-by-empty-leaf")


def _empty_first_page(*, wal: bool, page_size: int = _PAGE) -> bytes:
    """The first page of an empty database as SQLite writes it: a ``VACUUM`` of a new file."""
    with tempfile.TemporaryDirectory() as scratch:
        db = Path(scratch) / "empty.db"
        with contextlib.closing(sqlite3.connect(db)) as conn:
            conn.execute(f"PRAGMA page_size={page_size}")
            if wal:
                assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
            conn.execute("VACUUM")
        page = db.read_bytes()
    assert len(page) == page_size
    return page


def _replay_empty_first_page(
    path: Path, length: int = _PAGE, *, current_header: bool = True, donor_page_size: int = _PAGE
) -> None:
    """Write the first *length* bytes of an old, empty first page back over a used board.

    The page comes from an empty database in the board's own journal mode. Every other page
    stays where it was. The replayed header counts the pages of that old database and SQLite
    trusts the count, so it reads an empty database of one page: both integrity checks walk the
    pages the header counts and find nothing wrong, while the board's own pages sit outside
    them, unread. Behind the counted page when the old database had pages of the board's size
    or smaller than the file. Inside the page the header declares when the old database had
    larger pages and the file is shorter than one of them (``donor_page_size``). This is a
    constructed replay. The incident's stub holds no pages behind its first one and cannot
    produce it. With ``current_header=False`` the version-valid-for number no longer matches the
    change counter, SQLite stops trusting the count and sizes the database by the file instead.
    """
    data = bytearray(path.read_bytes())
    assert length <= len(data), "a longer replay would grow the file instead of covering it"
    data[:length] = _empty_first_page(wal=data[18] == 2, page_size=donor_page_size)[:length]
    if not current_header:
        data[92:96] = (int.from_bytes(data[24:28], "big") + 1).to_bytes(4, "big")
    path.write_bytes(bytes(data))


def _write_board_with_empty_first_page_replayed(path: Path) -> None:
    _write_board(path)
    _replay_empty_first_page(path)


def _write_board_with_first_108_bytes_replayed(path: Path) -> None:
    """Only the 100-byte header and the 8-byte header of the empty schema leaf come back."""
    _write_board(path)
    _replay_empty_first_page(path, 108)


def _write_board_with_untrusted_empty_header_replayed(path: Path) -> None:
    _write_board(path)
    _replay_empty_first_page(path, current_header=False)


def _write_incident_stub_with_a_page_after_it(path: Path) -> None:
    """The stub's one-page header, then a second page that the header does not count."""
    path.write_bytes(_INCIDENT_STUB_HEADER.ljust(_PAGE, b"\0") + bytes(_PAGE))


def _write_emptied_database_followed_by_a_page_its_header_does_not_count(path: Path) -> None:
    """A database that really was emptied (its header counts every page it holds), then one more page.

    The header counts many pages here, not one, so a rule fitted to the replay (a header that
    counts a single page) would let this through.
    """
    _write_dropped_tables(path)
    assert int.from_bytes(path.read_bytes()[28:32], "big") > 2
    path.write_bytes(path.read_bytes() + bytes(_PAGE))


def _write_small_board_with_larger_page_replayed(path: Path) -> None:
    """The first 108 bytes of an old, empty database with 64 KiB pages over a board of 512-byte pages.

    The board is 1536 bytes, shorter than the one 65536-byte page the replayed header declares.
    """
    _write_board(path, [("t_live", "running", "/elsewhere/live")], page_size=512)
    _replay_empty_first_page(path, 108, donor_page_size=65536)


def _write_incident_stub_cut_to(length: int):
    """The incident's stub with only its first *length* bytes: shorter than the one page its header declares."""

    def write(path: Path) -> None:
        path.write_bytes(_INCIDENT_STUB_HEADER.ljust(_PAGE, b"\0")[:length])

    return write


def _write_incident_stub_with_page_count_zeroed(path: Path) -> None:
    """A header that counts no pages, as SQLite wrote before 3.7.0: it declares nothing, so the file is unexplained."""
    data = bytearray(_INCIDENT_STUB_HEADER.ljust(_PAGE, b"\0"))
    data[28:32] = bytes(4)
    path.write_bytes(bytes(data))


def _board_with_live_sharer(kind: str, workspace: Path | str, *, page_size: int | None = None) -> Path:
    """A sibling board whose running task uses *workspace*; returns the file the scan reads.

    ``minimal`` has only the columns the scan reads. ``full`` is a board made by
    ``kb.create_board``, with the whole schema, checkpointed so the main file holds every page.
    With *page_size* the board is small: pages of that size, and (for ``minimal``) only the
    one running task, so the file is shorter than a single page of a larger size.
    """
    if kind == "minimal":
        rows = [(f"t_{n}", "ready", f"/elsewhere/{n}" * 20) for n in range(400)] if page_size is None else []
        rows.append(("t_live", "running", str(workspace)))
        db = kb.boards_root() / "sharer" / "kanban.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        _write_board(db, rows, page_size=page_size)
        return db
    assert kind == "full"
    if page_size is not None:
        # The page size of an existing file is the one thing ``create_board`` leaves alone.
        seed = kb.boards_root() / "sharer" / "kanban.db"
        seed.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.closing(sqlite3.connect(seed, isolation_level=None)) as conn:
            conn.execute(f"PRAGMA page_size={page_size}")
            conn.execute("VACUUM")
    kb.create_board("sharer")
    with kbc.connect_closing(board="sharer") as other:
        live = kb.create_task(other, title="still using the report")
        with kb.write_txn(other):
            other.execute(
                "UPDATE tasks SET status='running', workspace_path=? WHERE id=?",
                (str(workspace), live),
            )
        other.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return Path(other.execute("PRAGMA database_list").fetchone()[2])


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
    pytest.param(_write_emptied_auto_vacuum_database, id="emptied-auto-vacuum-database"),
    pytest.param(_write_emptied_database_with_64k_pages, id="emptied-database-with-64k-pages"),
    pytest.param(_write_empty_wal_database, id="empty-database-in-wal-mode"),
]

_CANNOT_TELL = [
    pytest.param(_write_garbage, id="not-a-database"),
    pytest.param(_write_magic_only, id="sqlite-magic-string-only"),
    pytest.param(_write_truncated_header, id="header-cut-short"),
    pytest.param(_write_board_cut_to_first_page, id="board-cut-to-its-first-page"),
    pytest.param(_write_board_with_damaged_schema, id="damaged-schema-page"),
    pytest.param(_write_board_with_zeroed_schema_cell_count, id="schema-page-cell-count-zeroed-reads-empty"),
    pytest.param(_write_board_with_schema_replaced_by_empty_leaf, id="schema-page-replaced-reads-empty"),
    pytest.param(_write_board_with_empty_first_page_replayed, id="empty-first-page-replayed-over-a-used-board"),
    pytest.param(_write_board_with_first_108_bytes_replayed, id="first-108-bytes-of-an-empty-page-replayed"),
    pytest.param(_write_board_with_untrusted_empty_header_replayed, id="empty-first-page-replayed-header-not-trusted"),
    pytest.param(_write_incident_stub_with_a_page_after_it, id="incident-stub-with-a-page-the-header-does-not-count"),
    pytest.param(
        _write_emptied_database_followed_by_a_page_its_header_does_not_count,
        id="emptied-database-with-a-page-the-header-does-not-count",
    ),
    pytest.param(_write_small_board_with_larger_page_replayed, id="empty-first-page-from-larger-pages-replayed"),
    pytest.param(_write_incident_stub_cut_to(107), id="incident-stub-cut-to-107-bytes"),
    pytest.param(_write_incident_stub_cut_to(512), id="incident-stub-cut-to-512-bytes"),
    pytest.param(_write_incident_stub_cut_to(_PAGE - 1), id="incident-stub-one-byte-short-of-its-page"),
    pytest.param(_write_incident_stub_with_page_count_zeroed, id="incident-stub-header-counts-no-pages"),
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
    assert not has_live_connection(db), "the scan's own connection must not outlive the scan"


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


def _empty_through_the_wal(db: Path) -> sqlite3.Connection:
    """A WAL-mode board that was filled, checkpointed, then emptied by DROP TABLE and VACUUM.

    The emptied database exists only in the log. The main file keeps its old pages and its old
    length until the next checkpoint. The returned connection keeps the log in place: close it.
    """
    writer = sqlite3.connect(db, isolation_level=None)
    assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, status TEXT, workspace_path TEXT, filler BLOB)")
    writer.execute("INSERT INTO tasks VALUES ('t_old', 'done', '/elsewhere', zeroblob(131072))")
    writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    writer.execute("DROP TABLE tasks")
    writer.execute("VACUUM")
    return writer


def test_board_emptied_through_the_wal_is_skipped_while_its_main_file_is_still_longer(kanban_home):
    """DROP TABLE and VACUUM commit an empty database to the log. Until the next checkpoint the
    main file keeps its old length, so it is longer than the database SQLite reads. Its own
    header still counts every page it holds, so this is an empty board, not a damaged one.

    Pins the other side of the extent rule: comparing the file with SQLite's page count instead
    of with the main file's own header would call this board unreadable.
    """
    db = kb.boards_root() / "emptied-board" / "kanban.db"
    db.parent.mkdir(parents=True)
    writer = _empty_through_the_wal(db)
    try:
        assert db.stat().st_size > 2 * _PAGE, "nothing has been checkpointed yet"
        with contextlib.closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as reader:
            assert reader.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone() is None
            assert reader.execute("PRAGMA page_count").fetchone()[0] * _PAGE < db.stat().st_size
        with kbc.connect_closing() as conn:
            tid, ws = _finished_scratch_task(conn)
            assert kbw._workspace_in_use_by_other(conn, tid, ws) is None
    finally:
        writer.close()


def _first_page_the_log_holds_for_the_empty_database(db: Path) -> bytes:
    """The newest copy of page 1 in a board's write-ahead log: the empty database's first page."""
    log = Path(str(db) + "-wal").read_bytes()
    page_size = int.from_bytes(log[8:12], "big")
    frame = 24 + page_size
    newest = None
    for n in range((len(log) - 32) // frame):
        offset = 32 + n * frame
        if int.from_bytes(log[offset:offset + 4], "big") == 1:
            newest = log[offset + 24:offset + frame]
    assert newest is not None, "the log should hold a copy of page 1"
    return newest


def test_board_whose_checkpoint_stopped_after_page_one_is_unreadable_until_a_writer_finishes_it(kanban_home):
    """A checkpoint copies pages from the log to the main file in order and cuts the file to its
    new length last. One that stops in between leaves the empty first page in front of the old
    pages, the shape of the stale-page replay. The scan cannot tell the two apart, so it waits
    (unknown); the next writer to finish the checkpoint settles it and the board is skipped.
    """
    db = kb.boards_root() / "emptied-board" / "kanban.db"
    db.parent.mkdir(parents=True)
    writer = _empty_through_the_wal(db)
    try:
        with kbc.connect_closing() as conn:
            tid, ws = _finished_scratch_task(conn)
            assert kbw._workspace_in_use_by_other(conn, tid, ws) is None
            data = bytearray(db.read_bytes())
            data[:_PAGE] = _first_page_the_log_holds_for_the_empty_database(db)
            db.write_bytes(bytes(data))
            interrupted_size = db.stat().st_size
            assert interrupted_size > 2 * _PAGE
            assert kbw._workspace_in_use_by_other(conn, tid, ws) == "unknown"
            writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            assert db.stat().st_size < interrupted_size, "the checkpoint should have cut the file"
            assert kbw._workspace_in_use_by_other(conn, tid, ws) is None
    finally:
        writer.close()


def _zero(start: int, stop: int):
    return lambda data: data.__setitem__(slice(start, stop), bytes(stop - start))


def _page_size_that_is_not_a_power_of_two_yet_matches_the_length(data: bytearray) -> None:
    """Pad the file to a multiple of 12288 and make the header say 12288-byte pages.

    The header then counts exactly the bytes in the file, so a length check alone is satisfied.
    Only the rule that a page size is a power of two can object.
    """
    page_size = 12288
    data.extend(bytes(-len(data) % page_size))
    data[16:18] = page_size.to_bytes(2, "big")
    data[28:32] = (len(data) // page_size).to_bytes(4, "big")
    assert int.from_bytes(data[28:32], "big") * page_size == len(data)


_MAIN_FILE_HEADER_DAMAGE = [
    pytest.param(lambda data: data.__setitem__(slice(0, 16), b"X" * 16), id="magic-string-overwritten"),
    pytest.param(_zero(16, 18), id="page-size-zeroed"),
    # SQLite takes the real page size from the log and reads on. 8000 is not a power of two and
    # (page count x page size) does not equal the length, so two rules refuse it. The next case
    # makes the length match, so that only the power-of-two rule is left.
    pytest.param(lambda data: data.__setitem__(slice(16, 18), (8000).to_bytes(2, "big")), id="page-size-not-a-power-of-two"),
    pytest.param(
        _page_size_that_is_not_a_power_of_two_yet_matches_the_length,
        id="page-size-not-a-power-of-two-and-the-length-matches",
    ),
    pytest.param(_zero(0, _PAGE), id="first-page-zeroed"),
]


@pytest.mark.parametrize("damage", _MAIN_FILE_HEADER_DAMAGE)
def test_board_emptied_through_the_wal_is_not_skipped_when_its_main_file_header_is_damaged(kanban_home, damage):
    """The log still holds a good first page, so SQLite reads an empty database and both
    integrity checks pass, whatever the main file's own first 100 bytes say. A main file whose
    header does not describe it cannot vouch for its length, so the pages behind it are
    unexplained and the board counts as unreadable.
    """
    db = kb.boards_root() / "emptied-board" / "kanban.db"
    db.parent.mkdir(parents=True)
    writer = _empty_through_the_wal(db)
    try:
        with kbc.connect_closing() as conn:
            tid, ws = _finished_scratch_task(conn)
            assert kbw._workspace_in_use_by_other(conn, tid, ws) is None
            data = bytearray(db.read_bytes())
            damage(data)
            db.write_bytes(bytes(data))
            with contextlib.closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as reader:
                assert reader.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone() is None
                assert reader.execute("PRAGMA quick_check(1)").fetchall() == [("ok",)]
            assert kbw._workspace_in_use_by_other(conn, tid, ws) == "unknown"
    finally:
        writer.close()


# A board with a running task on the completing task's workspace, then damaged so that SQLite
# reads it as having no tables. The task's pages are still in the file; only the schema is gone.
_DAMAGED_TO_READ_EMPTY = [
    pytest.param("minimal", "zeroed-cell-count", id="minimal-board-cell-count-zeroed"),
    pytest.param("minimal", "replaced-by-empty-leaf", id="minimal-board-schema-replaced"),
    pytest.param("full", "replaced-by-empty-leaf", id="full-board-schema-replaced"),
]


@pytest.mark.parametrize(("kind", "how"), _DAMAGED_TO_READ_EMPTY)
def test_the_damage_reads_as_an_empty_schema_without_any_error(kanban_home, kind, how):
    """Pins the premise of the next test: SQLite itself reports nothing wrong with these files.

    Only the integrity check notices. If a newer SQLite starts raising here, the guard is no
    longer what protects these shapes, and this is the test that says so.
    """
    db = _board_with_live_sharer(kind, "/elsewhere/ws")
    _damage_schema_page(db, how)
    with contextlib.closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as reader:
        assert reader.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone() is None
        assert reader.execute("PRAGMA quick_check(1)").fetchall() != [("ok",)]


@pytest.mark.parametrize(("kind", "how"), _DAMAGED_TO_READ_EMPTY)
def test_damaged_board_that_reads_as_empty_keeps_the_workspace_it_may_still_use(kanban_home, kind, how):
    with kbc.connect_closing() as conn:
        tid, ws = _running_scratch_task(conn)
        db = _board_with_live_sharer(kind, ws)
        assert kbw._workspace_in_use_by_other(conn, tid, ws) == "shared"
        _damage_schema_page(db, how)
        assert kbw._workspace_in_use_by_other(conn, tid, ws) == "unknown"
        assert kb.complete_task(conn, tid, summary="done")
        _assert_outcome(conn, tid, ws, removed=False)


# The same boards with their first page replaced by an old, empty one (the whole page, or only
# its first 108 bytes: the file header and the header of the empty schema leaf). The header now
# counts one page, SQLite trusts that count, and the board's own pages sit behind it.
_EMPTY_FIRST_PAGE_REPLAYED = [
    pytest.param("minimal", _PAGE, id="minimal-board-whole-first-page"),
    pytest.param("minimal", 108, id="minimal-board-first-108-bytes"),
    pytest.param("full", _PAGE, id="full-board-whole-first-page"),
    pytest.param("full", 108, id="full-board-first-108-bytes"),
]


@pytest.mark.parametrize(("kind", "length"), _EMPTY_FIRST_PAGE_REPLAYED)
def test_the_replay_reads_as_an_empty_database_that_passes_both_integrity_checks(kanban_home, kind, length):
    """Pins the premise of the next test: neither SQLite nor either integrity check sees a problem.

    Only the length of the file gives the replay away. If a newer SQLite starts to notice the
    pages past the size its header declares, the length check is no longer what protects these
    shapes, and this is the test that says so.
    """
    db = _board_with_live_sharer(kind, "/elsewhere/ws")
    used_bytes = db.stat().st_size
    assert used_bytes > _PAGE
    _replay_empty_first_page(db, length)
    assert db.stat().st_size == used_bytes
    with contextlib.closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as reader:
        assert reader.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone() is None
        assert reader.execute("PRAGMA page_count").fetchone()[0] == 1
        assert reader.execute("PRAGMA quick_check(1)").fetchall() == [("ok",)]
        assert reader.execute("PRAGMA integrity_check").fetchall() == [("ok",)]


@pytest.mark.parametrize(("kind", "length"), _EMPTY_FIRST_PAGE_REPLAYED)
def test_board_whose_old_empty_first_page_was_written_back_keeps_the_workspace_it_may_still_use(
    kanban_home, kind, length
):
    with kbc.connect_closing() as conn:
        tid, ws = _running_scratch_task(conn)
        db = _board_with_live_sharer(kind, ws)
        assert kbw._workspace_in_use_by_other(conn, tid, ws) == "shared"
        used_board = db.read_bytes()
        _replay_empty_first_page(db, length)
        assert kbw._workspace_in_use_by_other(conn, tid, ws) == "unknown"
        assert kb.complete_task(conn, tid, summary="done")
        _assert_outcome(conn, tid, ws, removed=False)
        # Nothing was lost on the board: put its first page back and the running task is still
        # there, still naming the directory that was kept.
        db.write_bytes(used_board)
        assert kbw._workspace_in_use_by_other(conn, tid, ws) == "shared"


@pytest.mark.parametrize(
    "write",
    [
        _write_incident_stub_with_a_page_after_it,
        _write_emptied_database_followed_by_a_page_its_header_does_not_count,
    ],
)
def test_a_page_the_header_does_not_count_is_invisible_to_sqlite(kanban_home, write):
    """Pins the premise of the cases that plant one: SQLite and both integrity checks see an
    intact empty database, so only the length of the file gives the extra page away.
    """
    db = _plant(write)
    with contextlib.closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as reader:
        assert reader.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone() is None
        assert reader.execute("PRAGMA quick_check(1)").fetchall() == [("ok",)]
        assert reader.execute("PRAGMA integrity_check").fetchall() == [("ok",)]


# The same boards with their first 108 or 512 bytes replaced by those of an empty database that had
# LARGER pages (the file header and the header of the empty schema leaf). The header now declares
# one page of 8, 32 or 64 KiB. The board is shorter than that, so SQLite reads its real pages as
# unused space inside the one big first page. The board is built with 512-byte pages, as one that
# was converted from larger pages would be.
_LARGER_PAGE_REPLAYED = [
    pytest.param("minimal", 108, 8192, id="minimal-board-108-bytes-from-8k-pages"),
    pytest.param("minimal", 512, 32768, id="minimal-board-512-bytes-from-32k-pages"),
    pytest.param("minimal", 108, 65536, id="minimal-board-108-bytes-from-64k-pages"),
    pytest.param("full", 108, 65536, id="full-board-108-bytes-from-64k-pages"),
    pytest.param("full", 512, 65536, id="full-board-512-bytes-from-64k-pages"),
]


@pytest.mark.parametrize(("kind", "length", "donor_page_size"), _LARGER_PAGE_REPLAYED)
def test_the_larger_page_replay_reads_as_an_empty_database_that_passes_both_integrity_checks(
    kanban_home, kind, length, donor_page_size
):
    """Pins the premise of the next test: neither SQLite nor either integrity check sees a problem.

    Only the length of the file gives the replay away. If a newer SQLite starts to notice a file
    shorter than the pages its header declares, the length check is no longer what protects
    these shapes, and this is the test that says so.
    """
    db = _board_with_live_sharer(kind, "/elsewhere/ws", page_size=512)
    used_bytes = db.stat().st_size
    assert 512 < used_bytes < donor_page_size
    _replay_empty_first_page(db, length, donor_page_size=donor_page_size)
    assert db.stat().st_size == used_bytes
    with contextlib.closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as reader:
        assert reader.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone() is None
        assert reader.execute("PRAGMA page_size").fetchone()[0] == donor_page_size
        assert reader.execute("PRAGMA page_count").fetchone()[0] == 1
        assert reader.execute("PRAGMA quick_check(1)").fetchall() == [("ok",)]
        assert reader.execute("PRAGMA integrity_check").fetchall() == [("ok",)]


@pytest.mark.parametrize(("kind", "length", "donor_page_size"), _LARGER_PAGE_REPLAYED)
def test_board_whose_first_page_came_back_from_larger_pages_keeps_the_workspace_it_may_still_use(
    kanban_home, kind, length, donor_page_size
):
    with kbc.connect_closing() as conn:
        tid, ws = _running_scratch_task(conn)
        db = _board_with_live_sharer(kind, ws, page_size=512)
        assert kbw._workspace_in_use_by_other(conn, tid, ws) == "shared"
        used_board = db.read_bytes()
        _replay_empty_first_page(db, length, donor_page_size=donor_page_size)
        assert kbw._workspace_in_use_by_other(conn, tid, ws) == "unknown"
        assert kb.complete_task(conn, tid, summary="done")
        _assert_outcome(conn, tid, ws, removed=False)
        # Nothing was lost on the board: put its first page back and the running task is still
        # there, still naming the directory that was kept.
        db.write_bytes(used_board)
        assert kbw._workspace_in_use_by_other(conn, tid, ws) == "shared"


def test_empty_sibling_that_this_process_holds_open_is_not_read_behind_its_back(kanban_home):
    """A bare ``open()`` and ``close()`` of a database file cancels the POSIX locks that every
    connection of this process holds on it (``sqlite_safe_read`` explains why). The length check
    reads the file's first bytes, so it goes through ``offline_file_access``: with a tracked
    connection to the sibling open, it refuses and the board counts as unreadable. Once the
    connection is gone the stub is skipped again.
    """
    db = _plant(_write_incident_stub)
    held = connect_tracked(db)
    try:
        with kbc.connect_closing() as conn:
            tid, ws = _finished_scratch_task(conn)
            assert kbw._workspace_in_use_by_other(conn, tid, ws) == "unknown"
    finally:
        held.close()
    with kbc.connect_closing() as conn:
        assert kbw._workspace_in_use_by_other(conn, tid, ws) is None


def test_board_whose_first_page_was_replayed_stays_unreadable_with_log_files_beside_it(kanban_home):
    """Log files next to the file do not vouch for it. A live board in WAL mode normally has a
    ``-wal`` and a ``-shm`` beside it, and these are what an unclean stop leaves behind. With
    nothing in the log, SQLite still reads page 1 from the main file, which is the replayed one.
    """
    db = _board_with_live_sharer("full", "/elsewhere/ws")
    _replay_empty_first_page(db)
    Path(str(db) + "-wal").write_bytes(b"")
    Path(str(db) + "-shm").write_bytes(b"")
    with kbc.connect_closing() as conn:
        tid, ws = _finished_scratch_task(conn)
        assert kbw._workspace_in_use_by_other(conn, tid, ws) == "unknown"


# ---------------------------------------------------------------------------
# The scan must leave SQLite's own locks alone
# ---------------------------------------------------------------------------

# Steps run in a separate process, so that SQLite's locks are seen from outside this one:
#   write      take the file exclusively and commit; a connection that is in the middle of a read keeps
#              this out in rollback-journal mode
#   share      the same, creating a ``tasks`` table with one running task on the workspace in argv[3]
#   read-lock  try to lock the byte range SQLite uses for readers. It is held while a read is in
#              progress, and in WAL mode for as long as a connection to the file is open.
_OTHER_PROCESS = textwrap.dedent(
    """
    import errno, fcntl, os, sqlite3, sys

    action, path = sys.argv[1], sys.argv[2]
    if action == "read-lock":
        fd = os.open(path, os.O_RDWR)
        try:
            fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB, 510, 0x40000002, os.SEEK_SET)
            print("free")
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EACCES):
                raise
            print("held")
        finally:
            os.close(fd)
    else:
        conn = sqlite3.connect(path, timeout=0.1, isolation_level=None)
        try:
            conn.execute("BEGIN EXCLUSIVE")
            if action == "share":
                conn.execute("CREATE TABLE tasks (id TEXT, status TEXT, workspace_path TEXT)")
                conn.execute("INSERT INTO tasks VALUES ('t_live', 'running', ?)", (sys.argv[3],))
            conn.execute("COMMIT")
            print("committed")
        except sqlite3.OperationalError as exc:
            print("refused: " + str(exc))
        finally:
            conn.close()
    """
)


def _other_process(*args: object) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _OTHER_PROCESS, *map(str, args)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _write_emptied_database_in_journal_mode(journal_mode: str):
    """A board file that was used and then emptied, in the given journal mode."""

    def write(path: Path) -> None:
        with contextlib.closing(sqlite3.connect(path, isolation_level=None)) as conn:
            assert conn.execute(f"PRAGMA journal_mode={journal_mode}").fetchone()[0].upper() == journal_mode
            conn.execute("CREATE TABLE old_data (x)")
            conn.execute("INSERT INTO old_data VALUES (zeroblob(65536))")
            conn.execute("DROP TABLE old_data")

    return write


@contextlib.contextmanager
def _scan_stopped_inside_its_integrity_check(monkeypatch: pytest.MonkeyPatch, run):
    """Run *run* on another thread and hold it inside ``PRAGMA quick_check`` of a sibling board.

    The stop is a SQLite progress handler on the connection that scan opens, so the scan's own code
    runs up to the integrity check and SQLite itself holds the read lock. The connection is
    changed where ``sqlite3.connect`` is called, which the scan reaches directly or through
    ``connect_tracked``, so the stop works whichever of the two opens it. Yields a dict that gets
    ``scan`` (what *run* returned) or ``error`` once the scan is let go again.
    """
    inside, resume = threading.Event(), threading.Event()
    outcome: dict[str, object] = {}
    scan_thread = "scan-stopped-inside-its-integrity-check"
    real_connect = sqlite3.connect

    class StoppingConnection(sqlite3.Connection):
        def execute(self, sql, *args):
            if sql == "PRAGMA quick_check(1)":
                stopped: list[bool] = []

                def stop():
                    if not stopped:
                        stopped.append(True)
                        inside.set()
                        resume.wait(30)
                    return 0

                self.set_progress_handler(stop, 1)
            return super().execute(sql, *args)

    def connect_for_the_stopped_scan(*args, **kwargs):
        if threading.current_thread().name == scan_thread:
            factory = kwargs.get("factory", sqlite3.Connection)
            # connect_tracked passes a factory of its own; keep it so the connection stays registered.
            kwargs["factory"] = (
                StoppingConnection
                if factory is sqlite3.Connection
                else type("StoppingTracked", (StoppingConnection, factory), {})
            )
        return real_connect(*args, **kwargs)

    def run_and_record():
        try:
            outcome["scan"] = run()
        except BaseException as exc:  # reported by the assertion in the caller
            outcome["error"] = repr(exc)

    monkeypatch.setattr(sqlite3, "connect", connect_for_the_stopped_scan)
    thread = threading.Thread(target=run_and_record, name=scan_thread)
    thread.start()
    try:
        deadline = time.monotonic() + 60
        while not inside.wait(0.05) and thread.is_alive() and time.monotonic() < deadline:
            pass
        assert inside.is_set(), f"the scan did not reach its integrity check: {outcome}"
        yield outcome
    finally:
        resume.set()
        thread.join(60)
    assert not thread.is_alive()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(
    "journal_mode",
    [pytest.param("DELETE", id="rollback-journal"), pytest.param("WAL", id="write-ahead-log")],
)
def test_a_second_scan_of_the_same_file_leaves_the_first_scans_read_lock_in_place(
    kanban_home, monkeypatch, journal_mode
):
    """Scan one is stopped inside its integrity check, holding SQLite's read lock on the sibling.
    Scan two reads the same file. The length check reads the header with a plain ``open()`` and
    ``close()``, and closing any descriptor of a file cancels every POSIX lock this process holds on
    it. When scan one's connection was not registered, scan two's read went ahead and cancelled the
    lock under scan one, so a writer in another process could commit in the middle of its read.
    Scan one's connection is registered now, so scan two's header read is refused instead and scan
    two counts the file as unreadable.
    """
    db = _plant(_write_emptied_database_in_journal_mode(journal_mode))
    with kbc.connect_closing() as conn:
        tid, ws = _finished_scratch_task(conn)

        def first_scan():
            with kbc.connect_closing() as other_thread_conn:
                return kbw._workspace_in_use_by_other(other_thread_conn, tid, ws)

        with _scan_stopped_inside_its_integrity_check(monkeypatch, first_scan) as first:
            assert _other_process("read-lock", db) == "held", (
                "scan one should hold SQLite's read lock inside its integrity check; "
                "its own header read must not have cancelled it"
            )
            second = kbw._workspace_in_use_by_other(conn, tid, ws)
            assert _other_process("read-lock", db) == "held", "scan two cancelled scan one's read lock"
            if journal_mode == "DELETE":
                assert _other_process("write", db).startswith("refused"), "a writer got in under scan one's read"
            assert second == "unknown"
        assert first == {"scan": None}
    assert not has_live_connection(db)


@pytest.mark.platforms("posix")
def test_completion_does_not_delete_a_workspace_that_a_sibling_commits_a_task_for_during_its_scan(
    kanban_home, monkeypatch
):
    """The harm the lock test guards against, end to end. Completion is stopped inside the integrity
    check of an empty sibling board and another scan of that board runs. A separate process then
    tries to give the sibling a running task on the completing task's workspace. The commit has to be
    refused while the check holds its read lock. If it were let through, completion would delete the
    directory with no deferral while the sibling kept the running row.
    """
    db = _plant(_write_emptied_database_in_journal_mode("DELETE"))
    with kbc.connect_closing() as conn:
        tid, ws = _running_scratch_task(conn)

        def complete():
            with kbc.connect_closing() as other_thread_conn:
                return kb.complete_task(other_thread_conn, tid, summary="done")

        with _scan_stopped_inside_its_integrity_check(monkeypatch, complete) as completion:
            second = kbw._workspace_in_use_by_other(conn, tid, ws)
            attempt = _other_process("share", db, ws)
            assert attempt.startswith("refused"), (attempt, second)
            assert second == "unknown"
        assert completion == {"scan": True}
        # Nothing used the directory, so completion finished the ordinary way.
        _assert_outcome(conn, tid, ws, removed=True)
    assert not has_live_connection(db)


def test_populated_sibling_that_this_process_keeps_open_still_reports_its_live_task(kanban_home):
    """The refusal of a header read matters only for a file with no tables. A process that keeps a
    board open (the gateway does) has it registered, so the guard refuses to read its header; the
    task rows are read the ordinary way and a live task there still protects the directory.
    """
    with kbc.connect_closing() as conn:
        tid, ws = _running_scratch_task(conn)
        db = _board_with_live_sharer("full", ws)
        held = connect_tracked(db)
        try:
            assert has_live_connection(db)
            assert kbw._workspace_in_use_by_other(conn, tid, ws) == "shared"
        finally:
            held.close()


@pytest.mark.parametrize(
    "failure",
    [PermissionError("header unreadable"), sqlite3.OperationalError("header read refused")],
    ids=["os-error", "sqlite-error"],
)
@pytest.mark.parametrize(
    ("kind", "expected"),
    [pytest.param("populated", "shared", id="board-with-tables"), pytest.param("empty", "unknown", id="no-tables")],
)
def test_a_header_that_cannot_be_read_matters_only_for_a_file_with_no_tables(
    kanban_home, monkeypatch, failure, kind, expected
):
    """The header is read before the scan knows whether the file has tables. A failed read is kept and
    held against the file only when the file turns out to hold none, because only then is the header
    all there is to vouch for it.
    """

    def cannot_read(db_file):
        raise failure

    monkeypatch.setattr(kbw, "_length_differs_from_declared_size", cannot_read)
    with kbc.connect_closing() as conn:
        tid, ws = _running_scratch_task(conn)
        if kind == "populated":
            _board_with_live_sharer("minimal", ws)
        else:
            _plant(_write_incident_stub)
        assert kbw._workspace_in_use_by_other(conn, tid, ws) == expected


# ---------------------------------------------------------------------------
# The real cleanup paths, with a sibling file planted where the incident had it
# ---------------------------------------------------------------------------

_SIBLINGS = [
    pytest.param(_write_incident_stub, True, id="no-tables-removed"),
    pytest.param(_write_garbage, False, id="unreadable-kept"),
    pytest.param(_write_board_with_schema_replaced_by_empty_leaf, False, id="schema-damaged-to-read-empty-kept"),
    pytest.param(_write_board_with_empty_first_page_replayed, False, id="empty-first-page-replayed-kept"),
    pytest.param(_write_small_board_with_larger_page_replayed, False, id="empty-first-page-from-larger-pages-kept"),
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
