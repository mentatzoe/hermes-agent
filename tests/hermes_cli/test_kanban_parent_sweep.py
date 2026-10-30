"""A finished parent's workspace goes when its last child finishes, even if that child still has an open child.

Regression for mentatzoe/aleph-vault#284, a gap in the deferred parent sweep of #33774.
``_cleanup_workspace`` returned at its active-children check before it swept the task's
parents. For a chain parent -> child -> grandchild:

1. The parent finishes while the child is open. It keeps its directory (the child may read it).
2. The child finishes while the grandchild is open. It keeps its own directory, but its parent's
   children are all finished now and nothing looked at the parent again.
3. The grandchild finishes. Its sweep removes the child's directory, one level up, never the parent's.

Everything here goes through real ``complete_task`` / ``archive_task`` calls on a throwaway home.
The table-driven test checks one rule after every step, against an oracle that reads only the
test's own bookkeeping: a task's directory exists while the task is unfinished or has an
unfinished child, and a directory several tasks share exists while any of them needs it.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import kanban_ops


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
    """A project repo whose history is fully pushed, so its worktrees count as holding no work."""
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


def _bind_scratch(conn, task_id: str) -> Path:
    """Create the scratch directory the dispatcher would create, and record it on the task."""
    workspace = Path(kbw.resolve_workspace(kb.get_task(conn, task_id)))
    kbw.set_workspace_path(conn, task_id, workspace)
    (workspace / "handoff.txt").write_text(task_id, encoding="utf-8")
    return workspace


# ---------------------------------------------------------------------------
# The chain from the issue, step by step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("middle", ["complete", "archive"])
def test_parent_child_grandchild_directories_after_each_step(kanban_home: Path, middle: str) -> None:
    """The issue's reproduction. Each tuple reads (parent, child, grandchild): True = directory exists."""
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child")
        grandchild = kb.create_task(conn, title="grandchild")
        kb.link_tasks(conn, parent, child)
        kb.link_tasks(conn, child, grandchild)
        dirs = [_bind_scratch(conn, t) for t in (parent, child, grandchild)]

        def exists() -> tuple[bool, bool, bool]:
            return (dirs[0].is_dir(), dirs[1].is_dir(), dirs[2].is_dir())

        assert exists() == (True, True, True)

        assert kb.complete_task(conn, parent, summary="parent done")
        assert exists() == (True, True, True), "the child is still open and may read the parent's files"

        if middle == "complete":
            assert kb.complete_task(conn, child, summary="child done")
        else:
            assert kb.archive_task(conn, child)
        assert exists() == (False, True, True), (
            "the parent is finished and so are all its children: its directory goes now; "
            "the child keeps its own until the grandchild finishes"
        )

        assert kb.complete_task(conn, grandchild, summary="grandchild done")
        assert exists() == (False, False, False)


def test_worktree_chain_follows_the_same_steps(kanban_home: Path, repo: Path) -> None:
    """Linked worktrees take the same path: the early return sat in front of both kinds."""
    with kbc.connect_closing() as conn:
        tasks = []
        for title in ("parent", "child", "grandchild"):
            tid = kb.create_task(conn, title=title, assignee="worker")
            target = repo / ".worktrees" / tid
            kbw._ensure_git_worktree(repo, target, f"wt/{tid}")
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET workspace_kind='worktree', workspace_path=?, branch_name=? WHERE id=?",
                    (str(target), f"wt/{tid}", tid),
                )
            tasks.append((tid, target))
        (parent, parent_wt), (child, child_wt), (grandchild, grandchild_wt) = tasks
        kb.link_tasks(conn, parent, child)
        kb.link_tasks(conn, child, grandchild)

        def exists() -> tuple[bool, bool, bool]:
            return (parent_wt.is_dir(), child_wt.is_dir(), grandchild_wt.is_dir())

        assert exists() == (True, True, True)
        assert kb.complete_task(conn, parent, summary="parent done")
        assert exists() == (True, True, True)
        assert kb.complete_task(conn, child, summary="child done")
        assert exists() == (False, True, True)
        assert kb.complete_task(conn, grandchild, summary="grandchild done")
        assert exists() == (False, False, False)


@pytest.mark.parametrize("parent_state", ["running", "blocked", "review"])
def test_unfinished_parent_keeps_its_directory_when_a_child_with_an_open_child_finishes(
    kanban_home: Path, parent_state: str
) -> None:
    """The sweep reaches a parent from a child that still holds its own directory. It must still
    leave a parent that has not finished: its worker's directory is in use."""
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child")
        grandchild = kb.create_task(conn, title="grandchild")
        kb.link_tasks(conn, parent, child)
        kb.link_tasks(conn, child, grandchild)
        assert kb.claim_task(conn, parent) is not None
        parent_ws = _bind_scratch(conn, parent)
        child_ws = _bind_scratch(conn, child)
        if parent_state == "blocked":
            assert kb.block_task(conn, parent, reason="waiting")
        elif parent_state == "review":
            assert kb.request_review(conn, parent, summary="please review", force=True)

        assert kb.archive_task(conn, child)

        assert kb.get_task(conn, parent).status == parent_state
        assert parent_ws.is_dir()
        assert child_ws.is_dir(), "the child still has an open child"


# ---------------------------------------------------------------------------
# One rule, every graph shape and completion order
# ---------------------------------------------------------------------------

GRAPHS = {
    "chain3": [("P", "C"), ("C", "G")],
    "chain4": [("P", "C"), ("C", "G"), ("G", "H")],
    "fork": [("P", "C1"), ("P", "C2"), ("C1", "G1")],
    "diamond": [("P", "C1"), ("P", "C2"), ("C1", "G"), ("C2", "G")],
}


def _names(edges):
    return list(dict.fromkeys(n for edge in edges for n in edge))


def _orders(edges):
    """Every order that completes each task after all its parents."""
    names = _names(edges)
    parents = {n: {p for p, c in edges if c == n} for n in names}
    for perm in itertools.permutations(names):
        position = {n: i for i, n in enumerate(perm)}
        if all(position[p] < position[n] for n in names for p in parents[n]):
            yield perm


def _sharing(edges, mode):
    """Which tasks use one directory. A decomposed child inherits its parent's path verbatim."""
    names = _names(edges)
    if mode == "none":
        return set()
    if mode == "pair":
        return set(edges[0])
    return set(names)


CASES = [
    pytest.param(edges, order, share, id=f"{graph}-{'.'.join(order)}-share_{share}")
    for graph, edges in GRAPHS.items()
    for order in _orders(edges)
    for share in ("none", "pair", "all")
]


@pytest.mark.parametrize(("edges", "order", "share"), CASES)
def test_a_directory_exists_exactly_while_something_needs_it(
    kanban_home: Path, edges, order, share
) -> None:
    """After every completion: a task's directory exists while the task is unfinished or has an
    unfinished child; a directory shared by several tasks exists while any of them needs it."""
    names = _names(edges)
    children = {n: [c for p, c in edges if p == n] for n in names}
    sharers = _sharing(edges, share)
    with kbc.connect_closing() as conn:
        ids = {n: kb.create_task(conn, title=n) for n in names}
        for parent, child in edges:
            kb.link_tasks(conn, ids[parent], ids[child])
        shared_dir = kb.workspaces_root() / "t_shared00000"
        dirs: dict[str, Path] = {}
        for n in names:
            if n in sharers:
                shared_dir.mkdir(parents=True, exist_ok=True)
                kbw.set_workspace_path(conn, ids[n], shared_dir)
                dirs[n] = shared_dir
            else:
                dirs[n] = _bind_scratch(conn, ids[n])

        finished: set[str] = set()

        def needed(n: str) -> bool:
            return n not in finished or any(c not in finished for c in children[n])

        def expected() -> dict[str, bool]:
            return {n: any(needed(m) for m in names if dirs[m] == dirs[n]) for n in names}

        def actual() -> dict[str, bool]:
            return {n: dirs[n].is_dir() for n in names}

        assert actual() == expected()
        for n in order:
            assert kb.complete_task(conn, ids[n], summary=f"{n} done")
            finished.add(n)
            assert actual() == expected(), f"after {n} finished (finished so far: {sorted(finished)})"


# ---------------------------------------------------------------------------
# A directory held for an open child is in use, for every caller of the scan
# ---------------------------------------------------------------------------


def test_gc_leaves_a_directory_a_finished_task_holds_for_its_open_child(kanban_home: Path) -> None:
    """An archived task sharing its directory with a finished task that has an open child.

    The finished task keeps the directory for that child, so ``kanban gc`` must not take it
    away. It goes when the child finishes."""
    with kbc.connect_closing() as conn:
        holder = kb.create_task(conn, title="holder")
        archived = kb.create_task(conn, title="archived sharer")
        child = kb.create_task(conn, title="holder's child")
        kb.link_tasks(conn, holder, child)
        shared = kb.workspaces_root() / "t_shared11111"
        shared.mkdir(parents=True)
        (shared / "handoff.txt").write_text("for the child", encoding="utf-8")
        for t in (holder, archived):
            kbw.set_workspace_path(conn, t, shared)
        child_ws = _bind_scratch(conn, child)

        assert kb.archive_task(conn, archived)
        assert kb.complete_task(conn, holder, summary="holder done")
        assert shared.is_dir()

        args = argparse.Namespace(event_retention_days=0, log_retention_days=0)
        assert kanban_ops._cmd_gc(args) == 0
        assert shared.is_dir(), "gc removed a directory the holder keeps for its open child"
        assert child_ws.is_dir()

        assert kb.complete_task(conn, child, summary="child done")
        assert not shared.exists()
        assert not child_ws.exists()


@pytest.mark.parametrize("holder_finishes", ["complete", "archive"])
def test_parent_sweep_leaves_a_directory_shared_with_a_task_that_holds_it_for_its_open_child(
    kanban_home: Path, holder_finishes: str
) -> None:
    """A decomposed child inherits its parent's workspace_path, so parent and child can name one
    directory. The child finishes while its own child is open and keeps the directory for it.
    The new sweep reaches the parent from there; it must not take the shared directory away.
    Either way the holder leaves the board (done or archived), it still holds the directory."""
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent")
        holder = kb.create_task(conn, title="holder (shares the parent's directory)")
        grandchild = kb.create_task(conn, title="holder's child")
        kb.link_tasks(conn, parent, holder)
        kb.link_tasks(conn, holder, grandchild)
        shared = kb.workspaces_root() / "t_shared22222"
        shared.mkdir(parents=True)
        (shared / "handoff.txt").write_text("for the grandchild", encoding="utf-8")
        for t in (parent, holder):
            kbw.set_workspace_path(conn, t, shared)
        grandchild_ws = _bind_scratch(conn, grandchild)

        assert kb.complete_task(conn, parent, summary="parent done")
        if holder_finishes == "complete":
            assert kb.complete_task(conn, holder, summary="holder done")
        else:
            assert kb.archive_task(conn, holder)
        assert shared.is_dir(), "the sweep removed a directory the holder keeps for its open child"
        assert grandchild_ws.is_dir()

        assert kb.complete_task(conn, grandchild, summary="grandchild done")
        assert not shared.exists()
        assert not grandchild_ws.exists()
