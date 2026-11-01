"""Managed checkout dependencies survive recursive cleanup (aleph-vault#284, R4).

The paired ordinary nonempty scratch control lives in test_kanban_parent_sweep.
These tests check history as well as checkout files: the older unpublished
version cannot be recovered from the surviving latest checkout alone.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as ws
from hermes_cli import kanban_ops
from tests.hermes_cli import test_kanban_parent_sweep as fixtures
from tests.hermes_cli.test_kanban_parent_sweep import (
    _bind_path,
    _bind_scratch,
    _git,
)

kanban_home = fixtures.kanban_home
repo = fixtures.repo


def _external_checkout(outer: Path, repo: Path, checkout: Path, branch: str, storage: str) -> Path:
    origin = repo.parent / "origin.git"
    if storage == "main":
        anchor = outer
        _git("init", str(anchor))
        _git("-C", str(anchor), "remote", "add", "origin", str(origin))
        _git("-C", str(anchor), "fetch", "origin")
        upstream = _git("-C", str(repo), "branch", "--show-current").strip()
        _git("-C", str(anchor), "checkout", "-b", "fixture", f"origin/{upstream}")
    elif storage == "bare":
        anchor = outer / "stores" / "object-store"
        _git("clone", "--bare", str(origin), str(anchor))
    elif storage == "separate":
        anchor = checkout.parent / "anchor"
        (outer / "stores").mkdir()
        _git("clone", "--separate-git-dir", str(outer / "stores" / "metadata"), str(origin), str(anchor))
    else:
        anchor = repo
    _git("-C", str(anchor), "config", "user.email", "fixture@example.invalid")
    _git("-C", str(anchor), "config", "user.name", "fixture")
    ws._ensure_git_worktree(anchor, checkout, branch)
    if storage == "alias":
        private = Path(_git("-C", str(checkout), "rev-parse", "--absolute-git-dir").strip())
        alias = outer / "git-store"
        alias.symlink_to(repo / ".git", target_is_directory=True)
        (checkout / ".git").write_text(f"gitdir: {alias / 'worktrees' / private.name}\n", encoding="utf-8")
    if storage == "traversal":
        private = Path(_git("-C", str(checkout), "rev-parse", "--absolute-git-dir").strip())
        prefix = outer / "cancelled-component"
        prefix.mkdir()
        locator = prefix / ".." / os.path.relpath(private, outer)
        (checkout / ".git").write_text(f"gitdir: {locator}\n", encoding="utf-8")
    if storage == "private":
        # Git supports an absolute commondir. Relocate only the per-checkout
        # directory under the removal root; the common object store stays out.
        old_private = Path(_git("-C", str(checkout), "rev-parse", "--absolute-git-dir").strip())
        common = _git("-C", str(checkout), "rev-parse", "--path-format=absolute", "--git-common-dir").strip()
        private = outer / "stores" / "per-checkout"
        private.parent.mkdir()
        shutil.move(str(old_private), str(private))
        (private / "commondir").write_text(common + "\n", encoding="utf-8")
        (checkout / ".git").write_text(f"gitdir: {private}\n", encoding="utf-8")
    return checkout


def _unpublished_versions(checkout: Path) -> tuple[str, str]:
    versions = []
    for text in ("earlier unpublished version\n", "latest unpublished version\n"):
        (checkout / "README.md").write_text(text, encoding="utf-8")
        _git("-C", str(checkout), "commit", "-am", text.strip())
        versions.append(_git("-C", str(checkout), "rev-parse", "HEAD").strip())
    return versions[0], versions[1]


def _assert_history(checkout: Path, versions: tuple[str, str]) -> None:
    assert (checkout / "README.md").read_text(encoding="utf-8") == "latest unpublished version\n"
    assert _git("-C", str(checkout), "rev-parse", "HEAD").strip() == versions[1]
    assert _git("-C", str(checkout), "show", f"{versions[0]}:README.md") == "earlier unpublished version\n"


@pytest.mark.parametrize("storage", ["main", "bare", "separate", "private", "alias", "traversal"])
@pytest.mark.parametrize("finish", ["complete", "archive"])
@pytest.mark.parametrize("hold_until_grandchild", [False, True])
def test_external_checkout_history_survives_all_cleanup_retries(
    kanban_home, repo, tmp_path, storage, finish, hold_until_grandchild
):
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent")
        outer = _bind_scratch(conn, parent)
        child = kb.create_task(conn, title="external checkout", parents=[parent], workspace_kind="worktree")
        checkout = _external_checkout(outer, repo, tmp_path / "checkout", f"wt/{child}", storage)
        ws.set_workspace_path(conn, child, checkout)
        task = kb.get_task(conn, child)
        assert task is not None
        assert ws.resolve_workspace(task) == checkout.resolve()
        versions = _unpublished_versions(checkout)
        grandchild = kb.create_task(conn, title="grandchild", parents=[child])
        grandchild_path = outer / "grandchild" if hold_until_grandchild else kb.workspaces_root() / grandchild
        _bind_path(conn, grandchild, grandchild_path)
        assert kb.complete_task(conn, parent, summary="done")
        _assert_history(checkout, versions)
        if finish == "archive":
            assert kb.archive_task(conn, child)
        else:
            assert kb.complete_task(conn, child, summary="done, grandchild open")
        _assert_history(checkout, versions)
        assert outer.is_dir()
        assert kb.complete_task(conn, grandchild, summary="done")
        # Git canonicalizes locator components away. An unverified locator makes the
        # dependency unknown, so even unrelated scratch cleanup is deferred.
        assert grandchild_path.exists() == (storage in {"alias", "traversal"})
        _assert_history(checkout, versions)

        # A fresh traversal and then direct archive/GC must honor the same
        # retained dependency, not just stop the first upward walk.
        sibling = kb.create_task(conn, title="later sibling", parents=[parent])
        _bind_scratch(conn, sibling)
        assert kb.complete_task(conn, sibling, summary="retry ancestor")
        _assert_history(checkout, versions)
        assert kb.archive_task(conn, parent)
        assert kanban_ops._cmd_gc(argparse.Namespace(event_retention_days=0, log_retention_days=0)) == 0
        _assert_history(checkout, versions)


@pytest.mark.parametrize("other_board", [False, True])
@pytest.mark.parametrize("holder_state", ["ready", "done", "archived"])
def test_external_permanent_checkout_protects_git_storage_on_either_board(
    kanban_home, repo, tmp_path, other_board, holder_state
):
    if other_board:
        kb.create_board("other")
    with contextlib.ExitStack() as stack:
        conn = stack.enter_context(kbc.connect_closing())
        other = stack.enter_context(kbc.connect_closing(board="other")) if other_board else conn
        parent = kb.create_task(conn, title="scratch owner")
        outer = _bind_scratch(conn, parent)
        checkout = _external_checkout(outer, repo, tmp_path / "checkout", "permanent", "bare")
        holder = kb.create_task(other, title="permanent", workspace_kind="dir", workspace_path=str(checkout))
        versions = _unpublished_versions(checkout)
        if holder_state == "done":
            assert kb.complete_task(other, holder, summary="done")
        elif holder_state == "archived":
            assert kb.archive_task(other, holder)
        assert kb.complete_task(conn, parent, summary="done")
        assert kb.archive_task(conn, parent)
        assert kanban_ops._cmd_gc(argparse.Namespace(event_retention_days=0, log_retention_days=0)) == 0
        _assert_history(checkout, versions)


@pytest.mark.parametrize("failure", ["timeout", "nonzero", "incomplete", "relative", "damaged-gitfile"])
def test_unknown_git_dependency_never_authorizes_recursive_removal(
    kanban_home, repo, tmp_path, monkeypatch, failure
):
    from hermes_cli import _subprocess_compat as compat

    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="scratch owner")
        outer = _bind_scratch(conn, parent)
        checkout = _external_checkout(outer, repo, tmp_path / "checkout", "retained", "main")
        kb.create_task(conn, title="retained checkout", workspace_kind="dir", workspace_path=str(checkout))
        real_probe = compat.bounded_probe_run

        def probe(argv, **kwargs):
            if "rev-parse" in argv and str(checkout) in argv:
                if failure == "timeout":
                    return None
                if failure == "nonzero":
                    return subprocess.CompletedProcess(argv, 128, "", "unreadable metadata")
                output = str(outer / ".git") + "\n" if failure == "incomplete" else ".git\n.git\n"
                return subprocess.CompletedProcess(argv, 0, output, "")
            return real_probe(argv, **kwargs)

        if failure == "damaged-gitfile":
            (checkout / ".git").write_text("not a gitfile\n", encoding="utf-8")
        else:
            monkeypatch.setattr(compat, "bounded_probe_run", probe)
        assert ws._workspace_in_use_by_other(conn, parent, outer) == "unknown"
        assert kb.complete_task(conn, parent, summary="done")
        assert kb.archive_task(conn, parent)
        assert kanban_ops._cmd_gc(argparse.Namespace(event_retention_days=0, log_retention_days=0)) == 0
        assert (outer / "handoff.txt").is_file()


def test_cleanup_releases_storage_after_native_git_teardown(kanban_home, repo, tmp_path):
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent")
        outer = _bind_scratch(conn, parent)
        child = kb.create_task(conn, title="child", parents=[parent], workspace_kind="worktree")
        checkout = _external_checkout(outer, repo, tmp_path / "checkout", f"wt/{child}", "main")
        ws.set_workspace_path(conn, child, checkout)
        grandchild = kb.create_task(conn, title="grandchild", parents=[child])
        _bind_scratch(conn, grandchild)
        assert kb.complete_task(conn, parent, summary="done")
        assert kb.complete_task(conn, child, summary="done")
        assert outer.exists() and checkout.exists()
        assert kb.complete_task(conn, grandchild, summary="done")
        assert not checkout.exists(), "normal non-force Git teardown must still run"
        assert not outer.exists(), "released Git storage is not a permanent hold"


def test_git_discovery_ignores_an_ambient_repository_override(kanban_home, repo, tmp_path, monkeypatch):
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="scratch owner")
        outer = _bind_scratch(conn, parent)
        checkout = _external_checkout(outer, repo, tmp_path / "checkout", "retained", "separate")
        kb.create_task(conn, title="permanent", workspace_kind="dir", workspace_path=str(checkout))
        for name, value in {"GIT_DIR": str(repo / ".git"), "GIT_COMMON_DIR": str(repo / ".git"),
                            "GIT_WORK_TREE": str(repo), "GIT_CEILING_DIRECTORIES": str(checkout.parent)}.items():
            monkeypatch.setenv(name, value)
        assert ws._workspace_in_use_by_other(conn, parent, outer) == "shared"


def test_unrelated_non_git_or_missing_workspace_does_not_hold_scratch(kanban_home, tmp_path):
    with kbc.connect_closing() as conn:
        owner = kb.create_task(conn, title="owner")
        outer = _bind_scratch(conn, owner)
        for name in ("plain", "missing"):
            path = tmp_path / name
            if name == "plain":
                path.mkdir()
                (path / "keep.txt").write_text("permanent", encoding="utf-8")
            kb.create_task(conn, title=name, workspace_kind="dir", workspace_path=str(path))
        assert kb.complete_task(conn, owner, summary="done")
        assert not outer.exists()
        assert (tmp_path / "plain" / "keep.txt").read_text(encoding="utf-8") == "permanent"


def test_removing_part_of_a_managed_git_store_is_also_refused(kanban_home, repo, tmp_path):
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="scratch owner")
        outer = _bind_scratch(conn, parent)
        checkout = _external_checkout(outer, repo, tmp_path / "checkout", "retained", "main")
        kb.create_task(conn, title="retained", workspace_kind="dir", workspace_path=str(checkout))
        versions = _unpublished_versions(checkout)
        assert kb.complete_task(conn, parent, summary="retained dependency remains")
        objects = outer / ".git" / "objects"
        owner = kb.create_task(conn, title="legacy path inside the store", workspace_path=str(objects))
        assert kb.complete_task(conn, owner, summary="done")
        assert objects.is_dir()
        _assert_history(checkout, versions)


@pytest.mark.parametrize("failure", ["unreadable-entry", "broken-git-link"])
def test_git_entry_that_cannot_be_read_is_not_absent(kanban_home, repo, tmp_path, monkeypatch, failure):
    with kbc.connect_closing() as conn:
        owner = kb.create_task(conn, title="scratch owner")
        outer = _bind_scratch(conn, owner)
        checkout = _external_checkout(outer, repo, tmp_path / "checkout", "retained", "main")
        kb.create_task(conn, title="retained", workspace_kind="dir", workspace_path=str(checkout))
        if failure == "broken-git-link":
            (checkout / ".git").unlink()
            (checkout / ".git").symlink_to(tmp_path / "missing-gitfile")
        else:
            original = Path.lstat

            def lstat(path, *args, **kwargs):
                if path == checkout / ".git":
                    raise PermissionError("Git entry is unreadable")
                return original(path, *args, **kwargs)

            monkeypatch.setattr(Path, "lstat", lstat)
        assert ws._workspace_in_use_by_other(conn, owner, outer) == "unknown"
        assert kb.complete_task(conn, owner, summary="done")
        assert (outer / "handoff.txt").is_file()
