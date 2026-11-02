"""Kanban removes eligible workspaces, not their task branches (aleph-vault#284).

A holder may register a ref-parent directory during or after native removal.
Omitting optional branch deletion preserves it without a new admission race.
"""
from contextlib import ExitStack
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as ws
from tests.hermes_cli import test_kanban_native_removal_boundary as fixtures

kanban_home = fixtures.kanban_home
repo = fixtures.repo
git = fixtures.git
native_task = fixtures.native_task


def _observe_native_removal(monkeypatch, repo, path, private, *, retry, after_attempt):
    """Use a real Git lock refusal, then real non-force removal; never fake success."""
    real_git = ws._git
    attempts = []

    def remove(root, *args, **kwargs):
        if args[:2] != ('worktree', 'remove'):
            return real_git(root, *args, **kwargs)
        assert args == ('worktree', 'remove', str(path))
        if retry and not attempts:
            git(repo, 'worktree', 'lock', str(path))
            try:
                result = real_git(root, *args, **kwargs)
            finally:
                git(repo, 'worktree', 'unlock', str(path))
            assert result.returncode != 0
            assert path.is_dir() and private.is_dir()
        else:
            result = real_git(root, *args, **kwargs)
            assert result.returncode == 0
            assert not path.exists() and not private.exists()
        attempts.append(result.returncode)
        after_attempt(result.returncode)
        print({'native_attempt': len(attempts), 'exit': result.returncode,
               'stderr': result.stderr, 'checkout_exists': path.exists(),
               'private_exists': private.exists()}, flush=True)
        return result

    monkeypatch.setattr(ws, '_git', remove)
    return attempts


@pytest.mark.parametrize('timing', ['first-success', 'retry-between', 'retry-success'])
@pytest.mark.parametrize('board', ['current', 'sibling'])
def test_native_removal_keeps_late_registered_ref_parent(
    kanban_home, repo, tmp_path, monkeypatch, timing, board,
):
    if board == 'sibling':
        kb.create_board('retained')
    with ExitStack() as stack:
        conn = stack.enter_context(kbc.connect_closing())
        holder_conn = stack.enter_context(kbc.connect_closing(board='retained')) if board == 'sibling' else conn
        path = tmp_path / 'parent'
        task, private = native_task(conn, repo, path)
        branch = f'refs/heads/wt/{task}'
        tip = git(repo, 'rev-parse', branch)
        held = repo / '.git/refs/heads/wt'
        registered = []

        def register_after_native(exit_code):
            register_now = (exit_code != 0) if timing == 'retry-between' else (exit_code == 0)
            if register_now:
                assert held.is_dir()
                registered.append(kb.create_task(
                    holder_conn, title='retained ref directory', workspace_kind='dir',
                    workspace_path=str(held),
                ))

        retry = timing != 'first-success'
        attempts = _observe_native_removal(
            monkeypatch, repo, path, private, retry=retry, after_attempt=register_after_native,
        )
        assert kb.complete_task(conn, task, summary='done')
        print({'timing': timing, 'board': board, 'native_exits': attempts,
               'held_exists': held.is_dir()}, flush=True)
        assert attempts == ([128, 0] if retry else [0])
        assert len(registered) == 1
        assert not path.exists() and not private.exists()
        assert held.is_dir(), 'a registered permanent directory must survive workspace cleanup'
        assert git(repo, 'rev-parse', '--verify', branch) == tip
        holder = kb.get_task(holder_conn, registered[0])
        assert holder is not None
        assert holder.workspace_kind == 'dir' and holder.workspace_path == str(held)


@pytest.mark.parametrize('retry', [False, True])
def test_unheld_task_branch_survives_successful_native_removal(
    kanban_home, repo, tmp_path, monkeypatch, retry,
):
    with kbc.connect_closing() as conn:
        path = tmp_path / 'parent'
        task, private = native_task(conn, repo, path)
        branch = f'refs/heads/wt/{task}'
        tip = git(repo, 'rev-parse', branch)
        attempts = _observe_native_removal(
            monkeypatch, repo, path, private, retry=retry, after_attempt=lambda _code: None,
        )
        assert kb.complete_task(conn, task, summary='done')
        assert attempts == ([128, 0] if retry else [0])
        assert not path.exists() and not private.exists()
        assert git(repo, 'rev-parse', '--verify', branch) == tip


def test_same_task_rematerialization_reuses_retained_tip_after_main_advances(
    kanban_home, repo, tmp_path,
):
    with kbc.connect_closing() as conn:
        # A removed target must retain a repository anchor for public resolution.
        path = repo / '.worktrees' / 'retained-task'
        task, private = native_task(conn, repo, path)
        old_tip = git(path, 'rev-parse', 'HEAD')
        old_content = (path / 'README.md').read_text()
        assert kb.complete_task(conn, task, summary='done')
        assert not path.exists() and not private.exists()
        (repo / 'README.md').write_text('main advanced\n')
        git(repo, 'commit', '-am', 'advance main')
        git(repo, 'push', 'origin', 'HEAD')
        new_tip = git(repo, 'rev-parse', 'HEAD')
        assert new_tip != old_tip

        retained_task = kb.get_task(conn, task)
        assert retained_task is not None
        assert ws.resolve_workspace(retained_task) == path.resolve()
        assert Path(git(path, 'rev-parse', '--absolute-git-dir')).is_dir()
        assert git(path, 'rev-parse', 'HEAD') == old_tip
        assert (path / 'README.md').read_text() == old_content
        assert git(path, 'symbolic-ref', '--short', 'HEAD') == f'wt/{task}'
        # New task IDs still begin from the main checkout's current HEAD.
        new_path = tmp_path / 'new-task'
        new_task, _ = native_task(conn, repo, new_path)
        assert new_task != task
        assert git(new_path, 'rev-parse', 'HEAD') == new_tip
        assert (new_path / 'README.md').read_text() == 'main advanced\n'
        print({'same_task_tip': old_tip, 'new_task_tip': new_tip}, flush=True)
