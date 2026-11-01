"""Native cleanup protects both recursive roots, without retaining safe siblings.

The nested-store case extends the preserved independent run403 reproducer.
Each lifecycle phase reads an earlier unpublished commit, not just HEAD/files.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as ws
from hermes_cli import kanban_ops
from tests.hermes_cli import test_kanban_parent_sweep as fixtures
from tests.hermes_cli.test_kanban_parent_sweep import _bind_scratch

kanban_home = fixtures.kanban_home
repo = fixtures.repo


def git(path, *args):
    result = subprocess.run(['git', '-C', str(path), *args], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, (args, result.stdout, result.stderr)
    return result.stdout.strip()


def native_task(conn, repo, path, *, parents=()):
    task = kb.create_task(conn, title='native fixture', parents=list(parents),
                          workspace_kind='worktree', workspace_path=str(path))
    ws._ensure_git_worktree(repo, path, f'wt/{task}')
    return task, Path(git(path, 'rev-parse', '--absolute-git-dir'))


def unpublished(path):
    commits = []
    for text in ['earlier unpublished\n', 'latest unpublished\n']:
        (path / 'README.md').write_text(text)
        git(path, 'commit', '-am', text.strip())
        commits.append(git(path, 'rev-parse', 'HEAD'))
    return commits


def assert_history(path, commits):
    assert (path / 'README.md').read_text() == 'latest unpublished\n'
    assert git(path, 'show', f'{commits[0]}:README.md') == 'earlier unpublished'
    assert git(path, 'rev-parse', 'HEAD') == commits[-1]


def gc():
    assert kanban_ops._cmd_gc(argparse.Namespace(event_retention_days=0, log_retention_days=0)) == 0


@pytest.mark.parametrize('finish', ['complete', 'archive'])
@pytest.mark.parametrize('storage', ['nested', 'sibling'])
def test_native_lifecycle_discriminator(kanban_home, repo, tmp_path, finish, storage):
    with kbc.connect_closing() as conn:
        parent_path = tmp_path / 'parent-worktree'
        parent, parent_private = native_task(conn, repo, parent_path)
        child_path = tmp_path / 'retained-child'
        if storage == 'nested':
            child_store = parent_private / 'retained-store'
            git(tmp_path, 'clone', '--bare', str(repo.parent / 'origin.git'), str(child_store))
            git(child_store, 'config', 'user.name', 'Boundary fixture')
            git(child_store, 'config', 'user.email', 'fixture@example.invalid')
        else:
            child_store = repo
        child, child_private = native_task(conn, child_store, child_path, parents=[parent])
        assert ws.resolve_workspace(kb.get_task(conn, child)) == child_path.resolve()
        grandchild = kb.create_task(conn, title='grandchild', parents=[child])
        grandchild_path = _bind_scratch(conn, grandchild)
        assert (grandchild_path / 'handoff.txt').is_file()
        commits = unpublished(child_path)
        # A real stale registration catches accidental broad prune, not just rmtree(W).
        stale_path = tmp_path / 'unrelated-stale'
        git(repo, 'worktree', 'add', '--detach', str(stale_path), 'HEAD')
        stale_slot = Path(git(stale_path, 'rev-parse', '--absolute-git-dir'))
        stale_head = (stale_slot / 'HEAD').read_bytes()
        shutil.rmtree(stale_path)
        assert git(parent_path, 'status', '--porcelain') == ''
        assert git(parent_path, 'log', '--oneline', 'HEAD', '--not', '--remotes') == ''

        def check(phase, parent_expected):
            print({'phase': phase, 'storage': storage, 'parent_exists': parent_path.exists(),
                   'private_exists': parent_private.exists()}, flush=True)
            assert_history(child_path, commits)
            assert parent_path.exists() == parent_expected
            assert parent_private.exists() == parent_expected
            assert child_private.is_dir()
            assert (stale_slot / 'HEAD').read_bytes() == stale_head

        assert kb.complete_task(conn, parent, summary='parent done')
        check('P complete', True)
        if finish == 'complete':
            assert kb.complete_task(conn, child, summary='C finished; G open')
        else:
            assert kb.archive_task(conn, child)
        check('C finished; G open', storage == 'nested')
        assert grandchild_path.is_dir()
        assert kb.complete_task(conn, grandchild, summary='G complete')
        assert not grandchild_path.exists()
        check('G complete', storage == 'nested')
        later = kb.create_task(conn, title='later retry', parents=[parent])
        _bind_scratch(conn, later)
        assert kb.complete_task(conn, later, summary='retry ancestor')
        check('later ancestor retry', storage == 'nested')
        assert kb.archive_task(conn, parent)
        gc()
        check('archive and GC', storage == 'nested')


@pytest.mark.parametrize('entry', ['complete', 'archive', 'gc'])
@pytest.mark.parametrize('role', ['checkout', 'private', 'common', 'empty-container'])
def test_native_effects_protect_registered_holders(kanban_home, repo, tmp_path, entry, role):
    with kbc.connect_closing() as conn:
        parent_path = tmp_path / 'parent'
        parent, private = native_task(conn, repo, parent_path)
        child_path = tmp_path / 'child'
        if role == 'common':
            store = private / 'foreign'
            git(tmp_path, 'clone', '--bare', str(repo.parent / 'origin.git'), str(store))
            git(store, 'config', 'user.email', 'fixture@example.invalid')
            git(store, 'config', 'user.name', 'Boundary fixture')
        else:
            store = repo
        if role == 'checkout':
            child_path = private / 'child'
        child, child_private = native_task(conn, store, child_path)
        if role == 'private':
            relocated = private / 'relocated-private'
            shutil.move(str(child_private), str(relocated))
            (relocated / 'commondir').write_text(str(repo / '.git') + '\n')
            (child_path / '.git').write_text(f'gitdir: {relocated}\n')
        if role == 'empty-container':
            kb.create_task(conn, title='permanent container', workspace_kind='dir',
                           workspace_path=str(private.parent))
        commits = unpublished(child_path)
        assert kb.complete_task(conn, child, summary='retained unpublished work')
        if entry == 'gc':
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET status='archived' WHERE id=?", (parent,))
            gc()
        elif entry == 'archive':
            assert kb.archive_task(conn, parent)
        else:
            assert kb.complete_task(conn, parent, summary='done')
        assert parent_path.is_dir() and private.is_dir()
        assert_history(child_path, commits)


@pytest.mark.parametrize('holder_kind', ['main', 'bare'])
def test_shared_common_role_allows_native_removal(kanban_home, repo, tmp_path, holder_kind):
    with kbc.connect_closing() as conn:
        parent_path = tmp_path / 'parent'
        parent, private = native_task(conn, repo, parent_path)
        holder = repo if holder_kind == 'main' else repo / '.git'
        kb.create_task(conn, title='common holder', workspace_kind='dir', workspace_path=str(holder))
        commits = unpublished(repo)
        assert kb.complete_task(conn, parent, summary='done')
        assert not parent_path.exists() and not private.exists()
        assert_history(repo, commits)


@pytest.mark.parametrize('fault', ['probe', 'version', 'env'])
def test_unknown_native_identity_keeps_both_roots(kanban_home, repo, tmp_path, monkeypatch, fault):
    from hermes_cli import _subprocess_compat as compat
    with kbc.connect_closing() as conn:
        path = tmp_path / 'parent'
        task, private = native_task(conn, repo, path)
        if fault == 'probe':
            real_probe = compat.bounded_probe_run

            def probe(args, **kwargs):
                if str(path) in args and 'rev-parse' in args:
                    return None
                return real_probe(args, **kwargs)

            monkeypatch.setattr(compat, 'bounded_probe_run', probe)
        elif fault == 'version':
            real_git = ws._git

            def different_version(root, *args, **kwargs):
                if args == ('--version',):
                    return subprocess.CompletedProcess(args, 0, 'git version 999.0\n', '')
                return real_git(root, *args, **kwargs)

            monkeypatch.setattr(ws, '_git', different_version)
        else:
            monkeypatch.setenv('GIT_DIR', str(repo / '.git'))
        assert kb.complete_task(conn, task, summary='done')
        assert kb.archive_task(conn, task)
        gc()
        assert path.is_dir() and private.is_dir()


@pytest.mark.parametrize('fault', ['backlink', 'alias', 'duplicate'])
def test_noncanonical_native_registration_keeps_both_roots(kanban_home, repo, tmp_path, fault):
    with kbc.connect_closing() as conn:
        path = tmp_path / 'parent'
        task, private = native_task(conn, repo, path)
        if fault == 'backlink':
            (private / 'gitdir').write_text(str(tmp_path / 'wrong/.git') + '\n')
        elif fault == 'alias':
            alias = tmp_path / 'private-alias'
            alias.symlink_to(private, target_is_directory=True)
            (path / '.git').write_text(f'gitdir: {alias}\n')
        else:
            alias = tmp_path / 'checkout-alias'
            alias.symlink_to(path, target_is_directory=True)
            duplicate = private.parent / 'duplicate'
            shutil.copytree(private, duplicate)
            (duplicate / 'gitdir').write_text(str(alias / '.git') + '\n')
        assert kb.complete_task(conn, task, summary='done')
        assert kb.archive_task(conn, task)
        gc()
        assert path.is_dir() and private.is_dir()


@pytest.mark.parametrize('fault', ['unknown-holder', 'new-nested-holder'])
def test_native_retry_rechecks_retained_dependencies(kanban_home, repo, tmp_path, monkeypatch, fault):
    with kbc.connect_closing() as conn:
        path = tmp_path / 'parent'
        task, private = native_task(conn, repo, path)
        real_git = ws._git
        attempts = []

        def failed_remove(root, *args, **kwargs):
            if args[:2] == ('worktree', 'remove'):
                attempts.append(args)
                holder = private / 'retained'
                holder.mkdir()
                (holder / 'handoff.txt').write_text('keep me')
                kb.create_task(conn, title='new holder', workspace_kind='dir', workspace_path=str(holder))
                if fault == 'unknown-holder':
                    (holder / '.git').write_text('unresolvable locator\n')
                return subprocess.CompletedProcess(args, 1, '', 'handle pending')
            return real_git(root, *args, **kwargs)

        monkeypatch.setattr(ws, '_git', failed_remove)
        assert kb.complete_task(conn, task, summary='done')
        assert len(attempts) == 1
        assert path.is_dir() and private.is_dir()
        assert (private / 'retained/handoff.txt').read_text() == 'keep me'


@pytest.mark.skipif(sys.platform == 'win32' or os.geteuid() == 0, reason='requires POSIX directory write denial')
@pytest.mark.parametrize('response', ['nonzero', 'timeout-after-effect'])
def test_partial_native_failure_does_not_escalate(kanban_home, repo, tmp_path, monkeypatch, response):
    with kbc.connect_closing() as conn:
        path = tmp_path / 'denied-parent'
        task, private = native_task(conn, repo, path)
        child_path = tmp_path / 'retained-child'
        native_task(conn, repo, child_path)
        commits = unpublished(child_path)
        stale = private.parent / 'stale-sentinel'
        stale.mkdir()
        (stale / 'gitdir').write_text(str(tmp_path / 'absent/.git'))
        (stale / 'sentinel').write_text('not pruned')
        real_git = ws._git
        attempts = []

        def denied_remove(root, *args, **kwargs):
            if args[:2] != ('worktree', 'remove'):
                return real_git(root, *args, **kwargs)
            assert args == ('worktree', 'remove', str(path))
            # The suite already runs inside a write-deny sandbox. POSIX modes
            # deny removal in this owned W without nesting/relaxing that sandbox.
            mode = path.stat().st_mode & 0o777
            path.chmod(0o500)
            try:
                result = real_git(root, *args, **kwargs)
            finally:
                path.chmod(mode)
            attempts.append(result.returncode)
            assert result.returncode != 0
            assert path.is_dir() and not private.exists(), result.stderr
            print({'native_exit': result.returncode, 'checkout_kept': path.is_dir(),
                   'private_removed': not private.exists(), 'stderr': result.stderr}, flush=True)
            if response == 'timeout-after-effect':
                raise subprocess.TimeoutExpired(args, 60)
            return result

        monkeypatch.setattr(ws, '_git', denied_remove)
        assert kb.complete_task(conn, task, summary='done')
        assert len(attempts) == 1
        assert path.is_dir() and not private.exists()
        assert kb.archive_task(conn, task)
        gc()
        assert len(attempts) == 1  # failed discovery is not another deletion opportunity
        assert path.is_dir()
        assert (stale / 'sentinel').read_text() == 'not pruned'
        assert_history(child_path, commits)
        assert ws._workspace_in_use_by_other(conn, 'other-task', tmp_path / 'unrelated') == 'unknown'


def test_gc_keeps_native_workspace_for_own_active_child(kanban_home, repo, tmp_path):
    with kbc.connect_closing() as conn:
        path = tmp_path / 'parent'
        task, private = native_task(conn, repo, path)
        kb.create_task(conn, title='open child', parents=[task])
        assert kb.archive_task(conn, task)
        gc()
        assert path.is_dir() and private.is_dir()


@pytest.mark.parametrize('branch_state', ['mismatched', 'detached', 'aliased-metadata'])
def test_uncertain_optional_branch_survives_native_removal(kanban_home, repo, tmp_path, branch_state):
    with kbc.connect_closing() as conn:
        path = tmp_path / 'parent'
        task, private = native_task(conn, repo, path)
        expected = f'wt/{task}'
        if branch_state == 'mismatched':
            other = 'wt/other'
            git(repo, 'branch', other)
            ws.set_branch_name(conn, task, other)
        elif branch_state == 'detached':
            git(path, 'checkout', '--detach')
        else:
            config = repo / '.git/config'
            target = tmp_path / 'external-config'
            config.rename(target)
            config.symlink_to(target)
        assert kb.complete_task(conn, task, summary='done')
        assert not path.exists() and not private.exists()
        assert git(repo, 'show-ref', '--verify', f'refs/heads/{expected}')
        if branch_state == 'mismatched':
            assert git(repo, 'show-ref', '--verify', 'refs/heads/wt/other')


def test_retry_keeps_new_unpublished_commits(kanban_home, repo, tmp_path, monkeypatch):
    with kbc.connect_closing() as conn:
        path = tmp_path / 'parent'
        task, private = native_task(conn, repo, path)
        real_git = ws._git
        attempts = []
        commits = []

        def failed_once(root, *args, **kwargs):
            if args[:2] == ('worktree', 'remove'):
                attempts.append(args)
                commits.extend(unpublished(path))
                return subprocess.CompletedProcess(args, 1, '', 'handle pending')
            return real_git(root, *args, **kwargs)

        monkeypatch.setattr(ws, '_git', failed_once)
        assert kb.complete_task(conn, task, summary='done')
        assert len(attempts) == 1
        assert path.is_dir() and private.is_dir()
        assert_history(path, commits)


def test_native_safety_and_execution_use_captured_environment(kanban_home, repo, tmp_path, monkeypatch):
    from hermes_cli import worktree_ops
    with kbc.connect_closing() as conn:
        path = tmp_path / 'parent'
        task, private = native_task(conn, repo, path)
        real_dirty = worktree_ops._worktree_is_dirty
        observed = []

        def changed_environment(*args, **kwargs):
            observed.append(kwargs['env'])
            monkeypatch.setenv('GIT_DIR', str(tmp_path / 'nonexistent-repository'))
            return real_dirty(*args, **kwargs)

        with monkeypatch.context() as patcher:
            patcher.setattr(worktree_ops, '_worktree_is_dirty', changed_environment)
            assert kb.complete_task(conn, task, summary='done')
        assert len(observed) == 1 and 'GIT_DIR' not in observed[0]
        assert not path.exists() and not private.exists()


@pytest.mark.parametrize('protection', ['locked', 'submodule'])
def test_native_nonforce_protections_remain(kanban_home, repo, tmp_path, protection):
    with kbc.connect_closing() as conn:
        path = tmp_path / 'parent'
        task, private = native_task(conn, repo, path)
        if protection == 'locked':
            git(repo, 'worktree', 'lock', str(path))
        else:
            git(path, '-c', 'protocol.file.allow=always', 'submodule', 'add', str(repo.parent / 'origin.git'), 'module')
            git(path, 'commit', '-am', 'published submodule')
            git(path, 'push', 'origin', 'HEAD')
            assert git(path, 'log', '--oneline', 'HEAD', '--not', '--remotes') == ''
        assert kb.complete_task(conn, task, summary='done')
        assert kb.archive_task(conn, task)
        gc()
        assert path.is_dir() and private.is_dir()
