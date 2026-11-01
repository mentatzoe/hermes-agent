"""Admission for Kanban's non-force native worktree removal, not a Git crawler.

Git removes the checkout AND its registered private slot, even on some failed
removals. Its final worktrees-container rmdir is empty-only, not a prune.
Only canonical W/A/K identities and the existing managed holder set are used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import stat


# Removal effects inspected in upstream v2.39.5 / Apple Git-154 and v2.53.0.
# Unknown executors retain the worktree until their effects are checked.
_REMOVE_VERSIONS = {
    "git version 2.39.5", "git version 2.39.5 (Apple Git-154)", "git version 2.53.0",
}
_LOCATOR_ENV = (
    "GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES", "GIT_SHALLOW_FILE", "GIT_NAMESPACE",
    "GIT_CONFIG",
)


@dataclass(frozen=True)
class WorktreeRemoval:
    checkout: Path
    private: Path
    common: Path
    env: dict[str, str] = field(compare=False, repr=False)

    @property
    def repo(self) -> Path:
        return self.common.parent

    def git(self, path: Path, *args: str, timeout: int = 10):
        from hermes_cli.kanban_db_workspace import _git
        return _git(path, *args, timeout=timeout, env=self.env)

    def output(self, path: Path, *args: str) -> str:
        result = self.git(path, *args)
        if result.returncode != 0:
            raise OSError(f"Cannot validate native worktree: {args}")
        return result.stdout.strip()

    def validate(self) -> None:
        """Bind the selected native registration to both of its deletion roots.

        Rechecked before a retry. Missing metadata after a partial failure is
        unknown, never a new empty holder set or a reason to use rmtree/force.
        """
        from hermes_cli import kanban_db_workspace as ws
        dirs = ws._workspace_git_dirs(str(self.checkout), env=self.env)
        if tuple(map(ws._path_key, dirs)) != tuple(map(ws._path_key, (
            self.private, self.common, self.private, self.common,
        ))):
            raise OSError("Worktree storage identity changed")
        if ws._path_key(self.checkout) != ws._path_key(ws._lexical_path(self.checkout)):
            raise OSError("Noncanonical checkout")
        if ws._path_key(self.checkout.resolve(strict=True)) != ws._path_key(self.checkout):
            raise OSError("Aliased checkout")
        if self.common.name != '.git' or self.private.parent != self.common / 'worktrees':
            raise OSError("Not a canonical linked-worktree slot")
        if not self.private.name or self.checkout == self.repo:
            raise OSError("Not a linked checkout")
        main_dirs = ws._workspace_git_dirs(str(self.repo), env=self.env)
        if len(main_dirs) != 4 or any(ws._path_key(p) != ws._path_key(self.common) for p in main_dirs):
            raise OSError("Main repository identity disagrees")
        if ws._path_key(self.output(self.checkout, 'rev-parse', '--show-toplevel')) != ws._path_key(self.checkout):
            raise OSError("Workspace is not the checkout root")
        backlink = ws._git_locator_target(self.private / 'gitdir')
        if ws._path_key(backlink) != ws._path_key(self.checkout / '.git'):
            raise OSError("Native slot backlink disagrees")
        entries = self.output(self.repo, 'worktree', 'list', '--porcelain', '-z').split('\0')
        registrations = [entry[9:] for entry in entries if entry.startswith('worktree ')]
        matches = [p for p in registrations
                   if ws._path_key(Path(p).resolve(strict=False)) == ws._path_key(self.checkout)]
        if len(matches) != 1 or ws._path_key(matches[0]) != ws._path_key(self.checkout):
            raise OSError("Missing or ambiguous native registration")
        storage = self.git(self.repo, 'config', '--get', 'extensions.refStorage')
        if storage.returncode != 1 and not (storage.returncode == 0 and storage.stdout.strip() == 'files'):
            raise OSError("Unknown native reference storage")

    def overlaps(self, stored: str) -> bool:
        """Whether a managed holder depends on W, A or the empty-only container.

        K as a shared common store (including private==common in main/bare
        holders) is compatible with deleting this one validated A. Nothing
        inside A, nor a different private/common ancestor of A, is exempt.
        """
        from hermes_cli import kanban_db_workspace as ws
        dirs = [ws._lexical_path(p) for p in ws._workspace_git_dirs(stored, env=self.env)]
        checkout = ws._lexical_path(Path(stored).expanduser().resolve(strict=False))
        if checkout.is_relative_to(self.checkout) or checkout.is_relative_to(self.private):
            return True
        if checkout == self.private.parent:  # Git may rmdir the empty container
            return True
        for directory in dirs:
            if directory.is_relative_to(self.checkout) or self.checkout.is_relative_to(directory):
                return True
            if directory == self.common:
                continue
            if directory.is_relative_to(self.private) or self.private.is_relative_to(directory):
                return True
        return False

    def optional_branch(self, task_id: str, branch_name: str | None) -> str | None:
        """Only the actual, ordinary auto branch; unknown redirections keep it.

        The caller separately admits optional edits to refs/reflogs/packed-refs
        and config against managed holders. No branch deletion follows failure.
        """
        from hermes_cli import kanban_db_workspace as ws
        branch = (branch_name or '').strip() or f'wt/{task_id}'
        if not branch.startswith('wt/'):
            return None
        actual = self.git(self.checkout, 'symbolic-ref', '--quiet', 'HEAD')
        if actual.returncode or actual.stdout.strip() != f'refs/heads/{branch}':
            return None
        # Validate only the native files this optional operation can mutate.
        # Ref aliases, non-files backends and redirected metadata are not followed.
        for relative in (f'refs/heads/{branch}', f'logs/refs/heads/{branch}', 'packed-refs', 'config'):
            path = self.common / relative
            if ws._path_key(path.resolve(strict=False)) != ws._path_key(path):
                return None
            try:
                if not stat.S_ISREG(path.lstat().st_mode):
                    return None
            except FileNotFoundError:
                continue
        loose = self.common / f'refs/heads/{branch}'
        if loose.exists() and loose.read_text(encoding='ascii').startswith('ref:'):
            return None
        return branch


def discover_removal(path: str) -> WorktreeRemoval:
    from hermes_cli import kanban_db_workspace as ws
    from hermes_cli._subprocess_compat import noninteractive_repo_git_env

    # Refuse ambiguous inherited selection rather than probe one repo and
    # mutate another. Capture one environment for admission, safety and remove.
    if any(name in os.environ for name in _LOCATOR_ENV):
        raise OSError("Ambient Git repository override")
    checkout = ws._lexical_path(Path(path).expanduser())
    env = noninteractive_repo_git_env(checkout)
    if env is None:
        raise OSError("Cannot establish native Git environment")
    env['GIT_DISCOVERY_ACROSS_FILESYSTEM'] = '1'
    dirs = ws._workspace_git_dirs(str(checkout), env=env)
    if len(dirs) != 4:
        raise OSError("Cannot establish linked-worktree storage")
    removal = WorktreeRemoval(checkout, ws._lexical_path(dirs[2]), ws._lexical_path(dirs[3]), env)
    if removal.output(checkout, '--version') not in _REMOVE_VERSIONS:
        raise OSError("Unreviewed native worktree removal version")
    removal.validate()
    return removal
