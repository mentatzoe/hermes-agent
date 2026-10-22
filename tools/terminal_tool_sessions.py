"""Delegated shell state over a shared backend, without sharing its snapshot."""

import contextlib
import threading

from tools.environments.base import BaseEnvironment

_lock = threading.RLock()
_sessions: dict[str, "DelegatedShellSession"] = {}


class DelegatedShellSession(BaseEnvironment):
    """Own shell artifacts and cwd; borrow only the backend's execution transport."""

    def __init__(self, backend, cwd):
        self.backend = backend
        self.is_local = backend.is_local
        self._stdin_mode = backend._stdin_mode
        self._profile_scoped_passthrough = backend._profile_scoped_passthrough
        self._sudo_nopasswd_probe_supported = backend._sudo_nopasswd_probe_supported
        super().__init__(cwd=cwd, timeout=backend.timeout, env=dict(backend.env))
        self.init_session()

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def get_temp_dir(self):
        return self.backend.get_temp_dir()

    def _quote_shell_path(self, path):
        return self.backend._quote_shell_path(path)

    def _snapshot_script_kwargs(self, cwd):
        result = super()._snapshot_script_kwargs(cwd)
        result["quoted_cwd"] = self.backend._quote_cwd_for_cd(cwd)
        return result

    def _additional_profile_scoped_passthrough_names(self):
        return self.backend._additional_profile_scoped_passthrough_names()

    def _before_execute(self):
        self.backend._before_execute()

    def _run_bash(self, cmd_string, **kwargs):
        return self.backend._run_bash(cmd_string, **kwargs)

    def _kill_process(self, proc):
        return self.backend._kill_process(proc)

    def _force_kill_process(self, proc):
        return self.backend._force_kill_process(proc)

    def _extract_cwd_from_output(self, result):
        super()._extract_cwd_from_output(result)
        if self.is_local:
            from tools.environments.local import _msys_to_windows_path
            self.cwd = _msys_to_windows_path(self.cwd)

    def cleanup(self):
        # Never close the borrowed connection/container. Use the transport directly:
        # sourcing either shell's snapshot here would change or resurrect its state.
        if self.__dict__.get("_closed") or not self.__dict__.get("_snapshot_path"):
            return
        self._closed = True
        paths = " ".join(self._quote_shell_path(p) for p in (self._snapshot_path, self._cwd_file))
        proc = self._run_bash(f"rm -f -- {paths}", timeout=10)
        self._wait_for_process(proc, timeout=10)
        self._snapshot_ready = False


def is_delegated_shell(task_id):
    from tools.terminal_tool import _container_alias_lock, _container_aliases
    with _container_alias_lock:
        return bool(task_id and task_id in _container_aliases)


def bind_shell_session(backend, task_id, cwd=None):
    """Resolve shell identity independently of the container's collapsed cache key."""
    if not is_delegated_shell(task_id):
        return backend
    from tools.terminal_tool import _qualify_task_key, get_session_cwd
    key = _qualify_task_key(task_id)
    with _lock:
        session = _sessions.get(key)
        if session is None or session.backend is not backend:
            if session is not None:
                with contextlib.suppress(Exception):
                    session.cleanup()
            session = DelegatedShellSession(backend, cwd or get_session_cwd(task_id) or backend.cwd)
            _sessions[key] = session
        return session


def release_shell_session(task_id):
    from tools.terminal_tool import _qualify_task_key
    with _lock:
        session = _sessions.pop(_qualify_task_key(task_id), None)
    if session is not None:
        session.cleanup()


def release_backend_sessions(backend):
    with _lock:
        sessions = [s for s in _sessions.values() if s.backend is backend]
        for key in [k for k, s in _sessions.items() if s.backend is backend]:
            del _sessions[key]
    for session in sessions:
        with contextlib.suppress(Exception):
            session.cleanup()
