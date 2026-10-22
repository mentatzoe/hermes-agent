"""Trusted observation waits, frozen while the gateway saves restart recovery.

Only native waits register here. A shell command or a process being observed is
not evidence that the observing turn is safe to interrupt.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import math
import os
from pathlib import Path
import stat
import threading
import time

from tools.interrupt import is_interrupted

_owner: ContextVar[tuple | None] = ContextVar("restart_wait_owner", default=None)
_creation_lock = threading.Lock()
_CONDITIONS = ("timer", "file_exists", "file_missing", "lock_free")


@dataclass
class _WaitState:
    changed: threading.Condition = field(default_factory=threading.Condition)
    tickets: dict = field(default_factory=dict)
    claim: "WaitClaim | None" = None
    yielded_turn: object = None


def _state(agent):
    with _creation_lock:
        state = vars(agent).get("_restart_wait_state")
        if state is None:
            state = agent._restart_wait_state = _WaitState()
        return state


@contextmanager
def bind_wait_owner(agent):
    bound = _owner.get()
    if bound is None or bound[0] is not agent:
        bound = (agent, getattr(agent, "_current_turn_id", None))
    token = _owner.set(bound)
    try:
        yield
    finally:
        _owner.reset(token)


def capture_wait_turn(fn):
    """Snapshot the turn BEFORE its workers and copied contexts are submitted."""
    @wraps(fn)
    def run(agent, *args, **kwargs):
        token = _owner.set((agent, getattr(agent, "_current_turn_id", None)))
        try:
            return fn(agent, *args, **kwargs)
        finally:
            _owner.reset(token)
    return run


def _bound_turn(agent):
    bound = _owner.get()
    return bound[1] if bound is not None and bound[0] is agent else getattr(agent, "_current_turn_id", None)


def was_yielded_for_restart(agent):
    state = vars(agent).get("_restart_wait_state") if hasattr(agent, "__dict__") else None
    return state is not None and state.yielded_turn is not None and state.yielded_turn == getattr(agent, "_current_turn_id", None)


def wait_for_dispatch(agent):
    """Fence queued tool workers until a restart claim commits or is abandoned."""
    state = _state(agent)
    turn = _bound_turn(agent)
    with state.changed:
        while state.claim is not None and state.claim.turn_id == turn:
            state.changed.wait(0.1)
        return getattr(agent, "_current_turn_id", None) == turn and (turn is None or state.yielded_turn != turn)


@dataclass
class WaitClaim:
    state: _WaitState
    turn_id: str
    task_ids: frozenset

    def release(self, *, interrupted=False):
        with self.state.changed:
            if self.state.claim is not self:
                return
            if interrupted:
                self.state.yielded_turn = self.turn_id
            self.state.claim = None
            self.state.changed.notify_all()


def claim_waits(agent):
    """Freeze a whole turn only when every executing tool worker is a native wait."""
    turn = getattr(agent, "_current_turn_id", None)
    tracker_lock = vars(agent).get("_tool_worker_threads_lock")
    if not turn or tracker_lock is None:
        return None
    state = _state(agent)
    with tracker_lock, state.changed:
        tids = set(agent._tool_worker_threads)
        if not tids or state.claim is not None or state.yielded_turn == turn:
            return None
        if not tids <= state.tickets.keys():
            return None
        tickets = [state.tickets[tid] for tid in tids]
        if any(ticket[0] != turn for ticket in tickets):
            return None
        state.claim = claim = WaitClaim(state, turn, frozenset(ticket[1] for ticket in tickets))
        return claim


def _lock_free(path):
    if os.name != "posix":
        raise ValueError("lock_free observation requires POSIX flock; use an exit receipt on this host")
    import fcntl
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except FileNotFoundError:
        return True
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("lock_free requires an existing regular lock file")
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    finally:
        os.close(fd)


def _ready(condition, path):
    if condition == "lock_free":
        return _lock_free(path)
    if condition == "file_exists":
        return path.is_file()
    if condition == "file_missing":
        return not path.exists()
    return False


def _wait_loop(state, turn, condition, path, timeout, agent):
    deadline = time.monotonic() + timeout
    while True:
        ready = _ready(condition, path)
        with state.changed:
            # A condition may resolve while persistence is in flight. Do not let
            # the tool return into another model/tool call until the claim ends.
            while state.claim is not None and state.claim.turn_id == turn:
                state.changed.wait(0.1)
            stale = agent is not None and getattr(agent, "_current_turn_id", None) != turn
            if stale or is_interrupted() or (turn is not None and state.yielded_turn == turn):
                return {"status": "interrupted", "condition": condition}
            if ready:
                return {"status": "ready", "condition": condition}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {"status": "ready" if condition == "timer" else "timeout", "condition": condition}
            state.changed.wait(min(0.1, remaining))


def wait_for_condition(condition, path=None, timeout=180, *, task_id=""):
    """Observe a timer/receipt/lock without running code or consuming the receipt."""
    try:
        if condition not in _CONDITIONS:
            raise ValueError("condition must be timer, file_exists, file_missing or lock_free")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number of seconds")
        if condition == "timer":
            if path is not None:
                raise ValueError("timer does not accept path")
        elif not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("path must be absolute for file and lock observations")
        path = Path(path) if path is not None else None
        bound = _owner.get()
        agent, turn = bound if bound is not None else (None, None)
        if agent is not None and getattr(agent, "_current_turn_id", None) != turn:
            return {"status": "interrupted", "condition": condition}
        state = _state(agent) if agent is not None else _WaitState()
        tid = threading.get_ident()
        with state.changed:
            state.tickets[tid] = (turn, str(task_id or ""))
        try:
            return _wait_loop(state, turn, condition, path, timeout, agent)
        finally:
            with state.changed:
                while state.claim is not None and state.claim.turn_id == turn:
                    state.changed.wait(0.1)
                state.tickets.pop(tid, None)
    except (OSError, ValueError) as exc:
        return {"status": "error", "error": str(exc)}


def handle_wait_for(args, task_id):
    if args.get("session_id") or args.get("data"):
        return {"status": "error", "error": "wait_for observes a condition, not a process or command"}
    if args.get("condition") in ("file_exists", "file_missing", "lock_free"):
        from tools.terminal_tool import _get_env_config
        if _get_env_config()["env_type"] != "local":
            return {"status": "error", "error": "File and lock wait_for require the local terminal backend; remote filesystem observation is unsupported."}
    return wait_for_condition(args.get("condition"), args.get("path"), args.get("timeout", 180), task_id=task_id)


def extend_process_schema(schema):
    schema["description"] += (
        " wait_for: read-only observation of a timer, receipt file or POSIX lock, without running shell code. "
        "File/lock conditions require the local terminal backend; timers work with any backend. "
        "Use it instead of shell sleep/poll loops. A gateway restart saves and interrupts a waiting-only "
        "turn for recovery; work it owns is not made restart-safe by waiting on it. Recheck the observed "
        "state after ready: readiness is a snapshot, not ownership or event acknowledgement."
    )
    props = schema["parameters"]["properties"]
    props["action"]["enum"].append("wait_for")
    props["session_id"]["description"] = "From terminal background output; any unique prefix works. Required except for list and wait_for."
    props["timeout"]["description"] = "Max seconds for wait or wait_for (default 180); timer waits for this duration."
    props["condition"] = {"type": "string", "enum": list(_CONDITIONS), "description": "Required for wait_for only."}
    props["path"] = {"type": "string", "description": "Absolute local-terminal receipt or lock path for wait_for; remote filesystem waits are unsupported. Omit for timer. Never creates or changes the file."}


def launch_block_reason():
    """Use the running host's in-memory flag, not a stale profile status file."""
    from gateway.restart import active_gateway_runner
    runner = active_gateway_runner()
    if runner is not None and getattr(runner, "_restart_requested", False) is True:
        return "Gateway restart requested; no new terminal or Python execution was started. Resume after restart."
    return None
