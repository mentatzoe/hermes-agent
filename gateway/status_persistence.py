"""Persistence-time ownership checks for gateway runtime status."""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_cli.active_sessions import _FileLock


@contextmanager
def runtime_status_lock(path: Path):
    """Stable mutex shared by runtime publishers and external status editors.

    POSIX flock needs no writable descriptor. A root-run migration can create
    this empty lock before the service user starts, so make new locks readable
    even with a restrictive umask; never chmod or replace an existing inode.
    Windows byte-range locking retains the existing writable-descriptor path.
    """
    lock = path.with_suffix(path.suffix + ".lock")
    if os.name == "nt":
        with _FileLock(lock):
            yield
        return

    import fcntl

    lock.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        fd = os.open(lock, os.O_RDONLY | os.O_CREAT | os.O_EXCL, 0o644)
        created = True
    except FileExistsError:
        fd = os.open(lock, os.O_RDONLY)
    try:
        if created:
            os.fchmod(fd, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def persist_runtime_status(
    path: Path, payload: dict[str, Any], write_fn: Callable[[Path, dict[str, Any]], None]
) -> bool:
    """Serialize ownership validation through atomic replacement across writers.

    Lock a stable sibling, not the JSON inode that atomic replacement unlinks.
    The lock lives only on the background writer thread and is released by the
    OS if a writer dies; another process's queued write then checks the owner
    it actually finds, rather than the one it observed before waiting.
    """
    with runtime_status_lock(path):
        merged = merge_runtime_status_on_disk(path, payload)
        if merged is None:
            return False
        write_fn(path, merged)
        return True


def merge_runtime_status_on_disk(
    path: Path, payload: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """Merge external fields, or refuse to replace another live owner's snapshot.

    Check on the writer thread, not while preparing an in-memory update: the
    queued snapshot can predate the current owner, and publishers must not do
    filesystem or process probes on the event loop.
    """
    from gateway import status

    existing = status._read_json_file(path)
    if not isinstance(existing, dict):
        return payload
    owner = status._live_pid_from_record(existing)
    if (owner is not None and owner != os.getpid()
            and status._record_matches_live_gateway_pid(existing, owner)):
        return None
    # Runtime fields belong to the gateway, but operator intent can change
    # after its canonical snapshot was cached. Never replay that stale intent.
    merged = {**existing, **payload}
    if "desired_state" in existing:
        merged["desired_state"] = existing["desired_state"]
    return merged
