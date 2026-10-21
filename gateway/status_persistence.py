"""Persistence-time ownership checks for gateway runtime status."""

import os
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_cli.active_sessions import _FileLock


def persist_runtime_status(
    path: Path, payload: dict[str, Any], write_fn: Callable[[Path, dict[str, Any]], None]
) -> bool:
    """Serialize ownership validation through atomic replacement across writers.

    Lock a stable sibling, not the JSON inode that atomic replacement unlinks.
    The lock lives only on the background writer thread and is released by the
    OS if a writer dies; another process's queued write then checks the owner
    it actually finds, rather than the one it observed before waiting.
    """
    with _FileLock(path.with_suffix(path.suffix + ".lock")):
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
    if (status._pid_from_record(existing) != os.getpid()
            and status._live_pid_from_record(existing) is not None):
        return None
    # External writers seed fields such as desired_state. Preserve those,
    # while the gateway's canonical snapshot wins for the fields it owns.
    return {**existing, **payload}
