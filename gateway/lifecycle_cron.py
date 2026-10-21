"""Explicit cron dispatch boundaries carried by the gateway lifecycle ledger.

No heartbeat inference: only a clean previous life contributes a closed window.
Canceled drains remove the open boundary; retained windows are immutable and bounded.
"""
from __future__ import annotations

import logging
import math
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)
_lock = threading.RLock()
MAX_WINDOWS = 3
_DISPATCH_FIELDS = ("dispatch_stopped_at", "dispatch_stopped_pid", "dispatch_stopped_homes")


def _epoch(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def startup_windows(previous: dict, started: float) -> list[dict]:
    """Keep closed records, append only a provably clean, ordered previous boundary."""
    retained = previous.get("restart_windows")
    windows = [dict(w) for w in retained if valid_window(w)] if isinstance(retained, list) else []
    stop, birth = previous.get("dispatch_stopped_at"), previous.get("start_time")
    if (previous.get("phase") == "exited" and _epoch(stop) and _epoch(birth)
            and previous.get("dispatch_stopped_pid") == previous.get("pid")
            and type(previous.get("pid")) is int
            and birth <= stop <= started):
        window = {"start": stop, "end": started, "pid": previous["pid"],
                  "homes": previous.get("dispatch_stopped_homes", [])}
        if valid_window(window):
            windows.append(window)
    return windows[-MAX_WINDOWS:]


def valid_window(window) -> bool:
    return (isinstance(window, dict) and _epoch(window.get("start"))
            and _epoch(window.get("end")) and window["start"] <= window["end"]
            and type(window.get("pid")) is int and window["pid"] > 0
            and isinstance(window.get("homes"), list)
            and bool(window["homes"]) and all(isinstance(h, str) for h in window["homes"]))


def stop_dispatch(runner, *, external=False, home: Path | None = None) -> None:
    """Set the actual dispatch gate and preserve its first boundary for this life."""
    from gateway import lifecycle_ledger as ledger

    setattr(runner, "_external_drain_active" if external else "_draining", True)
    try:
        _record_stop(home)
    except Exception:
        logger.warning("Could not record cron dispatch stop", exc_info=True)


def _record_stop(home) -> None:
    from gateway import lifecycle_ledger as ledger
    from cron.scheduler_ownership import ticked_homes

    with _lock:
        record = ledger._read_json(ledger.get_lifecycle_sentinel_path(home)) or {}
        if record.get("pid") != os.getpid() or record.get("phase") != "running":
            return
        if record.get("dispatch_stopped_at") is not None:
            return
        root = home if home is not None else ledger._process_hermes_home()
        homes = {str(Path(h).resolve()) for h in ticked_homes().values()}
        homes.add(str(root.resolve()))
        record.update(dispatch_stopped_at=ledger.time.time(), dispatch_stopped_pid=os.getpid(),
                      dispatch_stopped_homes=sorted(homes))
        ledger._write_sentinel(record, home)


def resume_dispatch(runner, *, home: Path | None = None) -> None:
    """A canceled external drain is not a restart window."""
    from gateway import lifecycle_ledger as ledger

    runner._external_drain_active = False
    if runner._draining:
        return
    try:
        _record_resume(home)
    except Exception:
        logger.warning("Could not clear canceled cron dispatch stop", exc_info=True)


def _record_resume(home) -> None:
    from gateway import lifecycle_ledger as ledger

    with _lock:
        record = ledger._read_json(ledger.get_lifecycle_sentinel_path(home)) or {}
        if record.get("pid") != os.getpid():
            return
        for key in _DISPATCH_FIELDS:
            record.pop(key, None)
        ledger._write_sentinel(record, home)
