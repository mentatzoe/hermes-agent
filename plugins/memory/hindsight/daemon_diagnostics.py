"""Diagnostic wrapper around hindsight-api for embedded daemon launches.

This module is launched as ``python daemon_diagnostics.py`` by the Hermes
Hindsight memory plugin. It delegates to the installed ``hindsight_api.main``
entry point while adding structured shutdown diagnostics that explain graceful
shutdown windows where the Python PID is alive after uvicorn has closed the
LISTEN socket.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from typing import Any


def _safe_active_task_snapshot(memory: Any) -> dict[str, Any]:
    """Best-effort snapshot of active Hindsight worker tasks.

    Hindsight internals vary by release, so this intentionally uses duck typing.
    It returns a stable shape even when a release exposes no worker registry.
    """
    if memory is None:
        return {"count": 0, "stages": [], "source": "memory-uninitialized"}

    candidates = []
    for attr in ("worker", "_worker", "poller", "_poller", "worker_poller", "_worker_poller"):
        obj = getattr(memory, attr, None)
        if obj is not None:
            candidates.append(obj)
    candidates.append(memory)

    for obj in candidates:
        for attr in ("active_tasks", "_active_tasks", "in_flight", "_in_flight", "tasks", "_tasks"):
            value = getattr(obj, attr, None)
            if not value:
                continue
            if isinstance(value, dict):
                items = list(value.values())
            elif isinstance(value, (list, tuple, set)):
                items = list(value)
            else:
                continue
            stages = []
            for item in items:
                stage = getattr(item, "stage", None)
                if stage is None and isinstance(item, dict):
                    stage = item.get("stage") or item.get("type") or item.get("operation")
                if stage is not None:
                    stages.append(str(stage))
            return {"count": len(items), "stages": stages, "source": f"{type(obj).__name__}.{attr}"}

    return {"count": 0, "stages": [], "source": "not-exposed"}


def _emit(event: str, payload: dict[str, Any], *, level: int = logging.INFO) -> None:
    line = f"{event} {json.dumps(payload, sort_keys=True)}"
    logging.getLogger("hindsight_api.shutdown_diagnostics").log(level, line)
    log_path = os.environ.get("HINDSIGHT_API_DAEMON_LOG")
    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:
            pass


def main() -> None:
    import uvicorn
    import hindsight_api.main as api_main

    original_signal_handler = api_main._signal_handler
    original_uvicorn_run = uvicorn.run
    shutdown_started_at: float | None = None

    def diagnostic_signal_handler(signum, frame):
        nonlocal shutdown_started_at
        shutdown_started_at = time.monotonic()
        try:
            signal_name = signal.Signals(signum).name
        except Exception:
            signal_name = str(signum)
        _emit(
            "hindsight_api_shutdown_signal",
            {
                "pid": os.getpid(),
                "signal": signal_name,
                "active_tasks": _safe_active_task_snapshot(getattr(api_main, "_memory", None)),
                "phase": "before-signal-handler",
            },
        )
        return original_signal_handler(signum, frame)

    def diagnostic_uvicorn_run(*args, **kwargs):
        started_at = time.monotonic()
        _emit(
            "hindsight_api_uvicorn_start",
            {"pid": os.getpid(), "host": kwargs.get("host"), "port": kwargs.get("port")},
        )
        try:
            return original_uvicorn_run(*args, **kwargs)
        finally:
            elapsed_start = shutdown_started_at or started_at
            _emit(
                "hindsight_api_shutdown_complete",
                {
                    "pid": os.getpid(),
                    "active_tasks": _safe_active_task_snapshot(getattr(api_main, "_memory", None)),
                    "shutdown_elapsed_ms": int((time.monotonic() - elapsed_start) * 1000),
                    "phase": "after-uvicorn-run-returned",
                },
            )

    api_main._signal_handler = diagnostic_signal_handler
    uvicorn.run = diagnostic_uvicorn_run
    api_main.main()


if __name__ == "__main__":
    main()
