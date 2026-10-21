"""One epoch-seconds creation-time reading for psutil process identities."""
from __future__ import annotations

import sys


def process_create_time(proc) -> float:
    """Return the process's birth in epoch seconds, without macOS's boot-time adjustment.

    psutil's public macOS reading shifts after a clock step relative to its import-time
    boot time. The kernel's stored birth does not. Keep full precision here: callers
    that need centisecond fingerprints quantize separately. Errors propagate so each
    caller retains its own unavailable-process policy.
    """
    if sys.platform == "darwin":
        raw = getattr(getattr(proc, "_proc", None), "create_time", None)
        if raw is not None:
            try:
                return float(raw(monotonic=True))
            except TypeError:
                pass  # older psutil without this keyword keeps the public fallback
    return float(proc.create_time())
