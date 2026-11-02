"""Optional provider-owned, read-only memory for a delegated child at birth.

Ordinary providers and the builtin writer stay absent. Providers opt in by
returning a distinct, already-bound reader from ``fork_read_only``; the child
owns that reader's lifecycle, never the parent's provider or pending writes.
"""
from __future__ import annotations

import logging

from agent.memory_manager import MemoryManager, inject_memory_provider_tools, memory_provider_tools_exposed

logger = logging.getLogger(__name__)


def inherit_read_only_memory(parent, child) -> None:
    """Attach only explicit read-only forks, before the child's first prompt."""
    source = getattr(parent, '_memory_manager', None)
    if not isinstance(source, MemoryManager) or not memory_provider_tools_exposed(parent):
        return
    reader = MemoryManager()
    reader.read_only = True
    try:
        for provider in source.providers:
            fork = getattr(provider, 'fork_read_only', None)
            if provider.name == 'builtin' or not callable(fork):
                continue
            candidate = fork(session_id=child.session_id)
            if candidate is not None and candidate is not provider:
                try:
                    reader.add_provider(candidate)
                except Exception:
                    if candidate not in reader.providers:
                        candidate.shutdown()
                    raise
        if not reader.providers:
            return
        child._memory_manager = reader
        inject_memory_provider_tools(child)
    except Exception:
        # No fallback to writable provider initialization or the parent's manager.
        reader.shutdown_all()
        child._memory_manager = None
        logger.warning('Read-only child memory was unavailable; no writable fallback was installed', exc_info=True)
