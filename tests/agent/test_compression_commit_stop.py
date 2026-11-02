"""Hard stop waits for an admitted transcript commit, independent of prompt setup (#60).

Kept outside the oversized concurrent-fork suite so its fixture can own and join
both caller threads and the compression executor without growing that module.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

from agent import conversation_compression as cc
from hermes_state import SessionDB
from tests.agent.test_compression_concurrent_fork import _build_agent_with_db
from tests.agent.thread_sync_helpers import CheckedThread, join_checked


class _ObservedLock:
    """Observe a contended acquire, forwarding every operation to the real fence lock."""

    def __init__(self, lock, waiting):
        self.lock, self.waiting = lock, waiting

    def acquire(self, *args, **kwargs):
        return self.lock.acquire(*args, **kwargs)

    def release(self):
        self.lock.release()

    def __enter__(self):
        if not self.lock.acquire(blocking=False):
            self.waiting.set()
            self.lock.acquire()
        return self

    def __exit__(self, *_exc):
        self.lock.release()


def test_hard_stop_waits_for_commit_already_admitted(tmp_path, monkeypatch):
    """Stop must reach the held fence, and may return only after the real DB commit."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HARD_CANCEL_AFTER_COMMIT_ADMISSION"
    db.create_session(session_id, source="tui")
    agent = _build_agent_with_db(db, session_id)
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"
    # Prompt construction runs git/workspace probes. This test covers the commit
    # fence, not prompt contents; keep the real compression/SQLite path intact.
    monkeypatch.setattr(agent, "_build_system_prompt", MagicMock(return_value="sys"))
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    commit_started, allow_commit = threading.Event(), threading.Event()
    stop_waiting, stop_returned, committed = threading.Event(), threading.Event(), threading.Event()
    original_archive = db.archive_and_compact

    def blocked_archive(*args, **kwargs):
        commit_started.set()
        assert allow_commit.wait(timeout=5)
        result = original_archive(*args, **kwargs)
        committed.set()
        return result

    def hard_stop():
        agent.hard_interrupt("stop after commit admission")
        assert committed.is_set(), "hard stop returned before the admitted transcript commit"
        stop_returned.set()

    monkeypatch.setattr(db, "archive_and_compact", blocked_archive)
    agent.context_compressor.compress.return_value = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    agent.context_compressor.compress.side_effect = None
    compression = CheckedThread(
        lambda: agent._compress_context(messages, "sys", approx_tokens=120_000), name="commit-owner",
    )
    stop = CheckedThread(hard_stop, name="hard-stop-owner")
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="commit-test") as executor:
            monkeypatch.setattr(cc, "_get_compress_timeout_executor", lambda: executor)
            compression.start()
            try:
                assert commit_started.wait(timeout=2)
                fence = agent._active_compression_commit_fence
                assert fence.commit_in_flight
                fence._lock = _ObservedLock(fence._lock, stop_waiting)
                stop.start()
                assert stop_waiting.wait(timeout=2), "hard stop never waited on the admitted commit"
                assert not stop_returned.is_set()
            finally:
                allow_commit.set()
                join_checked(compression, stop)
        assert committed.is_set() and stop_returned.is_set()
        assert compression.value[0][0]["content"] == "[CONTEXT COMPACTION] summary"
        assert agent._hard_interrupt_requested.is_set()
        assert db.get_compression_lock_holder(session_id) is None
    finally:
        db.close()
