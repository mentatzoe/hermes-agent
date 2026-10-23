"""Regression guard for the cascading-interrupt hang (PR #6600).

Original diagnosis and fix by Kristian Vastveit (@kristianvast) in PR #6600,
against the then-inline ``_interruptible_api_call`` /
``_interruptible_streaming_api_call`` methods in run_agent.py. Those methods
have since been extracted into ``agent/chat_completion_helpers.py``, so the
fix is reapplied there and these tests target the extracted functions.

The bug: when ``agent.interrupt()`` fires during an active LLM call, the main
poll loop force-closes the worker-local httpx client to stop token generation.
That raises a transport error (RemoteProtocolError) on the worker — the
EXPECTED consequence of our own close, not a network bug. The streaming retry
loop misclassified it as a transient connection error and retried, each doomed
retry stalling for the full stream-stale timeout (up to 300s). Because the
gateway caches AIAgent instances per session, the stale worker outlived the
turn and raced the next turn's request — the root of the multi-minute
cascading-interrupt hang.

The fix: a request-local ``_request_cancelled`` token set by the poll loop
right before the force-close. The worker's exception handler checks it and
exits cleanly (no retry, no fallback, no "reconnecting" status) instead of
treating the forced error as transient.
"""
import threading
import time
import types
from unittest.mock import MagicMock

import httpx
import pytest

from agent import chat_completion_helpers as cch




def _make_agent():
    """A MagicMock agent wired with just enough surface for the helpers."""
    agent = MagicMock()
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    agent.verbose_logging = False
    # _compute_non_stream_stale_timeout / streaming setup helpers return
    # benign values; the real call path is mocked per-test.
    agent._compute_non_stream_stale_timeout.return_value = 5.0
    return agent


def test_non_streaming_cancel_does_not_surface_network_error():
    """A force-close during a non-streaming call must raise InterruptedError,
    not the swallowed transport error."""
    agent = _make_agent()

    create_calls = {"n": 0}
    fake_client = MagicMock()

    def _create(**kwargs):
        create_calls["n"] += 1
        # Simulate the main thread firing an interrupt mid-call, then the
        # force-close raising a transport error on this worker.
        agent._interrupt_requested = True
        time.sleep(0.3)  # let the poll loop observe the interrupt + force-close
        raise httpx.RemoteProtocolError("peer closed connection")

    fake_client.chat.completions.create.side_effect = _create
    agent._create_request_openai_client.return_value = fake_client
    agent._close_request_openai_client = MagicMock()
    agent._abort_request_openai_client = MagicMock()

    t0 = time.time()
    with pytest.raises(InterruptedError):
        cch.interruptible_api_call(agent, {"model": "x", "messages": []})
    elapsed = time.time() - t0

    # The forced RemoteProtocolError must NOT surface as the raised error.
    assert create_calls["n"] == 1
    assert elapsed < 10.0, f"interrupt took {elapsed:.1f}s — should be near-instant (guarding the 30s+ hang)"






# ---------------------------------------------------------------------------
# #67142: direct-Anthropic stale/interrupt watchdog must abort the request-local
# client from the poll (stranger) thread and NEVER close/rebuild the shared
# _anthropic_client — closing it there released a live TLS FD that the kernel
# recycled into a SQLite handle, writing a TLS record over a DB header.
# ---------------------------------------------------------------------------


def _make_anthropic_agent():
    agent = _make_agent()
    agent.api_mode = "anthropic_messages"
    return agent


@pytest.mark.parametrize("completion", ["after_timeout", "during_abort_join"])
def test_anthropic_non_streaming_stale_aborts_request_client_not_shared(monkeypatch, completion):
    """Control whether the provider finishes before or after the abort join.

    A pending worker times out; a response received during the join still wins.
    Both paths must abort only the request socket and close it on its owner
    thread. Regression for mentatzoe/aleph-vault#262.
    """
    agent = _make_anthropic_agent()
    agent._compute_non_stream_stale_timeout.return_value = 0.05
    agent._codex_silent_hang_hint = MagicMock(return_value=None)
    poll_tid = threading.get_ident()
    started, release, closed = threading.Event(), threading.Event(), threading.Event()
    abort_tids, close_tids, workers = [], [], []
    response = object()
    request_client = MagicMock()
    agent._create_request_anthropic_client = MagicMock(return_value=request_client)

    def _abort(client, *, reason):
        abort_tids.append(threading.get_ident())

    def _close(client, *, reason):
        close_tids.append(threading.get_ident())
        closed.set()

    agent._abort_request_anthropic_client = MagicMock(side_effect=_abort)
    agent._close_request_anthropic_client = MagicMock(side_effect=_close)

    def _create(_api_kwargs, *, client):
        assert client is request_client
        started.set()
        # Only the test releases this call, never a sleep/deadline. The finally
        # below releases and joins the worker even if an assertion fails.
        release.wait()
        return response

    class ControlledWorker(threading.Thread):
        def join(self, timeout=None):
            # Make client registration precede stale detection even when the
            # scheduler delays starting the worker beyond the first poll.
            assert started.wait(10.0), "provider worker did not start"
            if timeout == 2.0 and completion == "during_abort_join":
                # Choose a schedule that can make the old sleep-based test fail:
                # a complete response arrives while the post-abort join runs.
                agent._abort_request_anthropic_client.assert_called_once_with(
                    request_client, reason="stale_call_kill"
                )
                release.set()
                assert closed.wait(10.0), "provider worker did not clean up"
            super().join(timeout)

    def _worker(*args, **kwargs):
        worker = ControlledWorker(*args, **kwargs)
        workers.append(worker)
        return worker

    # Replace only this module's threading reference, not threading.Thread
    # process-wide. Registry locks and thread identities remain real.
    monkeypatch.setattr(cch, "threading", types.SimpleNamespace(
        Thread=_worker, Lock=threading.Lock, get_ident=threading.get_ident,
    ))
    agent._anthropic_messages_create = MagicMock(side_effect=_create)

    try:
        if completion == "after_timeout":
            with pytest.raises(TimeoutError, match="Non-streaming API call timed out"):
                cch.interruptible_api_call(agent, {"model": "x", "messages": []})
            assert not release.is_set()
        else:
            assert cch.interruptible_api_call(agent, {"model": "x", "messages": []}) is response
    finally:
        release.set()
        for worker in workers:
            # Bypass the scheduling hook during unconditional teardown.
            threading.Thread.join(worker, timeout=10.0)
            assert not worker.is_alive(), "provider worker leaked from the test"

    agent._anthropic_messages_create.assert_called_once_with(
        {"model": "x", "messages": []}, client=request_client
    )
    agent._anthropic_client.close.assert_not_called()
    agent._rebuild_anthropic_client.assert_not_called()
    agent._abort_request_anthropic_client.assert_called_once_with(
        request_client, reason="stale_call_kill"
    )
    assert abort_tids == [poll_tid]
    assert len(workers) == 1
    agent._close_request_anthropic_client.assert_called_once_with(
        request_client, reason="request_complete"
    )
    assert closed.is_set()
    assert close_tids == [workers[0].ident]
    assert workers[0].ident != poll_tid


