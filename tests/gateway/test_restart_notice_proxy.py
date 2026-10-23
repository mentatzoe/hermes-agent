"""Proxy transport termination is distinct from successful notice completion."""

import json
import pytest

from gateway import restart_notifications as notices
from gateway.wake import deliver_wake
from tests.gateway.test_restart_internal_notifications import route
from tests.gateway.test_restart_notice_lifecycle import production_turns
from tests.gateway.test_kanban_wake_acceptance import drain
from tests.gateway.test_proxy_mode import _FakeSession, _FakeSSEResponse, _patch_aiohttp


@pytest.mark.asyncio
@pytest.mark.parametrize("status,finish,extras,done,completed", [
    (200, "stop", {}, True, True),
    (503, "stop", {}, True, False),
    (200, "error", {"error": {"message": "failed"}}, True, False),
    (200, "length", {"hermes": {"partial": True}}, True, False),
    (200, "stop", {"hermes": {"completed": False}}, True, False),
    (200, None, {}, True, False),
    (200, "stop", {}, False, False),
])
async def test_proxy_requires_successful_terminal_result(tmp_path, status, finish, extras, done, completed):
    old, adapter, source, key, sid, _ = route(tmp_path, restarting=False)

    async def pause(*args):
        pass

    async def worker(text):
        raise AssertionError("local executor must not run in proxy mode")

    production_turns(old, adapter, source, key, sid, pause, worker)
    old._get_proxy_url = lambda: "http://review-proxy.invalid"
    old._run_still_current_fn = lambda *args: lambda: True
    old._proxy_stream_consumer = lambda *args: None
    terminal = {"choices": [{"delta": {}, "finish_reason": finish}], **extras}
    chunks = ['data: {"choices":[{"delta":{"content":"done"}}]}\n',
              "data: " + json.dumps(terminal) + "\n"]
    if done:
        chunks.append("data: [DONE]\n")
    session = _FakeSession(_FakeSSEResponse(status=status, error_text="temporarily unavailable", sse_chunks=chunks))
    with _patch_aiohttp(session):
        await deliver_wake(adapter, text="proxy-owned notice", source=source)
        await drain(adapter)
    assert session.captured_json["messages"][-1]["content"] == "proxy-owned notice"
    old._draining = old._restart_requested = True
    await adapter.cancel_background_tasks()
    new, fresh, _, _, _, received = route(tmp_path, restarting=False)
    for _ in range(3):
        await notices.replay_pending(new)
        await drain(fresh)
    assert [row[0] for row in received] == ([] if completed else ["proxy-owned notice"])
