"""Drain replies never become durable agent-answer obligations (north-star 22)."""
import asyncio
from unittest.mock import AsyncMock
import pytest
from gateway import delivery_ledger as dl
from gateway.platforms.base import EphemeralReply, SendResult
from tests.gateway.restart_test_helpers import RestartTestAdapter
from tests.gateway.test_shared_room_steer_identity import _busy_runner, _event, _zoe


@pytest.mark.asyncio
@pytest.mark.parametrize('busy', [False, True])
async def test_drain_reply_is_ephemeral_and_never_ledgered(tmp_path, monkeypatch, busy):
    runner, _, agent = _busy_runner(mode='queue')
    runner._draining = True
    runner._status_action_gerund = lambda: 'restarting'
    event = _event('hello', _zoe())
    if busy:
        runner._running_agents['k'] = agent
        reply = await runner._hm_handle_running_session_message(event, event.source, 'k')
    else:
        _, reply, _ = await runner._hm_dispatch_quick_and_plugin_commands(event, event.source, None)
    assert isinstance(reply, EphemeralReply)
    monkeypatch.setattr(dl, '_db_path', lambda: tmp_path / 'state.db')
    adapter = RestartTestAdapter()
    adapter._message_handler = AsyncMock(return_value=reply)
    adapter.send = AsyncMock(return_value=SendResult(success=False, error='403 Missing Access'))
    adapter._active_sessions['k'] = asyncio.Event()
    await adapter._process_message_background(event, 'k')
    with dl._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM delivery_obligations').fetchone()[0] == 0
