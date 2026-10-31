"""The gateway preserves the human-typed subset beside its enriched model message."""
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig
from gateway.platforms.base import Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.typed_text import typed_text_for_event


def test_plain_event_text_is_the_typed_subset():
    event = SimpleNamespace(text="the words I wrote", internal=False,
                            _gateway_pending_stt_transcripts=[])
    assert typed_text_for_event(event) == "the words I wrote"


def test_internal_event_has_no_human_typed_text():
    event = SimpleNamespace(text="[Background process finished]", internal=True,
                            _gateway_pending_stt_transcripts=["not a person"])
    assert typed_text_for_event(event) == ""


def test_placeholder_is_not_typed_but_successful_stt_is():
    event = SimpleNamespace(
        text="(The user sent a message with no text content)", internal=False,
        _gateway_pending_stt_transcripts=["words heard from the recording"],
    )
    assert typed_text_for_event(event) == "words heard from the recording"


@pytest.fixture
def runner():
    r = GatewayRunner.__new__(GatewayRunner)
    r.config = GatewayConfig(group_sessions_per_user=False)
    r.adapters = {}
    r._model = "test-model"
    r._base_url = ""
    r._has_setup_skill = lambda: False
    return r


@pytest.mark.asyncio
async def test_preparation_keeps_typed_text_separate_from_host_framing(runner):
    source = SessionSource(platform=Platform.DISCORD, chat_id="room", chat_type="group", user_name="Alex")
    event = MessageEvent(
        text="answer this", source=source,
        channel_context="[Recent channel messages]\n[Robin] earlier question",
    )

    prepared = await runner._prepare_inbound_message_text(event=event, source=source, history=[])

    assert event._gateway_typed_user_content == "answer this"
    assert prepared.startswith("[Recent channel messages]")
    assert "[Alex] answer this" in prepared


def test_turn_runner_passes_typed_text_to_agent(runner, monkeypatch):
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    monkeypatch.setattr("tools.approval.register_gateway_notify", lambda *_: None)
    monkeypatch.setattr("tools.approval.unregister_gateway_notify", lambda *_: None)
    source = SessionSource(platform=Platform.DISCORD, chat_id="room", user_id="human-1", user_name="Alex")
    ctx = TurnContext(source=source, message="[Alex] answer this", session_id="s", session_key="k",
                      typed_user_message="answer this")
    turn = TurnRunner(runner, ctx)
    runner._consume_pending_native_image_paths = lambda *_: []

    class Agent:
        def run_conversation(self, _message, **kwargs):
            self.kwargs = kwargs
            return {"final_response": "done"}

    agent = Agent()
    turn._run_conversation_with_approval(agent, [], None, None, None)

    assert agent.kwargs["typed_user_message"] == "answer this"
    assert agent.kwargs["turn_author"] == {"id": "human-1", "name": "Alex", "is_bot": False}
