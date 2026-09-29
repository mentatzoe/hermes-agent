"""Machinery notices must never be filed as the person (local, Zoe 2026-09-29).

Shapes are real rows found filed under Zoe in Honcho; the speaker prefix
("[Zoe] ") comes from shared-channel attribution.
"""
import pytest

from plugins.memory.honcho import _is_internal_gateway_turn

MACHINERY = [
    "[ASYNC DELEGATION BATCH COMPLETE — deleg_994bc0e1]\nA background fan-out of 1 subagent(s) you dispatched earlier has finished.",
    "[Zoe] [ASYNC DELEGATION BATCH COMPLETE — deleg_994bc0e1]\nA background fan-out of 1 subagent(s)",
    "[Zoe] [ASYNC DELEGATION COMPLETE — deleg_1]\nA background subagent you dispatched earlier has finished.",
    "[IMPORTANT: Background process proc_6f943d67dd70 completed normally (exit code 0).\nCommand: x",
    "[Zoe] [IMPORTANT: Background process proc_209c7466fc3a exited (exit code 1).\nCommand: x",
    "[SYSTEM: Background process proc_1 matched watch pattern \"identity\".",
    "[System note: Your previous turn was interrupted before you could process the last tool result(s).",
    "[System note: The previous turn was interrupted by a gateway shutdown; the gateway is now back",
    "Internal safety-net handoff; no user-facing acknowledgement or machinery summary.",
    "[Zoe] Internal safety-net return for your existing Model roster commission.",
    "⏸ @eng Kanban t_c33c51f5 blocked: review-required: send_message wake",
    "[Zoe] ⏸ @pm Kanban t_abc123 blocked: Zoe final ready-to-ship review",
    "⏱ @sr-dev Kanban t_9f timed out (max_runtime=0s); will retry",
    "[Your active task list was preserved across context compression]",
]

HUMAN = [
    "[Zoe] I still got an error",
    "[Zoe] We changed the newspaper title... I think",
    "Why did I get an [ASYNC DELEGATION BATCH COMPLETE] message?",
    "The Kanban t_123 blocked notice keeps showing up, can you look?",
    "[CANONICAL SEED: USER.md]\nSubject peer: Zoe",
    "Some sessions are still in flight though, no?",
]


@pytest.mark.parametrize("text", MACHINERY)
def test_machinery_is_not_the_person(text):
    assert _is_internal_gateway_turn(text)


@pytest.mark.parametrize("text", HUMAN)
def test_human_words_are_kept(text):
    assert not _is_internal_gateway_turn(text)
