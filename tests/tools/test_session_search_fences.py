"""Recall must not interpret quoted compaction markers as handoffs."""
import pytest

from tools.session_search_tool import _strip_embedded_compaction


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_info_string_inside_fence_does_not_close_it(fence):
    text = (f"Example follows:\n{fence}\n{fence}python\n"
            f"[CONTEXT SUMMARY]: quoted example\nMy real words\n{fence}\nEnd of message")
    assert _strip_embedded_compaction(text) == text


@pytest.mark.parametrize("indent", ["    ", "\t"])
def test_indented_code_marker_is_not_a_handoff(indent):
    text = f"Example:\n\n{indent}[CONTEXT SUMMARY]: quoted\n\nKeep my words."
    assert _strip_embedded_compaction(text) == text


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_indented_run_cannot_close_fence(fence):
    text = f"{fence}\n    {fence}\n[CONTEXT SUMMARY]: quoted\n{fence}\nKeep my words."
    assert _strip_embedded_compaction(text) == text


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_indented_run_cannot_open_fence(fence):
    human = f"Example:\n\n    {fence}\n\nKeep my words."
    handoff = "[CONTEXT SUMMARY]: handoff"
    text = human + "\n" + handoff
    assert _strip_embedded_compaction(text) == human + f"\n[compaction handoff omitted: {len(handoff)} chars]"


def test_backtick_in_info_string_does_not_open_fence():
    human = "Example:\n```some`info\nKeep my words."
    handoff = "[CONTEXT SUMMARY]: handoff"
    assert _strip_embedded_compaction(human + "\n" + handoff) == (
        human + f"\n[compaction handoff omitted: {len(handoff)} chars]")


@pytest.mark.parametrize("fence", ["```", "~~~"])
@pytest.mark.parametrize("indent", ["", "   "])
def test_real_closing_fence_allows_following_handoff(fence, indent):
    human = f"Example:\n{indent}{fence}python\n[CONTEXT SUMMARY]: quoted\n{indent}{fence}   \t\nKeep my words."
    handoff = "[CONTEXT SUMMARY]: handoff"
    assert _strip_embedded_compaction(human + "\n" + handoff) == (
        human + f"\n[compaction handoff omitted: {len(handoff)} chars]")


@pytest.mark.parametrize("fence", ["```", "~~~with`info"])
def test_unclosed_fence_keeps_all_remaining_words(fence):
    text = f"{fence}\n[CONTEXT SUMMARY]: quoted\nKeep my words."
    assert _strip_embedded_compaction(text) == text


@pytest.mark.parametrize("fence,inner", [("````", "```"), ("~~~~", "~~~")])
def test_shorter_inner_fence_does_not_close_longer_outer_fence(fence, inner):
    text = (f"Example:\n{fence}\n{inner}\n[CONTEXT SUMMARY]: quoted\n"
            f"{inner}\n{fence}\nKeep this sentence.")
    assert _strip_embedded_compaction(text) == text
