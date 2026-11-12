from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.authored_turn_context import (
    AuthoredContextUnavailable, Source, block, capture, check_request,
    merge, render, selected_sections, snapshots,
)

PROFILE = (
    "### Why we have this\r\nMeaning.  \r\n\r\n"
    "### Me\r\nNot repeated.\r\n"
    "### Us\r\nA principle, with its qualification.\r\n\r\n"
)
NAMES = ("Why we have this", "Us")


@pytest.fixture
def agent(tmp_path):
    path = tmp_path / "authored.md"
    path.write_bytes(PROFILE.encode())
    return SimpleNamespace(
        _authored_turn_source=Source(path, NAMES),
        _authored_turn_expected={},
    )


def test_selection_preserves_whitespace():
    result = selected_sections(PROFILE, NAMES)
    assert "Meaning.  \r\n\r\n" in result
    assert result.endswith("qualification.\r\n\r\n")
    assert "Not repeated" not in result


@pytest.mark.parametrize("bad", [
    "### Why we have this\nOnly one section.",
    PROFILE + "### Us\nDuplicated.",
])
def test_missing_or_duplicate_is_not_partial_success(bad):
    with pytest.raises(AuthoredContextUnavailable):
        selected_sections(bad, NAMES)


@pytest.mark.parametrize("content", [
    "hi", "", "/skill anything",
    [{"type": "image_url", "image_url": {"url": "fixture"}}],
])
def test_every_admitted_shape_gets_complete_context(agent, content):
    row = {"role": "user", "content": deepcopy(content)}
    capture(agent, row)
    wire = render(row, content)
    check_request(agent, {"messages": [{"role": "user", "content": wire}]})
    assert render(row, wire) == wire
    assert row["content"] == content


def test_edit_changes_next_message_not_replay(agent):
    first = {"role": "user", "content": "first"}
    capture(agent, first)
    original = render(first, first["content"])
    agent._authored_turn_source.path.write_text(
        PROFILE.replace("Meaning.", "Changed meaning."), newline=""
    )
    second = {"role": "user", "content": "second"}
    capture(agent, second)
    assert render(first, first["content"]) == original
    assert "Changed meaning." in render(second, second["content"])


def test_merge_and_resume_keep_both_admissions(agent):
    first, second = (
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    )
    capture(agent, first)
    capture(agent, second)
    merge(first, second)
    resumed = deepcopy(first)
    wire = render(resumed, "first\n\nsecond")
    assert all(block(s) in wire for s in snapshots(resumed))
    check_request(agent, {"input": [{"role": "user", "content": wire}]})


def test_removed_context_blocks_inference(agent):
    row = {"role": "user", "content": "hello"}
    capture(agent, row)
    with pytest.raises(AuthoredContextUnavailable):
        check_request(agent, {"messages": [row]})


def test_unreadable_source_blocks_admission(agent):
    agent._authored_turn_source.path.unlink()
    with pytest.raises(AuthoredContextUnavailable):
        capture(agent, {"role": "user", "content": "hello"})
