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
        _authored_turn_source=Source.from_config({"file": str(path), "sections": list(NAMES)}, tmp_path),
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




def test_fallback_survives_restart_and_preserves_verification(agent):
    first = agent._authored_turn_source.snapshot()
    agent._authored_turn_source.path.unlink()
    source = Source.from_config({"file": str(agent._authored_turn_source.path),
                                "sections": list(NAMES)}, agent._authored_turn_source.home)
    fallback = source.snapshot()
    assert fallback["body"] == first["body"]
    assert fallback["sha256"] == first["sha256"]
    assert fallback["source_sha256"] == first["source_sha256"]
    assert fallback["verified_at"] == first["verified_at"]
    assert fallback["id"] != first["id"]
    assert fallback["degraded"] is True
    assert source.cache_path.stat().st_mode & 0o777 == 0o600
    assert source.cache_path.parent.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("bad", ["", "### Us\nPartial", "\xff"])
def test_invalid_current_uses_whole_verified_copy(agent, bad):
    source = agent._authored_turn_source
    first = source.snapshot()
    source.path.write_bytes(bad.encode("latin1"))
    fallback = source.snapshot()
    assert fallback["body"] == first["body"]
    assert fallback["degraded"]


@pytest.mark.parametrize("damage", ["hash", "binding", "date", "json"])
def test_corrupt_cache_never_supplies_text(agent, damage):
    import json
    source = agent._authored_turn_source
    source.snapshot()
    source.path.unlink()
    value = json.loads(source.cache_path.read_text())
    if damage == "hash": value["body"] += "changed"
    if damage == "binding": value["binding"]["home"] += "-other"
    if damage == "date": value["verified_at"] = "not a date"
    source.cache_path.write_text("{" if damage == "json" else json.dumps(value))
    with pytest.raises(AuthoredContextUnavailable):
        source.snapshot()


@pytest.mark.parametrize("change", ["home", "source", "sections"])
def test_cache_bound_to_exact_home_source_sections(agent, tmp_path, change):
    import shutil
    source = agent._authored_turn_source
    source.snapshot()
    source.path.unlink()
    home, path, names = source.home, source.path, list(NAMES)
    if change == "home": home = tmp_path / "other"
    if change == "source": path = tmp_path / "other.md"
    if change == "sections": names.reverse()
    other = Source.from_config({"file": str(path), "sections": names}, home)
    other.cache_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source.cache_path, other.cache_path)
    other.cache_path.chmod(0o600)
    with pytest.raises(AuthoredContextUnavailable): other.snapshot()


def test_disable_does_not_consult_fallback(agent):
    source = agent._authored_turn_source
    source.snapshot()
    source.path.unlink()
    assert Source.from_config(None, source.home) is None
    assert Source.from_config(False, source.home) is None
    assert Source.from_config({"enabled": False}, source.home) is None


def test_failure_episode_notice_and_recovery(agent):
    notices = []
    agent.status_callback = lambda event, text: notices.append(text)
    source = agent._authored_turn_source
    capture(agent, {"role": "user", "content": "first"})
    source.path.unlink()
    for _ in range(2): capture(agent, {"role": "user", "content": "next"})
    assert len(notices) == 1
    source.path.write_bytes(PROFILE.encode())
    row = {"role": "user", "content": "recovered"}
    capture(agent, row)
    assert snapshots(row)[0]["degraded"] is False
    source.path.unlink()
    capture(agent, {"role": "user", "content": "new episode"})
    assert len(notices) == 2


def test_current_source_survives_cache_write_failure(agent, monkeypatch):
    import agent.authored_turn_context as authored
    def failed(*a, **kw): raise OSError("disk unavailable")
    monkeypatch.setattr(authored.Source, "_write_verified", failed)
    notices = []
    agent.status_callback = lambda event, text: notices.append(text)
    row = {"role": "user", "content": "hello"}
    capture(agent, row)
    snapshot = snapshots(row)[0]
    assert snapshot["body"] == selected_sections(PROFILE, NAMES)
    assert snapshot["cache_write_failed"] and not snapshot["degraded"]
    assert len(notices) == 1




def test_empty_selected_section_is_invalid():
    with pytest.raises(AuthoredContextUnavailable):
        selected_sections("### Why we have this\n\n### Us\nReason.\n", NAMES)


def test_malformed_cache_shape_is_specific_unavailable(agent):
    source = agent._authored_turn_source
    source.snapshot()
    source.path.unlink()
    source.cache_path.write_text('[]')
    with pytest.raises(AuthoredContextUnavailable): source.snapshot()


def test_fallback_wire_exposes_original_verification(agent):
    source = agent._authored_turn_source
    first = source.snapshot()
    source.path.unlink()
    fallback = source.snapshot()
    wire = block(fallback)
    assert first["verified_at"] in wire
    assert first["source_sha256"] in wire
    assert "verified copy" in wire


def test_status_callback_failure_does_not_break_trustworthy_fallback(agent):
    source = agent._authored_turn_source
    source.snapshot()
    source.path.unlink()
    def broken(*a): raise RuntimeError("UI gone")
    agent.status_callback = broken
    row = {"role": "user", "content": "hi"}
    capture(agent, row)
    assert snapshots(row)[0]["degraded"]


def test_unreadable_source_blocks_admission(agent):
    agent._authored_turn_source.path.unlink()
    with pytest.raises(AuthoredContextUnavailable):
        capture(agent, {"role": "user", "content": "hello"})
