"""The identity header reads ~/.hermes files from the active Hermes home, and says when it can't.

Regression for the 10 Oct 2026 sandboxed trial: HOME was a scratch folder and HERMES_HOME
pointed elsewhere, so ``~/.hermes/memories/*.md`` expanded under HOME, the header came out
empty, and nothing warned.
"""
import logging
from pathlib import Path

import pytest

from agent import companion_compaction as cc
from agent.companion_compaction import CompanionConfig, IDENTITY_END, IDENTITY_HEADING

LOGGER = "agent.companion_compaction"


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """A served Hermes home and an unrelated HOME whose ~/.hermes holds different files."""
    hermes_home = tmp_path / "served-home"
    (hermes_home / "memories").mkdir(parents=True)
    (hermes_home / "memories" / "PEOPLE.md").write_text("Zoe uses they/them.\n")
    (hermes_home / "memories" / "SOUL-core.md").write_text("I am Aleph, served from the active home.\n")

    user_home = tmp_path / "scratch-user"
    (user_home / ".hermes" / "memories").mkdir(parents=True)
    (user_home / ".hermes" / "memories" / "PEOPLE.md").write_text("WRONG HOME: stale people file.\n")

    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(cc, "_warned", set(), raising=False)
    return hermes_home, user_home


def _body(header: str) -> str:
    assert header.startswith(IDENTITY_HEADING)
    assert header.rstrip().endswith(IDENTITY_END)
    return header[len(IDENTITY_HEADING):header.index(IDENTITY_END)].strip()


def test_tilde_hermes_resolves_through_active_home(homes):
    hermes_home, _ = homes
    cfg = CompanionConfig({"enabled": True, "identity_files": [
        "~/.hermes/memories/PEOPLE.md", "~/.hermes/memories/SOUL-core.md"]})
    body = _body(cfg.identity_header())
    assert "Zoe uses they/them." in body
    assert "served from the active home" in body
    assert "WRONG HOME" not in body


def test_tilde_hermes_resolves_when_home_has_no_hermes_dir(tmp_path, monkeypatch):
    hermes_home = tmp_path / "served"
    (hermes_home / "memories").mkdir(parents=True)
    (hermes_home / "memories" / "PEOPLE.md").write_text("Zoe uses they/them.\n")
    empty_user = tmp_path / "empty-user"
    empty_user.mkdir()
    monkeypatch.setenv("HOME", str(empty_user))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(cc, "_warned", set(), raising=False)
    cfg = CompanionConfig({"enabled": True, "identity_files": ["~/.hermes/memories/PEOPLE.md"]})
    assert "Zoe uses they/them." in _body(cfg.identity_header())


@pytest.mark.parametrize("spec", [
    "$HERMES_HOME/memories/PEOPLE.md",
    "${HERMES_HOME}/memories/PEOPLE.md",
    "~/.hermes/memories/PEOPLE.md#until:never-present",
])
def test_other_home_spellings_resolve_to_active_home(homes, spec):
    cfg = CompanionConfig({"enabled": True, "identity_files": [spec]})
    assert "Zoe uses they/them." in _body(cfg.identity_header())


def test_context_local_home_override_wins(homes, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    profile = tmp_path / "profile-home"
    (profile / "memories").mkdir(parents=True)
    (profile / "memories" / "PEOPLE.md").write_text("Profile people file.\n")
    cfg = CompanionConfig({"enabled": True, "identity_files": ["~/.hermes/memories/PEOPLE.md"]})
    token = set_hermes_home_override(profile)
    try:
        body = _body(cfg.identity_header())
    finally:
        reset_hermes_home_override(token)
    assert "Profile people file." in body
    assert "Zoe uses they/them." not in body


def test_other_paths_keep_their_expansion(homes, tmp_path, monkeypatch):
    _, user_home = homes
    (user_home / "notes.md").write_text("A file under HOME.\n")
    (user_home / ".hermes-other").mkdir()
    (user_home / ".hermes-other" / "x.md").write_text("Not the Hermes home.\n")
    absolute = tmp_path / "absolute.md"
    absolute.write_text("An absolute file.\n")
    monkeypatch.setenv("IDENTITY_DIR", str(tmp_path))
    assert cc._expand("~/notes.md") == user_home / "notes.md"
    assert cc._expand("~/.hermes-other/x.md") == user_home / ".hermes-other" / "x.md"
    assert cc._expand("$IDENTITY_DIR/absolute.md") == absolute
    assert cc._expand(str(absolute)) == absolute
    cfg = CompanionConfig({"enabled": True, "identity_files": [
        "~/notes.md", "~/.hermes-other/x.md", "$IDENTITY_DIR/absolute.md"]})
    body = _body(cfg.identity_header())
    for text in ("A file under HOME.", "Not the Hermes home.", "An absolute file."):
        assert text in body


def test_missing_file_warns_once_and_rest_of_header_is_built(homes, caplog):
    hermes_home, _ = homes
    cfg = CompanionConfig({"enabled": True, "people": ["Zoe (they/them)", "Aleph"],
                           "identity_files": ["~/.hermes/memories/PEOPLE.md",
                                              "~/.hermes/memories/SOUL-missing.md"]})
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        first = cfg.identity_header("Peer card: Zoe uses they/them.")
        second = cfg.identity_header("Peer card: Zoe uses they/them.")
    assert first == second
    body = _body(first)
    assert "- Zoe (they/them)" in body
    assert "Zoe uses they/them." in body
    assert "Peer card" in body
    missing = [r for r in caplog.records if "SOUL-missing.md" in r.getMessage()]
    assert len(missing) == 1
    assert missing[0].levelno == logging.WARNING
    assert str(hermes_home / "memories" / "SOUL-missing.md") in missing[0].getMessage()
    assert not [r for r in caplog.records if "header is empty" in r.getMessage()]


def test_unreadable_file_warns_and_does_not_raise(homes, caplog, monkeypatch):
    hermes_home, _ = homes
    target = hermes_home / "memories" / "PEOPLE.md"
    real_read = Path.read_text

    def read_text(self, *args, **kwargs):
        if self == target:
            raise PermissionError("fixture: unreadable")
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    cfg = CompanionConfig({"enabled": True, "identity_files": [
        "~/.hermes/memories/PEOPLE.md", "~/.hermes/memories/SOUL-core.md"]})
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        body = _body(cfg.identity_header())
    assert "served from the active home" in body
    assert any("could not be read" in r.getMessage() and "PEOPLE.md" in r.getMessage()
               for r in caplog.records)


def test_empty_header_warns_once(homes, caplog):
    cfg = CompanionConfig({"enabled": True, "identity_files": ["~/.hermes/memories/nope.md"]})
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        header = cfg.identity_header()
        cfg.identity_header("   ")
    assert _body(header) == ""
    empty = [r for r in caplog.records if "header is empty" in r.getMessage()]
    assert len(empty) == 1


def test_header_format_unchanged(homes):
    cfg = CompanionConfig({"enabled": True, "people": ["Zoe (they/them)"],
                           "identity_files": ["~/.hermes/memories/PEOPLE.md"]})
    assert cfg.identity_header("Card.") == (
        f"{IDENTITY_HEADING}\n\n- Zoe (they/them)\n\nZoe uses they/them.\n\nCard.\n{IDENTITY_END}\n\n")


def test_sibling_sessions_plugin_loads_from_active_home(homes):
    hermes_home, user_home = homes
    for root, label in ((hermes_home, "served"), (user_home / ".hermes", "wrong")):
        plugin = root / "plugins" / "sibling-sessions"
        plugin.mkdir(parents=True, exist_ok=True)
        (plugin / "__init__.py").write_text(
            "def _db_path():\n    return None\n"
            "def find_siblings(db, sid):\n    return [sid]\n"
            f"def render(sibs):\n    return 'other conversations from {label} home: ' + ','.join(sibs)\n")
    cfg = CompanionConfig({"enabled": True})
    cfg.session_id = "s-1"
    assert cfg.other_conversations() == "other conversations from served home: s-1"
    assert "from served home" in _body(cfg.identity_header())
