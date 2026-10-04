"""excludeChats: Honcho never saves turns from listed chats or their threads.

Zoe, 2026-09-30. In a shared Discord room each Hermes bot files the other
bots' posts under the user peer, so the room had to stop saving while every
other chat kept saving. Discord gives each thread its own chat id and reports
the channel separately as ``parent_chat_id``, so an entry for the channel must
also cover every existing and future thread under it. The live regression is
thread 1554661535073443841 under channel 1554601512842301531: an exact-chat
rule would have turned saving back on inside that thread.
"""

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import (
    EXCLUDE_ALL_CHATS,
    HonchoClientConfig,
    match_excluded_chat,
)

ROOM = "1554601512842301531"          # the excluded parent channel
ROOM_THREAD = "1554661535073443841"   # live thread under it (regression)
LATER_THREAD = "1554799999999999999"  # any thread created under it later
OTHER_CHANNEL = "1500000000000000001"
OTHER_THREAD = "1500000000000000002"
DM_CHANNEL = "1554000000000000009"

EXCLUDE = [f"discord:{ROOM}"]


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------

def _provider(*, save_messages=True, exclude_chats=None, chat=None) -> HonchoMemoryProvider:
    p = HonchoMemoryProvider()
    p._config = HonchoClientConfig(
        save_messages=save_messages,
        exclude_chats=list(exclude_chats if exclude_chats is not None else EXCLUDE),
    )
    p._chat_ref = dict(chat or {})
    p._manager = MagicMock()
    p._session_key = "test-session"
    p._session_initialized = True
    p._init_thread = None
    p._prefetch_thread = None
    p._sync_thread = None
    return p


def _room_message():
    return {"platform": "discord", "chat_id": ROOM}


def _thread_message(thread=ROOM_THREAD, parent=ROOM):
    return {
        "platform": "discord",
        "chat_id": thread,
        "thread_id": thread,
        "parent_chat_id": parent,
    }


def _sync(p):
    p.sync_turn("user says", "assistant says")
    if p._sync_thread is not None:
        p._sync_thread.join(timeout=5)


def _wait_for(mock_call, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline and not mock_call.called:
        time.sleep(0.02)


# ---------------------------------------------------------------------------
# Provider save gate: excluded parent, its threads, and unrelated chats
# ---------------------------------------------------------------------------

class TestSyncTurnGate:
    def test_message_in_excluded_parent_channel_is_not_saved(self):
        p = _provider(chat=_room_message())
        _sync(p)
        p._manager.get_or_create.assert_not_called()
        p._manager.save.assert_not_called()

    def test_message_in_child_thread_of_excluded_parent_is_not_saved(self):
        # The live thread: its own chat id differs from the excluded channel.
        p = _provider(chat=_thread_message(ROOM_THREAD))
        _sync(p)
        p._manager.get_or_create.assert_not_called()
        p._manager.save.assert_not_called()

    def test_thread_created_later_under_excluded_parent_is_not_saved(self):
        p = _provider(chat=_thread_message(LATER_THREAD))
        _sync(p)
        p._manager.save.assert_not_called()

    def test_unrelated_channel_still_saves(self):
        p = _provider(chat={"platform": "discord", "chat_id": OTHER_CHANNEL})
        _sync(p)
        p._manager.get_or_create.assert_called_once()
        p._manager.save.assert_called_once()

    def test_unrelated_thread_still_saves(self):
        p = _provider(chat=_thread_message(OTHER_THREAD, parent=OTHER_CHANNEL))
        _sync(p)
        p._manager.save.assert_called_once()

    def test_dm_still_saves(self):
        p = _provider(chat={"platform": "discord", "chat_id": DM_CHANNEL})
        _sync(p)
        p._manager.save.assert_called_once()

    def test_same_id_on_another_platform_still_saves(self):
        p = _provider(chat={"platform": "telegram", "chat_id": ROOM})
        _sync(p)
        p._manager.save.assert_called_once()

    def test_no_exclusions_saves_everywhere(self):
        p = _provider(exclude_chats=[], chat=_thread_message(ROOM_THREAD))
        _sync(p)
        p._manager.save.assert_called_once()

    def test_save_messages_false_still_wins_for_unexcluded_chat(self):
        p = _provider(save_messages=False, chat={"platform": "discord", "chat_id": OTHER_CHANNEL})
        _sync(p)
        p._manager.save.assert_not_called()


class TestMemoryToolMirrorGate:
    def test_excluded_thread_skips_conclusion_mirror(self):
        p = _provider(chat=_thread_message(ROOM_THREAD))
        p.on_memory_write("add", "user", "user likes coffee")
        time.sleep(0.1)
        p._manager.create_conclusion.assert_not_called()

    def test_unrelated_chat_still_mirrors(self):
        p = _provider(chat={"platform": "discord", "chat_id": OTHER_CHANNEL})
        p.on_memory_write("add", "user", "user likes coffee")
        _wait_for(p._manager.create_conclusion)
        p._manager.create_conclusion.assert_called_once()


class TestFlushAndShutdownGate:
    def test_excluded_thread_skips_session_end_flush(self):
        p = _provider(chat=_thread_message(ROOM_THREAD))
        p.on_session_end([])
        p._manager.flush_all.assert_not_called()

    def test_excluded_thread_shutdown_stops_writer_without_flushing(self):
        p = _provider(chat=_thread_message(ROOM_THREAD))
        p.shutdown()
        p._manager.shutdown.assert_not_called()
        p._manager.flush_all.assert_not_called()
        p._manager.stop_async_writer.assert_called_once()

    def test_unrelated_chat_flushes_at_session_end(self):
        p = _provider(chat={"platform": "discord", "chat_id": OTHER_CHANNEL})
        p.on_session_end([])
        p._manager.flush_all.assert_called_once()


class TestReadsStayOn:
    def test_excluded_thread_still_answers_profile_reads(self):
        p = _provider(chat=_thread_message(ROOM_THREAD))
        p._manager.get_peer_card.return_value = ["IDENTITY: Zoe"]
        out = json.loads(p.handle_tool_call("honcho_profile", {"peer": "user"}))
        assert out == {"result": ["IDENTITY: Zoe"]}

    def test_excluded_thread_still_gets_system_prompt_block(self):
        p = _provider(chat=_thread_message(ROOM_THREAD))
        assert "Honcho Memory" in p.system_prompt_block()


class TestMemoryFileMigrationGate:
    """A new, empty session in an excluded chat gets no memory-file upload."""

    def _init_session(self, chat):
        p = HonchoMemoryProvider()
        p._config = HonchoClientConfig(exclude_chats=list(EXCLUDE))
        p._chat_ref = dict(chat)
        cfg = MagicMock()
        cfg.context_tokens = None
        cfg.session_strategy = "global"
        cfg.resolve_session_name.return_value = "sess"
        manager = MagicMock()
        manager.get_or_create.return_value = SimpleNamespace(messages=[])
        with patch("plugins.memory.honcho.client.get_honcho_client", return_value=MagicMock()), \
             patch("plugins.memory.honcho.session.HonchoSessionManager", return_value=manager):
            p._do_session_init(cfg, "sess")
        return manager

    def test_excluded_thread_skips_migration(self):
        manager = self._init_session(_thread_message(ROOM_THREAD))
        manager.migrate_memory_files.assert_not_called()

    def test_unrelated_chat_still_migrates(self):
        manager = self._init_session({"platform": "discord", "chat_id": OTHER_CHANNEL})
        manager.migrate_memory_files.assert_called_once()


# ---------------------------------------------------------------------------
# initialize() captures the chat identity the gateway passes in
# ---------------------------------------------------------------------------

def _mock_cfg(exclude_chats):
    cfg = MagicMock()
    cfg.enabled = True
    cfg.api_key = "test-key"
    cfg.base_url = None
    cfg.recall_mode = "tools"
    cfg.init_on_session_start = False
    cfg.save_messages = True
    cfg.exclude_chats = list(exclude_chats)
    cfg.dialectic_depth = 1
    cfg.dialectic_depth_levels = None
    cfg.reasoning_level_cap = "low"
    cfg.resolve_session_name.return_value = "sess"
    return cfg


class TestInitializeCapturesChat:
    def _init(self, **kwargs):
        p = HonchoMemoryProvider()
        with patch(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            return_value=_mock_cfg(EXCLUDE),
        ):
            p.initialize(session_id="sess", **kwargs)
        return p

    def test_thread_under_excluded_parent_is_excluded_after_initialize(self):
        p = self._init(
            platform="discord",
            chat_id=ROOM_THREAD,
            thread_id=ROOM_THREAD,
            parent_chat_id=ROOM,
            agent_context="primary",
        )
        assert p._excluded_chat_reason() is not None
        assert p._saving_blocked() is True

    def test_unrelated_thread_is_allowed_after_initialize(self):
        p = self._init(
            platform="discord",
            chat_id=OTHER_THREAD,
            thread_id=OTHER_THREAD,
            parent_chat_id=OTHER_CHANNEL,
            agent_context="primary",
        )
        assert p._excluded_chat_reason() is None
        assert p._saving_blocked() is False

    def test_thread_typed_session_without_ids_is_excluded_after_initialize(self):
        p = self._init(
            platform="discord",
            chat_id=ROOM_THREAD,
            chat_type="thread",
            agent_context="primary",
        )
        assert p._saving_blocked() is True

    def test_cli_session_is_allowed(self):
        p = self._init(platform="cli", agent_context="primary")
        assert p._saving_blocked() is False


# ---------------------------------------------------------------------------
# Matcher and config parsing
# ---------------------------------------------------------------------------

class TestMatcher:
    def test_parent_match_names_the_rule(self):
        reason = match_excluded_chat(
            EXCLUDE, platform="discord", chat_id=ROOM_THREAD,
            thread_id=ROOM_THREAD, parent_chat_id=ROOM,
        )
        assert reason and ROOM in reason

    def test_thread_without_parent_fails_closed_when_platform_has_entries(self):
        # A Discord thread reached without its parent cannot be cleared.
        assert match_excluded_chat(
            EXCLUDE, platform="discord", chat_id=ROOM_THREAD, thread_id=ROOM_THREAD,
        )

    def test_thread_typed_chat_without_thread_or_parent_fails_closed(self):
        # Some synthetic sources carry chat_type="thread" but drop thread_id.
        assert match_excluded_chat(
            EXCLUDE, platform="discord", chat_id=ROOM_THREAD, chat_type="thread",
        )

    def test_plain_channel_without_parent_is_not_suspect(self):
        assert match_excluded_chat(
            EXCLUDE, platform="discord", chat_id=OTHER_CHANNEL, chat_type="group",
        ) is None

    def test_thread_without_parent_is_allowed_when_no_entry_for_platform(self):
        assert match_excluded_chat(
            EXCLUDE, platform="telegram", chat_id="1", thread_id="2",
        ) is None

    def test_thread_under_listed_parent_on_chat_id_platforms(self):
        # Slack/Telegram style: chat_id is the parent, thread_id the thread.
        assert match_excluded_chat(
            ["slack:C123"], platform="slack", chat_id="C123", thread_id="1700.1",
        )

    def test_bare_id_matches_any_platform(self):
        assert match_excluded_chat([ROOM], platform="discord", chat_id=ROOM)
        assert match_excluded_chat([ROOM], platform="telegram", chat_id=ROOM)

    def test_integer_ids_compare_as_strings(self):
        assert match_excluded_chat(
            EXCLUDE, platform="Discord", chat_id=int(ROOM_THREAD),
            thread_id=int(ROOM_THREAD), parent_chat_id=int(ROOM),
        )

    def test_malformed_sentinel_excludes_everything(self):
        assert match_excluded_chat(
            [EXCLUDE_ALL_CHATS], platform="discord", chat_id=OTHER_CHANNEL,
        )


def _write_config(tmp_path, data):
    path = tmp_path / "honcho.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestConfigParsing:
    def test_host_entry_is_read(self, tmp_path):
        path = _write_config(tmp_path, {
            "hosts": {"hermes": {"enabled": True, "excludeChats": [f"discord:{ROOM}"]}},
        })
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=path)
        assert cfg.exclude_chats == [f"discord:{ROOM}"]

    def test_root_entry_is_inherited(self, tmp_path):
        path = _write_config(tmp_path, {
            "excludeChats": [f"discord:{ROOM}"],
            "hosts": {"hermes": {"enabled": True}},
        })
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=path)
        assert cfg.exclude_chats == [f"discord:{ROOM}"]

    def test_host_list_replaces_root_list(self, tmp_path):
        path = _write_config(tmp_path, {
            "excludeChats": ["discord:1"],
            "hosts": {"hermes": {"enabled": True, "excludeChats": []}},
        })
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=path)
        assert cfg.exclude_chats == []

    def test_entries_are_normalized(self, tmp_path):
        path = _write_config(tmp_path, {
            "hosts": {"hermes": {"enabled": True, "excludeChats": [
                f" Discord:{ROOM} ", int(OTHER_CHANNEL), "", None, f"discord:{ROOM}",
            ]}},
        })
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=path)
        assert cfg.exclude_chats == [f"discord:{ROOM}", OTHER_CHANNEL]

    def test_malformed_value_fails_closed(self, tmp_path):
        path = _write_config(tmp_path, {
            "hosts": {"hermes": {"enabled": True, "excludeChats": {"discord": ROOM}}},
        })
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=path)
        assert cfg.exclude_chats == [EXCLUDE_ALL_CHATS]

    def test_absent_key_means_no_exclusions(self, tmp_path):
        path = _write_config(tmp_path, {"hosts": {"hermes": {"enabled": True}}})
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=path)
        assert cfg.exclude_chats == []


# The AIAgent -> provider kwargs boundary is covered in
# tests/run_agent/test_memory_provider_init.py (constructing a full agent
# is not network-free, which this package forbids).
