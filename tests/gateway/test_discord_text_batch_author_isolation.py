"""Discord text batching must never merge different authors (Zoe, 2026-09-30).

In shared sessions (threads, or group_sessions_per_user=false) every
participant shares one session key.  Batching exists only to re-join one
author's client-side 2000-char splits, so the key must include the author.
"""
from types import SimpleNamespace

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType, SessionSource
from plugins.platforms.discord.adapter import DiscordAdapter


def _adapter(**extra):
    a = DiscordAdapter.__new__(DiscordAdapter)
    a.config = SimpleNamespace(extra={"group_sessions_per_user": False, **extra})
    a._session_key_profile = lambda source: None
    return a


def _ev(user_id, thread_id=None):
    src = SessionSource(platform=Platform.DISCORD, chat_id="c1", chat_type="thread" if thread_id else "group",
                        user_id=user_id, user_name=user_id, thread_id=thread_id)
    return MessageEvent(text="x", message_type=MessageType.TEXT, source=src)


def test_different_authors_never_share_a_batch_in_a_shared_channel():
    a = _adapter()
    assert a._text_batch_key(_ev("zoe")) != a._text_batch_key(_ev("sol-bot"))


def test_different_authors_never_share_a_batch_in_a_thread():
    a = _adapter()
    assert a._text_batch_key(_ev("zoe", "t1")) != a._text_batch_key(_ev("opus-bot", "t1"))


def test_same_author_splits_still_batch_together():
    a = _adapter()
    assert a._text_batch_key(_ev("zoe", "t1")) == a._text_batch_key(_ev("zoe", "t1"))
