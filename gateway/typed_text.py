"""Human-authored text and transcription before the gateway adds model-facing framing."""
import asyncio
import logging
import os
from typing import List, Optional, Tuple

logger = logging.getLogger("gateway.run")

EMPTY_TEXT_PLACEHOLDER = "(The user sent a message with no text content)"


def agent_visible_audio_paths(paths):
    """Ingress attachment paths in the same namespace as tool arguments."""
    from tools.credential_files import to_agent_visible_cache_path
    return [to_agent_visible_cache_path(os.path.abspath(path)) for path in paths]


def authored_turn_kwargs(run_conversation, ctx):
    """Forward host-owned sender facts only to agent signatures that accept them."""
    from agent.interrupt_compat import _accepts_keyword
    fields = {
        "turn_author": {"id": ctx.source.user_id or None, "name": ctx.source.user_name or None,
                        "is_bot": bool(getattr(ctx.source, "is_bot", False))},
        "typed_user_message": ctx.typed_user_message,
        "recoverable_audio_paths": ctx.recoverable_audio_paths,
    }
    return {key: value for key, value in fields.items() if _accepts_keyword(run_conversation, key)}


def typed_text_for_event(event) -> str:
    """Return only words the sender wrote or spoke, never gateway scaffolding.

    STT's visible transcript still includes a forwarded recording of our own reply;
    ``_gateway_typed_stt_transcripts`` excludes it. Pending events use the same filtered
    list. An internal event is machinery even when its text happens to look conversational.
    """
    if getattr(event, "internal", False):
        return ""
    parts = []
    text = getattr(event, "_gateway_typed_source_text", None)
    if text is None:
        text = getattr(event, "text", None) or ""
    if text.strip() and text.strip() != EMPTY_TEXT_PLACEHOLDER:
        parts.append(text)
    transcripts = getattr(event, "_gateway_typed_stt_transcripts", None)
    if transcripts is None:
        transcripts = getattr(event, "_gateway_pending_stt_transcripts", None) or []
    parts.extend(transcript for transcript in transcripts if transcript and transcript.strip())
    return "\n\n".join(parts)


def apply_message_timestamp(event, message_text):
    """Keep model-only prefixes out of persisted sender text; render timestamps only if enabled."""
    from gateway.run import _load_gateway_config, _message_timestamps_enabled
    from gateway.run_inbound import strip_discord_triggering_note
    persist_user_message = None
    persist_user_timestamp = None
    try:
        from hermes_time import get_timezone as _get_evt_tz
        from gateway.message_timestamps import (
            coerce_message_timestamp as _coerce_msg_ts,
            render_user_content_with_timestamp as _render_msg_ts,
            strip_leading_message_timestamps as _strip_msg_ts,
        )
        _evt_tz = _get_evt_tz()
        if message_text and isinstance(message_text, str):
            _clean_message_text, _embedded_ts = _strip_msg_ts(message_text, tz=_evt_tz)
            persist_user_message = strip_discord_triggering_note(event, _clean_message_text)
            _event_epoch = _coerce_msg_ts(getattr(event, "timestamp", None), tz=_evt_tz)
            persist_user_timestamp = _event_epoch if _event_epoch is not None else _embedded_ts
            if _message_timestamps_enabled(_load_gateway_config()):
                message_text = _render_msg_ts(_clean_message_text, persist_user_timestamp, tz=_evt_tz)
            else:
                # Toggle off: the model sees the clean message; timestamp stored for later opt-in.
                message_text = _clean_message_text
    except Exception as _ts_err:
        logger.debug("Message timestamp injection failed (non-fatal): %s", _ts_err, exc_info=True)
    return message_text, persist_user_message, persist_user_timestamp


async def transcribe_one_clip(self, path: str, transcribe_audio, transcribe_audio_local_fallback) -> Tuple[Optional[str], str]:
    """``(transcript_or_None, note)`` for one clip via configured STT with local fallback."""
    result = await asyncio.to_thread(transcribe_audio, path, None, "gateway")
    if not result.get("success"):
        fallback = await asyncio.to_thread(transcribe_audio_local_fallback, path)
        if fallback.get("success"):
            logger.info("Configured STT failed for %s; recovered with local STT", path)
            result = fallback
    if not result["success"]:
        logger.info("Voice transcription failed for %s: %s", path, result.get("error", "unknown error"))
        return None, self._untranscribed_audio_note(path)
    transcript = result["transcript"]
    # Empty quotes make the agent reply to nothing and can loop, so emit a sentinel note (#41603).
    if not (transcript or "").strip():
        return None, (
            "[The user sent a voice message but it came through "
            "empty or inaudible — speech-to-text returned no "
            "words. Do not guess at the content; ask the user "
            "to resend or type it out.]"
        )
    # A plain quoted line does not read as a meta-instruction about voice mode.
    from gateway.run_voice import _is_agents_own_voice
    if await asyncio.to_thread(_is_agents_own_voice, path, transcript):
        return transcript, (
            "[This recording is Aleph's own earlier voice reply, sent back — "
            f'the words are Aleph\'s, not the sender\'s.] "{transcript}"'
        )
    return transcript, f'"{transcript}"'


async def enrich_message_with_transcription(
    self, user_text: str, audio_paths: List[str], *, typed_transcripts: Optional[List[str]] = None,
    recoverable_audio_paths: Optional[List[str]] = None,
) -> tuple[str, List[str]]:
    """Return enriched text and successful transcripts, optionally collecting sender-authored speech."""
    from gateway.run import _probe_audio_duration
    audio_paths = list(dict.fromkeys(audio_paths))
    owned_paths = dict(zip(audio_paths, await asyncio.to_thread(agent_visible_audio_paths, audio_paths)))
    if recoverable_audio_paths is not None:
        recoverable_audio_paths.extend(owned_paths.values())
    if not getattr(self.config, "stt_enabled", True):
        notes = []
        for path in audio_paths:
            abs_path = await asyncio.to_thread(os.path.abspath, path)
            duration_str = await _probe_audio_duration(abs_path)
            suffix = f" (duration: {duration_str})" if duration_str else ""
            notes.append(f"[The user sent a voice message: {abs_path}{suffix}]")
        return (self._prepend_media_prefix("\n\n".join(notes), user_text) if notes else user_text), []

    try:
        from tools.transcription_tools import transcribe_audio, transcribe_audio_local_fallback
    except ModuleNotFoundError as e:
        logger.error("Transcription module unavailable: %s", e)
        return self._prepend_media_prefix("[voice message could not be transcribed]", user_text), []

    enriched_parts = []
    successful_transcripts: List[str] = []
    for path in audio_paths:
        try:
            logger.debug("Transcribing user voice: %s", path)
            transcript, note = await self._transcribe_one_clip(
                path, transcribe_audio, transcribe_audio_local_fallback,
            )
            if transcript is not None:
                successful_transcripts.append(transcript)
                if recoverable_audio_paths is not None:
                    recoverable_audio_paths.remove(owned_paths[path])
                if typed_transcripts is not None and not note.startswith(
                    "[This recording is Aleph's own earlier voice reply"
                ):
                    typed_transcripts.append(transcript)
            enriched_parts.append(note)
        except Exception as e:
            logger.exception("Transcription error: %s", e)
            enriched_parts.append(self._untranscribed_audio_note(path))

    if enriched_parts:
        user_text = self._prepend_media_prefix("\n\n".join(enriched_parts), user_text)
    return user_text, successful_transcripts
