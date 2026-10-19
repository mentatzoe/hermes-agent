"""Companion-mode compaction: identity header, letter, factual appendix, lane notes.

Opt-in via ``compression.companion`` in config.yaml (Zoe, 2026-09-29). The stock
template is a coding-session task log; for a personal agent it drops who the
people are (misgendering the user) and flattens relationship/register into
"Completed Actions". Companion mode changes three things and nothing else:

1. **Identity header, re-sourced every compaction, never paraphrased.** Built
   deterministically from configured files (SOUL, a people file) plus the memory
   provider's pre-compress context. It is *prepended outside* the summarizer's
   output, so repeated compactions can never drift it (no summary-of-summary).
2. **Letter template.** The summarizer writes a letter from the agent to its
   next self (what happened, how, in the user's words, register, what's open
   between us, fiction marked as fiction) plus a tight factual appendix that
   keeps retrieval indexes (issue numbers, paths, session ids, commands).
   The letter is REWRITTEN each time from previous letter + new turns.
3. **Lane notes.** One or two lines of lane-specific guidance selected by
   chat id / thread parent / display-name substring, appended to the prompt.

The deterministic lean sections (anchor index, verbatim user messages,
recovery footer) still attach after the summarizer's output, unchanged.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

# A configured path that names the Hermes home: ``~/.hermes``, ``$HERMES_HOME`` or
# ``${HERMES_HOME}``, alone or followed by a separator. These resolve through
# hermes_constants.get_hermes_home() so a profile or relocated home reads its own files.
_HOME_PREFIX = re.compile(r"^(?:~[/\\]\.hermes|\$HERMES_HOME|\$\{HERMES_HOME\})(?=$|[/\\])")

# Warnings already emitted in this process, keyed by (kind, path). Each gap warns once.
_warned: set = set()

IDENTITY_HEADING = "## Who Is Here (re-sourced at every compaction — not a summary)"
IDENTITY_END = "<!-- end of re-sourced identity header -->"
LETTER_HEADING = "## Letter To My Next Self"
APPENDIX_HEADING = "## Factual Appendix (for retrieval)"

_IDENTITY_MAX_CHARS = 6_000
_FILE_MAX_CHARS = 2_500


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home())


def _expand(path: str) -> Path:
    """Expand a configured path; a leading ``~/.hermes`` or ``$HERMES_HOME`` is the active home."""
    raw = str(path)
    match = _HOME_PREFIX.match(raw)
    if match:
        rest = raw[match.end():].lstrip("/\\")
        home = _hermes_home()
        return home / rest if rest else home
    return Path(os.path.expanduser(os.path.expandvars(raw)))


def _warn_once(kind: str, key: str, message: str, *args: Any) -> None:
    if (kind, key) in _warned:
        return
    _warned.add((kind, key))
    logger.warning(message, *args)


def _dedupe_card_lines(text: str) -> str:
    """Drop exact-duplicate lines (the two cards repeat facts)."""
    seen, out = set(), []
    for line in text.splitlines():
        k = line.strip().lower()
        if k and k in seen:
            continue
        seen.add(k)
        out.append(line)
    return "\n".join(out)


def _read_bounded(path: str, limit: int = _FILE_MAX_CHARS) -> str:
    p: Optional[Path] = None
    try:
        p = _expand(path)
        if not p.is_file():
            _warn_once("missing", str(p),
                       "companion: identity file %s (resolved to %s) is missing; "
                       "the identity header will not include it", path, p)
            return ""
        text = p.read_text(encoding="utf-8-sig", errors="ignore").strip()
    except Exception as exc:
        _warn_once("unreadable", str(p or path),
                   "companion: identity file %s (resolved to %s) could not be read (%s); "
                   "the identity header will not include it", path, p, exc)
        return ""
    if len(text) > limit:
        text = text[:limit].rsplit("\n", 1)[0] + "\n…[truncated at compaction; read the file for the rest]"
    return text


class CompanionConfig:
    """Parsed ``compression.companion`` block. Disabled unless ``enabled: true``."""

    def __init__(self, raw: Any):
        raw = raw if isinstance(raw, dict) else {}
        self.enabled = bool(raw.get("enabled", False))
        self.agent_name = str(raw.get("agent_name") or "the agent").strip()
        self.user_name = str(raw.get("user_name") or "").strip()
        self.people = [str(p).strip() for p in (raw.get("people") or []) if str(p).strip()]
        self.identity_files = [str(p) for p in (raw.get("identity_files") or []) if str(p).strip()]
        self.letter_budget_tokens = int(raw.get("letter_budget_tokens") or 2_500)
        self.appendix_budget_tokens = int(raw.get("appendix_budget_tokens") or 1_500)
        # How old a saved copy of the memory provider's context may be when a compaction gets
        # none (agent/companion_compaction_context.py); unset means a week, 0 turns it off.
        self.saved_context_max_age_hours = raw.get("saved_context_max_age_hours")
        lanes = raw.get("lanes") or []
        self.lanes = [l for l in lanes if isinstance(l, dict) and l.get("note")]
        self.session_id = ""

    # ------------------------------------------------------ other conversations
    def other_conversations(self) -> str:
        """Fresh list of my other recently active conversations (same query as the
        sibling-sessions plugin). Empty on any failure."""
        if not self.session_id:
            return ""
        try:
            import importlib.util

            p = _hermes_home() / "plugins" / "sibling-sessions" / "__init__.py"
            spec = importlib.util.spec_from_file_location("_companion_sib", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            sibs = mod.find_siblings(mod._db_path(), self.session_id)
            return mod.render(sibs).strip() if sibs else ""
        except Exception as exc:
            logger.debug("companion: other conversations unavailable (%s)", exc)
            return ""

    # ------------------------------------------------------------------ names
    def name_people(self, summary: str) -> str:
        """Replace role words in the deterministic scaffolding with names."""
        u = self.user_name
        if not u:
            return summary
        swaps = (
            ("## User Messages (verbatim, newest first)", f"## {u}'s Messages (verbatim, newest first)"),
            ("(Every real user message from the compacted region, quoted verbatim. These are the user's actual words",
             f"(Every real message from {u} in the compacted region, quoted verbatim. These are {u}'s actual words"),
            ("User asked (deterministic, from compacted turns):", f"{u} asked (deterministic, from compacted turns):"),
            ("User asked:", f"{u} asked:"),
        )
        for old, new in swaps:
            summary = summary.replace(old, new)
        return summary

    # ------------------------------------------------------------------ lanes
    def lane_note(self, origin: Optional[Dict[str, Any]]) -> str:
        """First matching lane note for this session origin, or ""."""
        if not origin or not self.lanes:
            return ""
        ids = {str(origin.get(k) or "") for k in ("chat_id", "thread_id", "parent_chat_id")}
        ids.discard("")
        name = str(origin.get("chat_name") or origin.get("display_name") or "").lower()
        for lane in self.lanes:
            match = lane.get("match") or {}
            want_ids = {str(x) for x in (match.get("chat_ids") or [])}
            subs = [str(s).lower() for s in (match.get("name_contains") or []) if str(s)]
            if (want_ids and ids & want_ids) or (subs and any(s in name for s in subs)):
                label = lane.get("name") or "this lane"
                return f"LANE ({label}): {str(lane['note']).strip()}"
        return ""

    # --------------------------------------------------------------- identity
    def identity_header(self, memory_context: str = "") -> str:
        parts = [IDENTITY_HEADING]
        if self.people:
            parts.append("\n".join(f"- {p}" for p in self.people))
        for path in self.identity_files:
            spec = str(path)
            stop = ""
            if "#until:" in spec:  # e.g. ~/.hermes/SOUL.md#until:## Style
                spec, stop = spec.split("#until:", 1)
            body = _read_bounded(spec, limit=20_000 if stop else _FILE_MAX_CHARS)
            if body and stop and stop in body:
                body = body.split(stop, 1)[0].rstrip()
            if len(body) > _FILE_MAX_CHARS:
                body = body[:_FILE_MAX_CHARS].rsplit("\n", 1)[0] + "\n…[truncated at compaction; read the file for the rest]"
            if body:
                parts.append(body)
        if memory_context and memory_context.strip():
            parts.append(_dedupe_card_lines(memory_context.strip())[:_FILE_MAX_CHARS])
        others = self.other_conversations()
        if others:
            parts.append(others)
        if len(parts) == 1:
            try:
                home = str(_hermes_home())
            except Exception:  # the warning must never block the header
                home = "unknown"
            _warn_once("empty-header", "",
                       "companion: identity header is empty (no people, identity files or memory "
                       "context reached it; Hermes home %s); the summarizer gets no names or pronouns",
                       home)
        text = "\n\n".join(parts)
        if len(text) > _IDENTITY_MAX_CHARS:
            text = text[:_IDENTITY_MAX_CHARS].rsplit("\n", 1)[0] + "\n…[identity header capped]"
        return text + "\n" + IDENTITY_END + "\n\n"

    # --------------------------------------------------------------- template
    def template_sections(self, historical_heading: str, historical_instructions: str,
                          pruned_heading: str, lane_note: str, today: str) -> str:
        me = self.agent_name
        people = "; ".join(self.people) if self.people else "the people in the conversation"
        lane = f"\n{lane_note}\n" if lane_note else ""
        dated = f" Today is {today}." if today else ""
        return f"""WHO IS IN THIS CONVERSATION (authoritative — never guess names or pronouns): {people}.
You are writing as {me}, for {me}.{dated}
{lane}
{historical_heading}
{historical_instructions}

{LETTER_HEADING}
[Written by {me} to the {me} who wakes after this compaction, in first person, plain prose.
Up to ~{self.letter_budget_tokens} tokens. REWRITE the whole letter from the previous letter plus
the new turns; never nest or quote earlier letters. Cover, in this order of importance:
- What this conversation is and what it has been like: the register (playful, tender, blunt,
  technical…), running jokes, what landed and what didn't, corrections and how they felt.
- What was said that matters, in the other person's own words where short (quote them).
- What is open between us — not just tasks: questions, promises, things left mid-breath.
- Fiction/roleplay: say what happened in the story and mark it clearly as fiction; characters
  are not the people. Keep canonical scene state if a scene is live.
- What I'd want to remember about myself from this stretch.
Use the correct names and pronouns above. Say names, not "the user". Attribute every quote
and every joke to the person who actually said it; if the turns don't make the speaker clear,
say that instead of guessing. Write it as a letter, in
paragraphs; no bullet lists and no task-log formatting in the letter.]

## In Flight
[Agreements, promises and authorisations made in THIS conversation that are not finished yet —
between us, not just tasks. One line each: what was agreed, who owes it, and its state. Include
things waiting on the other person. Drop anything already done. Write "None." if nothing.]

{APPENDIX_HEADING}
[Tight bullets, up to ~{self.appendix_budget_tokens} tokens. PRESERVE EXACTLY every retrieval
handle a future search would need: issue/PR numbers, file paths, session ids, commit SHAs,
cron/job ids, commands, URLs, counts, dates. One line each: what it is and its current state
(done / live / waiting on whom). Include running processes and anything authorised but not
yet done. No narrative here.]

{pruned_heading}
[If any [SKILL_PRUNED: ...reload with skill_view(...)] markers appear in the input, repeat
each one verbatim here. If none appear, omit this section entirely.]

Write only the sections above, in that order. No preamble."""


def strip_identity_header(summary: str) -> str:
    """Remove a previously prepended identity header (so it is never re-summarized)."""
    if not summary or IDENTITY_HEADING not in summary:
        return summary
    start = summary.find(IDENTITY_HEADING)
    end = summary.find(IDENTITY_END, start)
    if end == -1:
        return summary
    return (summary[:start] + summary[end + len(IDENTITY_END):]).strip()


def load_companion_config(compression_cfg: Any) -> CompanionConfig:
    raw = compression_cfg.get("companion") if isinstance(compression_cfg, dict) else None
    return CompanionConfig(raw)


def session_origin(session_db: Any, session_id: str) -> Optional[Dict[str, Any]]:
    """Origin metadata (chat ids + display name) for lane matching; None on any failure."""
    if not session_db or not session_id:
        return None
    try:
        row = session_db.get_session(session_id) or {}
    except Exception:
        return None
    origin: Dict[str, Any] = {}
    raw = row.get("origin_json")
    if raw:
        try:
            import json
            origin.update(json.loads(raw) if isinstance(raw, str) else dict(raw))
        except Exception:
            pass
    for key in ("chat_id", "thread_id", "display_name"):
        if row.get(key) and not origin.get(key):
            origin[key] = row[key]
    return origin or None
