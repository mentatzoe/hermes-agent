"""Route-independent human input facts, separate from model-facing text."""
from copy import deepcopy
from typing import NotRequired, TypedDict


class PendingHumanInput(TypedDict):
    text: str
    facts: NotRequired[dict]


def pending_human_input(text: str, facts: dict | None = None) -> PendingHumanInput:
    """Snapshot one caller's words and ownership before a surface can reuse them."""
    entry: PendingHumanInput = {"text": text.strip()}
    if facts is not None:
        entry["facts"] = deepcopy(facts)
    return entry


def redirect_text(entries: list[PendingHumanInput] | str) -> str:
    """Keep the existing model-visible separator between accepted corrections."""
    if isinstance(entries, str):
        return entries
    return "\n\n[Additional user correction]\n".join(entry["text"] for entry in entries)


def mid_turn_entries(row: dict) -> list[PendingHumanInput] | None:
    """One source list, in its live or persisted representation (never both)."""
    return row.get("steer_entries", (row.get("display_metadata") or {}).get("mid_turn_entries"))


def retain_mid_turn_entries(row: dict, entries: list[PendingHumanInput]) -> None:
    """Snapshot into the existing durable metadata column; keep the old live-key view.

    The two keys alias the same list, not independently maintained input ledgers.
    Follow-up opener entries retain their separate existing ``steer_entries`` key
    inside display_metadata: those replace the opener, whereas these accompany it.
    """
    snapshot = deepcopy(entries)
    row["steer_entries"] = snapshot
    row["display_metadata"] = {**(row.get("display_metadata") or {}), "mid_turn_entries": snapshot}


def merge_mid_turn_entries(previous: dict, following: dict) -> bool:
    """Conserve accepted occurrences when a canonical row absorbs its successor."""
    incoming = [*((following.get("display_metadata") or {}).get("steer_entries") or []),
                *(mid_turn_entries(following) or [])]
    if not incoming:
        return False
    retain_mid_turn_entries(previous, [*(mid_turn_entries(previous) or []), *incoming])
    return True


def mid_turn_human_row(agent, entries, *, content: str, **metadata) -> dict:
    """Build any delivered human row without deriving facts from its rendered text.

    ``steer_entries`` is the persisted metadata key used by older plugin versions;
    it now carries the same entry type for both steering and active redirects.
    """
    row = {"role": "user", "content": content, **metadata}
    if isinstance(entries, str):
        return row  # legacy caller has no per-human provenance
    retain_mid_turn_entries(row, entries)
    paths = [path for entry in entries for path in
             (entry.get("facts") or {}).get("recoverable_audio_paths", [])]
    if paths:
        current = getattr(agent, "_turn_recoverable_audio_paths", None) or []
        agent._turn_recoverable_audio_paths = list(dict.fromkeys([*current, *paths]))
    return row
