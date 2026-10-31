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


def mid_turn_human_row(agent, entries, *, content: str, **metadata) -> dict:
    """Build any delivered human row without deriving facts from its rendered text.

    ``steer_entries`` is the persisted metadata key used by older plugin versions;
    it now carries the same entry type for both steering and active redirects.
    """
    row = {"role": "user", "content": content, **metadata}
    if isinstance(entries, str):
        return row  # legacy caller has no per-human provenance
    row["steer_entries"] = deepcopy(entries)
    paths = [path for entry in entries for path in
             (entry.get("facts") or {}).get("recoverable_audio_paths", [])]
    if paths:
        current = getattr(agent, "_turn_recoverable_audio_paths", None) or []
        agent._turn_recoverable_audio_paths = list(dict.fromkeys([*current, *paths]))
    return row
