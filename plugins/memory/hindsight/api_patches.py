"""Hermes-managed local patches for installed Hindsight API packages.

These helpers deliberately keep the patch payload in the Hermes repository so
local Hindsight API deviations are reviewable and testable. They patch only the
installed local source file used by the embedded daemon; remote/cloud Hindsight
APIs cannot be changed from Hermes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import logging
import re

logger = logging.getLogger(__name__)

HINDSIGHT_API_REMOVE_NARRATOR_PATCH_MARKER = "HERMES-PATCH K6-B3 remove narrator bank profile name"

_NARRATOR_BLOCK_RE = re.compile(
    r'''    narrator_section = ""\n'''
    r'''    if agent_name:\n'''
    r'''        narrator_section = f'\\nNarrator: \{agent_name\} \(AI agent — first-person statements like "I did X" are the agent\\?'s own actions; classify as "assistant"\)'\n'''
)

_PATCHED_NARRATOR_BLOCK = f'''    # {HINDSIGHT_API_REMOVE_NARRATOR_PATCH_MARKER}: do not inline the bank
    # profile/display name as a Narrator line. Hindsight derives agent_name
    # from the bank profile; placing it in the extraction prompt caused bank
    # names such as AlephScratch/personal-bank to be extracted as entities.
    narrator_section = ""
'''


@dataclass(frozen=True)
class HindsightApiPatchResult:
    """Result for a local Hindsight API source patch attempt."""

    path: Path
    changed: bool
    reason: str


def remove_narrator_from_fact_extraction_source(source: str) -> str:
    """Return fact_extraction.py source with the Narrator prompt block removed.

    The patch is intentionally narrow and idempotent: it removes only the exact
    0.5.6/0.6.2 narrator section that injects ``agent_name`` into the rendered
    extraction prompt. If the source is already patched, it is returned
    unchanged. If the expected upstream block is absent, ``ValueError`` is
    raised so package drift is visible instead of silently ignored.
    """
    if HINDSIGHT_API_REMOVE_NARRATOR_PATCH_MARKER in source:
        return source
    patched, count = _NARRATOR_BLOCK_RE.subn(_PATCHED_NARRATOR_BLOCK, source, count=1)
    if count != 1:
        raise ValueError("Hindsight fact_extraction narrator block not found; patch needs review")
    return patched


def _default_fact_extraction_path() -> Path:
    spec = importlib.util.find_spec("hindsight_api.engine.retain.fact_extraction")
    if spec is None or spec.origin is None:
        raise RuntimeError("hindsight_api.engine.retain.fact_extraction is not importable")
    return Path(spec.origin)


def apply_hindsight_api_source_patches(
    *, fact_extraction_path: str | Path | None = None
) -> HindsightApiPatchResult:
    """Apply Hermes local Hindsight API source patches idempotently.

    This function writes the installed ``fact_extraction.py`` when no path is
    supplied. Tests pass a temporary path so the production venv is never touched
    during verification.
    """
    path = Path(fact_extraction_path) if fact_extraction_path is not None else _default_fact_extraction_path()
    source = path.read_text(encoding="utf-8")
    patched = remove_narrator_from_fact_extraction_source(source)
    if patched == source:
        return HindsightApiPatchResult(path=path, changed=False, reason="already patched")
    path.write_text(patched, encoding="utf-8")
    return HindsightApiPatchResult(path=path, changed=True, reason="removed narrator prompt block")


def ensure_hindsight_api_source_patches() -> None:
    """Best-effort local patch hook used before starting embedded Hindsight.

    Failing closed would make the whole memory provider unavailable when a user
    has a package version whose prompt already changed. Instead, log loudly and
    continue; the validation harness catches narrator regressions before any
    promotion back to auto-injection.
    """
    try:
        result = apply_hindsight_api_source_patches()
    except Exception as exc:  # pragma: no cover - defensive logging path
        logger.warning("Hindsight API local patch check failed: %s", exc, exc_info=True)
        return
    if result.changed:
        logger.warning("Applied local Hindsight API patch at %s: %s", result.path, result.reason)
    else:
        logger.debug("Hindsight API local patch unchanged at %s: %s", result.path, result.reason)
