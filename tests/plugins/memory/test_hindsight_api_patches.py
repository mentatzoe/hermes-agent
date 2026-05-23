"""Tests for Hermes-managed local Hindsight API source patches."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from plugins.memory.hindsight.api_patches import (
    HINDSIGHT_API_REMOVE_NARRATOR_PATCH_MARKER,
    apply_hindsight_api_source_patches,
    remove_narrator_from_fact_extraction_source,
)


UPSTREAM_BUILD_USER_MESSAGE = '''
def _build_user_message(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    event_date,
    context: str,
    metadata: dict[str, str] | None = None,
    agent_name: str | None = None,
) -> str:
    metadata_section = ""
    if metadata:
        metadata_lines = "\\n".join(f"  {k}: {v}" for k, v in metadata.items())
        metadata_section = f"\\nMetadata:\\n{metadata_lines}"

    narrator_section = ""
    if agent_name:
        narrator_section = f'\\nNarrator: {agent_name} (AI agent — first-person statements like "I did X" are the agent\'s own actions; classify as "assistant")'

    return f"""Extract facts from the following text chunk.
Context: {context}{metadata_section}{narrator_section}
Text:
{chunk}"""
'''


@pytest.mark.parametrize(
    "bank_profile_name",
    [
        "narrator-sentinel-12345",
        "personal-bank",
        "AlephScratch",
        "unnamed",
        "name with whitespace",
        "name-with-punctuation!?,.;:()[]{}",
    ],
)
def test_remove_narrator_patch_omits_bank_profile_name_from_rendered_prompt(bank_profile_name):
    patched = remove_narrator_from_fact_extraction_source(UPSTREAM_BUILD_USER_MESSAGE)
    namespace = {}
    exec(patched, namespace)

    rendered = namespace["_build_user_message"](
        "User: I prefer concise terminal output.\nAssistant: Noted.",
        0,
        1,
        None,
        "synthetic exchange",
        metadata=None,
        agent_name=bank_profile_name,
    )

    assert "Narrator:" not in rendered
    assert bank_profile_name not in rendered


def test_remove_narrator_patch_is_idempotent_and_marks_local_deviation():
    once = remove_narrator_from_fact_extraction_source(UPSTREAM_BUILD_USER_MESSAGE)
    twice = remove_narrator_from_fact_extraction_source(once)

    assert once == twice
    assert HINDSIGHT_API_REMOVE_NARRATOR_PATCH_MARKER in once


@pytest.mark.asyncio
async def test_patched_extraction_path_never_sends_or_returns_narrator_sentinel(monkeypatch):
    fact_extraction = pytest.importorskip("hindsight_api.engine.retain.fact_extraction")
    response_models = pytest.importorskip("hindsight_api.engine.response_models")
    TokenUsage = response_models.TokenUsage

    source_path = fact_extraction.__file__
    source = open(source_path, encoding="utf-8").read()
    patched_source = remove_narrator_from_fact_extraction_source(source)
    namespace = dict(fact_extraction.__dict__)
    exec(compile(patched_source, source_path, "exec"), namespace)
    patched_build_user_message = namespace["_build_user_message"]
    monkeypatch.setattr(fact_extraction, "_build_user_message", patched_build_user_message)

    captured_messages = []

    class FakeLLM:
        provider = "test"
        model = "fake"

        async def call(self, **kwargs):
            captured_messages.extend(kwargs["messages"])
            return {"facts": []}, TokenUsage()

    config = SimpleNamespace(
        retain_extraction_mode="standard",
        retain_extract_causal_links=False,
        retain_llm_max_retries=1,
        llm_max_retries=1,
        retain_llm_initial_backoff=None,
        llm_initial_backoff=0,
        retain_llm_max_backoff=None,
        llm_max_backoff=0,
        retain_max_completion_tokens=256,
        entity_labels=None,
        entities_allow_free_form=True,
    )

    facts, _usage = await fact_extraction._extract_facts_from_chunk(
        "User: I like robust tests.\nAssistant: I will add them.",
        0,
        1,
        datetime(2026, 5, 23, tzinfo=timezone.utc),
        "synthetic exchange",
        FakeLLM(),
        config,
        agent_name="narrator-sentinel-12345",
        metadata=None,
    )

    rendered_prompt = "\n".join(message["content"] for message in captured_messages)
    assert "narrator-sentinel-12345" not in rendered_prompt
    assert all("narrator-sentinel-12345" not in str(fact) for fact in facts)


def test_apply_hindsight_api_source_patches_writes_temp_fact_extraction(tmp_path):
    package = tmp_path / "hindsight_api" / "engine" / "retain"
    package.mkdir(parents=True)
    fact_extraction = package / "fact_extraction.py"
    fact_extraction.write_text(UPSTREAM_BUILD_USER_MESSAGE, encoding="utf-8")

    result = apply_hindsight_api_source_patches(fact_extraction_path=fact_extraction)

    patched = fact_extraction.read_text(encoding="utf-8")
    assert result.changed is True
    assert result.path == fact_extraction
    assert "Narrator:" not in patched
    assert HINDSIGHT_API_REMOVE_NARRATOR_PATCH_MARKER in patched
