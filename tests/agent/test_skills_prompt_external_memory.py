"""Honcho memory plan step 1: the main conversation's own prompt stops steering lessons into skills.

With an external memory provider active, three lines in Aleph's prompt change: the skills index
sentence about "the user's preferred approach", the "offer to save as a skill" line, and the
SKILLS_GUIDANCE opening ("record it with skill_manage"). Without one the upstream text is unchanged.
"""

from types import SimpleNamespace

import pytest

from agent.memory_manager import MemoryManager
from agent.prompt_builder import (
    SKILLS_GUIDANCE,
    SKILLS_GUIDANCE_EXTERNAL_MEMORY,
    build_skills_system_prompt,
    clear_skills_system_prompt_cache,
)
from agent.system_prompt import _tool_guidance_block

OLD_LINES = ("preferred approach", "offer to save as a skill", "record it with skill_manage")


@pytest.fixture(autouse=True)
def _fresh_skills_cache():
    clear_skills_system_prompt_cache(clear_snapshot=True)
    yield
    clear_skills_system_prompt_cache(clear_snapshot=True)


@pytest.fixture
def skills_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    skill = tmp_path / "skills" / "misc" / "demo-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo-skill\ndescription: Demo skill.\n---\n# Demo\n")
    return tmp_path / "skills"


def _index(skills_home, **kwargs):
    return build_skills_system_prompt(
        available_tools={"skill_view", "skills_list", "skill_manage"}, skills_dir_override=skills_home, **kwargs)


def test_skills_index_without_a_provider_keeps_the_upstream_lines(skills_home):
    index = _index(skills_home)
    assert "the user's preferred approach" in index
    assert "offer to save as a skill" in index


def test_skills_index_with_a_provider_points_personal_preferences_at_memory(skills_home):
    index = _index(skills_home, external_memory=True)
    assert "preferred approach" not in index and "offer to save as a skill" not in index
    assert "personal preferences are in memory, not in skills" in index
    # What the section is for survives: load skills, fix wrong steps, update a skill that was missing steps.
    assert "skill_view(name)" in index and "skill_manage(action='patch')" in index
    assert "update it before finishing" in index
    # Only personal or task-local material is sent to memory; a correction to a method stays skill material.
    assert "personal preference" in index and "correction about one task" not in index
    assert "demo-skill" in index


def test_the_two_index_variants_do_not_share_a_cache_entry(skills_home):
    with_memory = _index(skills_home, external_memory=True)
    without = _index(skills_home)
    assert with_memory != without
    assert _index(skills_home, external_memory=True) == with_memory
    assert _index(skills_home) == without


def _agent(manager=None, **overrides):
    base = dict(valid_tool_names={"skill_manage", "memory"}, _kanban_worker_guidance="", _memory_manager=manager,
                tools=[{"type": "function", "function": {"name": "memory"}}],
                enabled_toolsets=None, disabled_toolsets=None, _memory_enabled=False, _user_profile_enabled=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def _manager_with(provider_name):
    from agent.memory_provider import MemoryProvider

    class Provider(MemoryProvider):
        name = provider_name

        def is_available(self):
            return True

        def initialize(self, session_id, **kwargs):
            pass

        def get_tool_schemas(self):
            return []

    manager = MemoryManager()
    manager.add_provider(Provider())
    return manager


def test_tool_guidance_without_a_provider_is_unchanged():
    guidance = _tool_guidance_block(_agent(None)) or ""
    assert SKILLS_GUIDANCE in guidance and SKILLS_GUIDANCE_EXTERNAL_MEMORY not in guidance


def test_tool_guidance_with_a_provider_stops_recording_every_workflow_as_a_skill():
    guidance = _tool_guidance_block(_agent(_manager_with("external-test"))) or ""
    assert SKILLS_GUIDANCE_EXTERNAL_MEMORY in guidance and SKILLS_GUIDANCE not in guidance
    # The unconditional opening is gone; a workflow no skill covers can still become one.
    assert "When you work out a non-trivial workflow, record it" not in guidance
    assert "no skill covers" in guidance and "belongs in memory" in guidance
    # What goes to memory is personal or task-local material. A correction to a method is not named here,
    # because it is skill material and the sentence must not read as sending it to memory.
    assert "personal preference" in guidance and "correction about one task" not in guidance
    # The compaction-pruning contract that the opening sentence shares a block with stays.
    assert "[SKILL_PRUNED]" in guidance


def test_builtin_only_manager_does_not_count_as_a_provider():
    guidance = _tool_guidance_block(_agent(_manager_with("builtin"))) or ""
    assert SKILLS_GUIDANCE in guidance


def test_tool_guidance_ignores_a_provider_whose_tools_are_gated_off():
    gated = _agent(_manager_with("external-test"), tools=[], enabled_toolsets=["skills"], disabled_toolsets=["memory"])
    guidance = _tool_guidance_block(gated) or ""
    assert SKILLS_GUIDANCE in guidance and SKILLS_GUIDANCE_EXTERNAL_MEMORY not in guidance


def test_the_provider_variant_keeps_the_filter_safe_shape_of_the_original_opening():
    # The original opening was reworded to dodge a provider-side content filter (see the comment above
    # SKILLS_GUIDANCE). The variant must not reintroduce the rejected phrasing.
    for rejected in ("complex task", "5+", "tricky error", "so you can reuse it next time"):
        assert rejected not in SKILLS_GUIDANCE_EXTERNAL_MEMORY
    assert "\n" in SKILLS_GUIDANCE_EXTERNAL_MEMORY and ("\\" + "n") not in SKILLS_GUIDANCE_EXTERNAL_MEMORY
