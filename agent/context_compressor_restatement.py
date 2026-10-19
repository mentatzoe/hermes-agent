"""What compaction restates after the summary while a request is still running.

A cron job's prompt, a ``/skill`` turn and a webhook route with skills open with the full text of every
skill they load and end with the instruction. Restating such a request whole carried all of that text past
the summary again (aleph-vault#203). When the agent can reload skills, the restatement keeps the instruction
and gives each skill as its ``[SKILL_PRUNED: ...]`` reload marker instead. A skill's resolved config block
stays, because ``skill_view`` does not return it. Anything ambiguous is restated whole.
"""

from __future__ import annotations

import re
from typing import Any

from agent.skill_commands import _SINGLE_SKILL_INSTRUCTION, _SINGLE_SKILL_MARKER, _SKILL_INVOCATION_PREFIX

# The activation note the cron and ``/skill`` builders write before each skill's text, at a line start.
_ACTIVATION_RE = re.compile(
    r"(?m)^" + re.escape(_SKILL_INVOCATION_PREFIX) + r'"([^"\n]+)" skill, [^\n]*'
    + re.escape(_SINGLE_SKILL_MARKER) + r"$"
)
# Bundle members are introduced by the bundle's name, not their own, so such a request is restated whole.
_BUNDLE_RE = re.compile(r"(?m)^\[(?:Loaded as part of the |IMPORTANT: The user has invoked the [^\n]* skill bundle, )")
# Written by ``agent.skill_commands._inject_skill_config`` after a skill's text; it ends on a line holding "]".
_SKILL_CONFIG_RE = re.compile(r"(?ms)^\[Skill config \(from .*?^\]$")
_INSTRUCTION_SEPARATOR = "\n\n" + _SINGLE_SKILL_INSTRUCTION
_RELOAD_NOTE = (
    "[IMPORTANT: This request came with the skills listed below, and their instructions still apply. Their "
    "text was left out when the context was compacted: reload each one with skill_view before acting on "
    "anything that depends on it.]"
)


def restated_task_text(content: Any, reload_skills: bool) -> str:
    """The request to restate after the handoff. A gateway reply pointer stays: the restated request still
    needs its disambiguation. With *reload_skills*, the skill text it carried becomes reload markers."""
    from agent.context_compressor import _restated_request_text

    text = _restated_request_text(content)
    return _without_skill_text(text) if reload_skills else text


def _without_skill_text(text: str) -> str:
    """*text* with each skill's block cut to its reload marker and config block; unchanged unless it holds
    exactly one instruction marker after single-skill activation notes (a quoted marker is ambiguous)."""
    from agent.context_compressor import _skill_pruned_marker
    from agent.skill_utils import normalize_skill_lookup_name

    if text.count(_INSTRUCTION_SEPARATOR) != 1:
        return text
    scaffold, _, instruction = text.partition(_INSTRUCTION_SEPARATOR)
    headers = list(_ACTIVATION_RE.finditer(scaffold))
    if not headers or _BUNDLE_RE.search(scaffold):
        return text
    skills: list[str] = []
    for header, end in zip(headers, [h.start() for h in headers[1:]] + [len(scaffold)]):
        configs = [m.group(0) for m in _SKILL_CONFIG_RE.finditer(scaffold, header.end(), end)]
        skills += [_skill_pruned_marker(normalize_skill_lookup_name(header.group(1))), *configs[-1:]]
    lead = scaffold[: headers[0].start()].strip()
    parts = (lead, "\n".join([_RELOAD_NOTE, *skills]), _SINGLE_SKILL_INSTRUCTION + instruction)
    return "\n\n".join(part for part in parts if part)


def note_skill_reload(agent: Any) -> None:
    """Tell *agent*'s compressor whether a restated request may point at its skills: only when the agent can
    reload them with ``skill_view``. Compactions run by a helper agent without that tool restate in full."""
    from agent.context_compressor import ContextCompressor

    compressor = getattr(agent, "context_compressor", None)
    if isinstance(compressor, ContextCompressor):
        compressor._skill_reload_available = "skill_view" in (getattr(agent, "valid_tool_names", None) or ())
