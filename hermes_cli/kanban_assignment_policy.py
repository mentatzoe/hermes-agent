"""Shared-board assignment policy, independent of the dispatching profile.

The root home may reserve its conversational profile. Keeping the check at
mutation boundaries covers CLI, dashboard, tools, automation and graph builders;
read-side normalization must still accept historical assignments.
"""
from __future__ import annotations


class AssignmentPolicyError(ValueError):
    """An assignment refused by the shared home's policy."""


def _canonical_assignee(assignee: str | None) -> str | None:
    """Lowercase-assignee normalization for Kanban rows (dashboard/CLI parity)."""
    if assignee is None:
        return None
    from hermes_cli.profiles import normalize_profile_name

    return normalize_profile_name(assignee)


def default_assignment_allowed() -> bool:
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_cli.kanban_db import kanban_home

    path = kanban_home() / "config.yaml"
    try:
        cfg = load_user_config_effective(config_path=path, fail_closed=True)
        value = cfg.get("kanban", {}).get("allow_default_assignee", True)
    except Exception as exc:
        raise AssignmentPolicyError("cannot read shared kanban.allow_default_assignee policy") from exc
    if type(value) is not bool:
        raise AssignmentPolicyError("shared kanban.allow_default_assignee must be a boolean")
    return value


def assignment_allowed(name: str | None) -> bool:
    return _canonical_assignee(name) != "default" or default_assignment_allowed()


def assignment_assignee(name: str | None) -> str | None:
    name = _canonical_assignee(name)
    if not assignment_allowed(name):
        raise AssignmentPolicyError(
            "Kanban assignee 'default' is reserved for conversation by the shared home; "
            "choose a worker profile (kanban.allow_default_assignee=false)."
        )
    return name


def resolve_dispatch_default(default_assignee: str | None) -> str | None:
    """Resolve a real, claimable default without assigning forbidden profiles.

    If the profiles module cannot load, the downstream spawn gate still buckets
    missing profiles as nonspawnable. Policy read failures always fail closed.
    """
    from hermes_cli.kanban_db_dispatch import _profile_exists_fn

    name = (default_assignee or "").strip() or None
    if name:
        try:
            if not assignment_allowed(name):
                return None
        except AssignmentPolicyError:
            return None
        profile_exists = _profile_exists_fn()
        if profile_exists is not None and not profile_exists(name):
            return None
    return name
