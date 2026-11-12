"""Local author-controlled context, independent of memory providers."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
import stat
from uuid import uuid4

KEY = "authored_turn_context"
MAX_BYTES = 65536
GUIDANCE = (
    "Author-controlled profile sections, not new user input or inferred memory. "
    "Preserve their meaning and qualifications. They take precedence over "
    "generated descriptions of identity and boundaries, not system or safety "
    "instructions."
)


class AuthoredContextUnavailable(RuntimeError):
    pass


def selected_sections(text: str, names: tuple[str, ...]) -> str:
    """Preserve each selected section's bytes, including blank lines."""
    lines = text.splitlines(keepends=True)
    headings = []
    fence = None
    for index, line in enumerate(lines):
        token = line.rstrip("\r\n")
        mark = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", token)
        if mark:
            run, rest = mark.groups()
            if fence is None:
                fence = (run[0], len(run))
            elif run[0] == fence[0] and len(run) >= fence[1] and not rest.strip():
                fence = None
            continue
        if fence:
            continue
        match = re.match(r"^(#{1,6})[ \t]+(.+?)\s*$", token)
        if match:
            headings.append((index, len(match[1]), match[2]))

    out = []
    for name in names:
        matches = [h for h in headings if h[1:] == (3, name)]
        if len(matches) != 1:
            raise AuthoredContextUnavailable(
                "A required authored section is missing or duplicated"
            )
        start = matches[0][0]
        end = next(
            (i for i, level, _ in headings if i > start and level <= 3),
            len(lines),
        )
        out.append("".join(lines[start:end]))
    return "\n".join(out)


@dataclass(frozen=True)
class Source:
    path: Path
    sections: tuple[str, ...]

    @classmethod
    def from_config(cls, raw, home: Path):
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise AuthoredContextUnavailable("Invalid authored context configuration")
        value, names = raw.get("file"), raw.get("sections")
        if (
            not isinstance(value, str) or not value
            or "://" in value or "\x00" in value
            or not isinstance(names, list) or not names
            or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)
        ):
            raise AuthoredContextUnavailable("Invalid authored context configuration")
        path = Path(value)
        relative = not path.is_absolute()
        path = (home / path if relative else path).resolve()
        if relative and not path.is_relative_to(home.resolve()):
            raise AuthoredContextUnavailable("Relative authored source escaped its home")
        if path.suffix.lower() not in {".md", ".markdown"}:
            raise AuthoredContextUnavailable("Authored context must be Markdown")
        return cls(path, tuple(names))

    def snapshot(self) -> dict:
        try:
            fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
            with os.fdopen(fd, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise AuthoredContextUnavailable("Authored context is not a file")
                raw = stream.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise AuthoredContextUnavailable("Authored context exceeds its limit")
            body = selected_sections(raw.decode("utf-8"), self.sections)
        except (OSError, UnicodeError):
            raise AuthoredContextUnavailable(
                "Configured authored context could not be read"
            ) from None
        return {
            "id": uuid4().hex,
            "sha256": sha256(body.encode("utf-8")).hexdigest(),
            "body": body,
        }


def block(snapshot: dict) -> str:
    body = snapshot["body"]
    if sha256(body.encode("utf-8")).hexdigest() != snapshot["sha256"]:
        raise AuthoredContextUnavailable("Stored authored context is inconsistent")
    return (
        f'<authored-context id="{snapshot["id"]}" '
        f'sha256="{snapshot["sha256"]}">\n'
        f"{GUIDANCE}\n\n{body}\n</authored-context>"
    )


def snapshots(row: dict) -> list:
    return (row.get("display_metadata") or {}).get(KEY, [])


def capture(agent, row: dict) -> None:
    """Call only at actual input admission, never while replaying history."""
    source = getattr(agent, "_authored_turn_source", None)
    if source is None:
        return
    if snapshots(row):
        pending = getattr(agent, "_authored_turn_expected", {})
        agent._authored_turn_expected = {
            **pending, **{s["id"]: block(s) for s in snapshots(row)}
        }
        return
    snapshot = source.snapshot()
    row["display_metadata"] = {
        **(row.get("display_metadata") or {}), KEY: [snapshot]
    }
    pending = getattr(agent, "_authored_turn_expected", {})
    agent._authored_turn_expected = {**pending, snapshot["id"]: block(snapshot)}


def merge(previous: dict, following: dict) -> None:
    combined = {
        item["id"]: item
        for item in [*snapshots(previous), *snapshots(following)]
    }
    if combined:
        previous["display_metadata"] = {
            **(previous.get("display_metadata") or {}),
            KEY: deepcopy(list(combined.values())),
        }


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") for item in content
            if isinstance(item, dict)
            and item.get("type") in {"text", "input_text"}
        )
    return ""


def render(row: dict, content):
    """Idempotent rendering; leaves the caller's content containers untouched."""
    present = text_of(content)
    additions = [block(s) for s in snapshots(row) if block(s) not in present]
    if not additions:
        return content
    extra = "\n\n".join(additions)
    if isinstance(content, str):
        return content + "\n\n" + extra
    if isinstance(content, list):
        return [*content, {"type": "text", "text": extra}]
    raise AuthoredContextUnavailable("Unsupported authored-context message shape")


def check_request(agent, request: dict) -> None:
    """Fail before inference if a transform discarded current authored context."""
    expected = getattr(agent, "_authored_turn_expected", {})
    if not expected:
        return
    rows = request.get("messages", request.get("input"))
    if not isinstance(rows, list):
        raise AuthoredContextUnavailable("Unverified authored-context transport")
    visible = "\n".join(
        text_of(row.get("content")) for row in rows
        if isinstance(row, dict) and row.get("role") == "user"
    )
    if any(value not in visible for value in expected.values()):
        raise AuthoredContextUnavailable("Authored context was lost before inference")
