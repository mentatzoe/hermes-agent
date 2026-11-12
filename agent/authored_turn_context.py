"""Local author-controlled context, independent of memory providers."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import tempfile
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
    home: Path

    @classmethod
    def from_config(cls, raw, home: Path):
        if raw is None or raw is False or (isinstance(raw, dict) and raw.get("enabled") is False):
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
        return cls(path, tuple(names), home.resolve())

    def _read_current(self) -> dict:
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
            "version": 1, "binding": self.binding,
            "sha256": sha256(body.encode("utf-8")).hexdigest(),
            "source_sha256": sha256(raw).hexdigest(),
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "body": body,
        }

    @property
    def binding(self) -> dict:
        return {"home": str(self.home), "source": str(self.path), "sections": list(self.sections)}

    @property
    def cache_path(self) -> Path:
        key = sha256(json.dumps(self.binding, sort_keys=True).encode()).hexdigest()
        return self.home / "state" / "authored-turn-context" / (key + ".json")

    def _cache_directory(self, *, create=False) -> Path:
        directory = self.cache_path.parent
        if not directory.resolve().is_relative_to(self.home) or directory.is_symlink():
            raise OSError("Authored cache escaped its home")
        if create:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory.chmod(0o700)
        if directory.stat().st_mode & 0o077:
            raise OSError("Authored cache directory is not private")
        return directory

    def _write_verified(self, value: dict) -> None:
        directory = self._cache_directory(create=True)
        fd, temporary = tempfile.mkstemp(prefix=".verified-", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                os.fchmod(stream.fileno(), 0o600)
                json.dump(value, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.cache_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _read_verified(self) -> dict:
        self._cache_directory()
        fd = os.open(self.cache_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
                raise ValueError("Authored cache is not a private file")
            raw = stream.read(MAX_BYTES * 8 + 1)
        if len(raw) > MAX_BYTES * 8:
            raise ValueError("Oversized authored cache")
        value = json.loads(raw)
        if (value.get("version") != 1 or value.get("binding") != self.binding
                or not isinstance(value.get("body"), str)
                or len(value["body"].encode()) > MAX_BYTES + len(self.sections)
                or sha256(value["body"].encode()).hexdigest() != value.get("sha256")
                or not re.fullmatch(r"[0-9a-f]{64}", value.get("source_sha256", ""))):
            raise ValueError("Invalid authored cache")
        verified = datetime.fromisoformat(value["verified_at"])
        if verified.tzinfo is None or verified > datetime.now(timezone.utc):
            raise ValueError("Invalid verification time")
        selected_sections(value["body"], self.sections)
        return value

    def snapshot(self) -> dict:
        """Current file first, then only this binding's complete verified copy."""
        try:
            value = self._read_current()
        except AuthoredContextUnavailable:
            try:
                value = self._read_verified()
            except (OSError, ValueError, TypeError, KeyError, AuthoredContextUnavailable):
                raise AuthoredContextUnavailable(
                    "Authored context unavailable: current source and verified copy failed"
                ) from None
            return {**value, "id": uuid4().hex, "degraded": True}
        failed = False
        try:
            self._write_verified(value)
        except OSError:
            failed = True
        return {**value, "id": uuid4().hex, "degraded": False, "cache_write_failed": failed}


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


def capture(agent, row: dict, *, expect: bool = True) -> None:
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
    _report_fault(agent, snapshot)
    row["display_metadata"] = {
        **(row.get("display_metadata") or {}), KEY: [snapshot]
    }
    if expect:
        pending = getattr(agent, "_authored_turn_expected", {})
        agent._authored_turn_expected = {**pending, snapshot["id"]: block(snapshot)}


def _report_fault(agent, snapshot: dict) -> None:
    fault = "source" if snapshot.get("degraded") else "cache" if snapshot.get("cache_write_failed") else None
    previous = getattr(agent, "_authored_context_fault", None)
    agent._authored_context_fault = fault
    if fault is None or previous == fault:
        return
    message = (
        "Authored profile source unavailable; using the complete verified copy from "
        + snapshot["verified_at"] + " (SHA-256 " + snapshot["sha256"] + ")."
        if fault == "source" else
        "Authored profile read successfully, but its verified fallback could not be saved."
    )
    logging.getLogger(__name__).warning(message)
    callback = getattr(agent, "status_callback", None)
    if callable(callback):
        from gateway.warning_notifications import DiagnosticText
        callback("lifecycle", DiagnosticText(message))


def merge(previous: dict, following: dict) -> bool:
    combined = {
        item["id"]: item
        for item in [*snapshots(previous), *snapshots(following)]
    }
    changed = list(combined.values()) != snapshots(previous)
    if changed:
        previous["display_metadata"] = {
            **(previous.get("display_metadata") or {}),
            KEY: deepcopy(list(combined.values())),
        }
    return changed


def backfill(agent, row: dict, content, *, api_content=None) -> None:
    """Caller holds the persist lock. Never substitute a newest/text-only match."""
    db, sid = agent._session_db, agent.session_id
    row_id = row.get("_row_id")
    try:
        for attempt in range(2):
            sidecar_ok = api_content is None or db.set_message_api_content(sid, row_id, content, api_content) == 1
            if sidecar_ok and db.set_message_authored_snapshot(sid, row_id, content, snapshots(row)) == 1:
                row["_row_id"] = row_id
                return
            if attempt == 0:
                row_id = db.resolve_active_row_id(sid, row_id)
                if row_id is None:
                    break
    except Exception as exc:
        raise AuthoredContextUnavailable("Authored snapshot persistence failed") from exc
    raise AuthoredContextUnavailable("Authored snapshot durable row could not be resolved")


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
