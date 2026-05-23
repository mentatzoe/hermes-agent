"""Scratch-bank integration coverage for Hindsight retain mission routing."""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import uuid
from urllib.parse import urlsplit, urlunsplit

import pytest

from plugins.memory.hindsight import HindsightMemoryProvider


pytestmark = pytest.mark.integration


_DATABASE_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _quote_database_identifier(name: str) -> str:
    if not _DATABASE_NAME_RE.fullmatch(name):
        raise ValueError(f"unsafe scratch database name: {name!r}")
    return f'"{name}"'


def _scratch_database_url(name: str) -> tuple[str, str] | None:
    source = os.environ.get("HINDSIGHT_SCRATCH_DATABASE_URL")
    if not source:
        return None
    parts = urlsplit(source)
    if parts.scheme not in {"postgres", "postgresql"}:
        return None
    admin_url = urlunsplit((parts.scheme, parts.netloc, "/postgres", parts.query, parts.fragment))
    scratch_url = urlunsplit((parts.scheme, parts.netloc, f"/{name}", parts.query, parts.fragment))
    return admin_url, scratch_url


async def _create_database(admin_url: str, name: str) -> None:
    import asyncpg

    quoted_name = _quote_database_identifier(name)
    conn = await asyncpg.connect(admin_url)
    try:
        await conn.execute("CREATE DATABASE " + quoted_name)
    finally:
        await conn.close()


async def _drop_database(admin_url: str, name: str) -> None:
    import asyncpg

    quoted_name = _quote_database_identifier(name)
    conn = await asyncpg.connect(admin_url)
    try:
        await conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $1",
            name,
        )
        await conn.execute("DROP DATABASE IF EXISTS " + quoted_name)
    finally:
        await conn.close()


@pytest.mark.timeout(240)
@pytest.mark.skipif(
    os.environ.get("HINDSIGHT_RUN_SCRATCH_BANK_INTEGRATION") != "1",
    reason="set HINDSIGHT_RUN_SCRATCH_BANK_INTEGRATION=1 to run scratch-bank Hindsight integration",
)
def test_retain_mission_is_applied_to_scratch_bank_config(tmp_path, monkeypatch):
    """Provider init updates bank config, then retain uses that scratch bank."""
    # The embedded daemon may take longer than the repo's default 30-second
    # pytest-timeout signal alarm on cold starts. This integration has its own
    # explicit 240-second timeout marker, so clear any inherited SIGALRM before
    # starting the scratch-only daemon manager.
    if hasattr(signal, "alarm"):
        signal.alarm(0)
    monkeypatch.setenv("HINDSIGHT_EMBED_DAEMON_STARTUP_TIMEOUT", "180")
    # The embedded Postgres initdb inherits the worker process locale. Some
    # kanban worker environments expose LC_CTYPE=UTF-8 without a valid LANG,
    # which initdb rejects as an invalid locale. Pin a portable C locale for
    # this scratch-only daemon so the integration is independent of host shell
    # locale state.
    monkeypatch.setenv("LANG", "C")
    monkeypatch.setenv("LC_ALL", "C")
    monkeypatch.setenv("LC_CTYPE", "C")

    from hindsight import HindsightEmbedded
    from hindsight_client import Hindsight

    unique = uuid.uuid4().hex[:10]
    db_name = f"hindsight_k6_b4_{unique}"
    profile = f"k6-b4-{unique}"
    bank_id = f"k6-b4-scratch-{unique}"
    sentinel_mission = f"K6_B4_SENTINEL_RETAIN_MISSION_{unique}"
    database_urls = _scratch_database_url(db_name)
    admin_url, scratch_db_url = database_urls if database_urls is not None else (None, None)

    if database_urls is not None:
        asyncio.run(_create_database(admin_url, db_name))
    embedded = None
    provider = None
    client = None
    try:
        embedded = HindsightEmbedded(
            profile=profile,
            llm_provider=os.environ.get("HINDSIGHT_API_LLM_PROVIDER", "openai"),
            llm_api_key=os.environ.get("HINDSIGHT_API_LLM_API_KEY")
            or os.environ.get("HINDSIGHT_LLM_API_KEY", ""),
            llm_model=os.environ.get("HINDSIGHT_API_LLM_MODEL", "gpt-4o-mini"),
            llm_base_url=os.environ.get("HINDSIGHT_API_LLM_BASE_URL"),
            database_url=scratch_db_url or "pg0",
            idle_timeout=0,
        )
        embedded._ensure_started()
        client = Hindsight(base_url=embedded.url)
        client.create_bank(bank_id=bank_id, name="K6 B4 scratch", mission="scratch integration bank")

        config = {
            "mode": "cloud",
            "api_url": embedded.url,
            "bank_id": bank_id,
            "bank_retain_mission": sentinel_mission,
            "retain_async": False,
            "memory_mode": "tools",
        }
        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)

        provider = HindsightMemoryProvider()
        provider.initialize(session_id=f"session-{unique}", hermes_home=str(tmp_path), platform="cli")
        provider.sync_turn(
            "Please remember the scratch probe favorite color is K6 blue.",
            "Stored for the scratch integration probe.",
            session_id=f"session-{unique}",
        )
        provider.shutdown()

        bank_config = client.get_bank_config(bank_id)
        assert sentinel_mission in json.dumps(bank_config, sort_keys=True)
    finally:
        if provider is not None:
            provider.shutdown()
        if client is not None:
            try:
                client.delete_bank(bank_id)
            except Exception:
                pass
        if embedded is not None:
            embedded.close(stop_daemon=True)
        if database_urls is not None:
            asyncio.run(_drop_database(admin_url, db_name))
