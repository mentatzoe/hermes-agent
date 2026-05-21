# status-sidecar

A small durable ledger that holds **current operational hints** for the
agent (active kanban task, workspace, last tool, recent drift signals)
and appends them as a bounded `<system_status>` block to the user
message before each LLM call.

This is the HEARTBEAT recommendation from
`projects/aleph/hermes-harness/caveagent-status-ledger-design.md`. It is
*not* semantic memory and *not* canonical truth — it's a low-confidence
pointer the agent should verify with real tools before acting on.

## What it does

1. `pre_llm_call` hook reads `$HERMES_HOME/state/status.db` and, when the
   record is fresh (≤ 4 hours by default), returns a `{"context": "..."}`
   dict. The agent loop (`run_agent.py`) appends that to the user message
   via `_plugin_user_context`. **The system prompt is never touched** —
   prompt-cache prefix stays intact.

2. `status_update` tool lets the agent post a short drift signal or update
   the pointers. Length-bounded; the drift ring buffer keeps only the
   most recent 5 entries.

## Strict TTL — no laundering

Stale state (older than `STATUS_SIDECAR_TTL_SECONDS`, default 14400s) is
**dropped**, not surfaced. This is deliberate: it's better to inject
nothing than to let the agent treat an abandoned task from yesterday as
current truth.

## Rollback

To disable:

1. Remove `status-sidecar` from `plugins.enabled` in
   `~/.hermes/config.yaml` (or add it to `plugins.disabled`).
2. Ask Zoe to `/restart` the gateway / CLI — plugins are discovered at
   startup.
3. The ledger file at `$HERMES_HOME/state/status.db` is harmless when
   not read. To wipe it: `rm $HERMES_HOME/state/status.db*`.

## Failure modes

- **Missing ledger:** hook returns `None`, no block injected.
- **Corrupt SQLite file:** library swallows the
  `sqlite3.OperationalError`, returns `{}`, hook returns `None`.
- **Empty record:** hook returns `None`.
- **Stale record (> TTL):** hook returns `None`.
- **Oversized rendered block:** truncated to ~1200 chars with a
  `[truncated]` marker.

## Configuration

| env var | default | what it does |
|---|---|---|
| `STATUS_SIDECAR_TTL_SECONDS` | `14400` | Block is dropped if the ledger's `status_updated_at` is older than this many seconds. |

## Tests

```bash
/Users/zmll/.venv/bin/python -m pytest tests/plugins/test_status_sidecar_plugin.py -v
```

## Notes for future maintainers

- The plugin is **opt-in** like every standalone Hermes plugin. Activation
  requires `plugins.enabled: [..., status-sidecar]` in config.yaml plus a
  restart.
- The `status_update` tool is registered under the `memory` toolset so it
  ships alongside other "agent-writes-its-own-context" tools, not the
  always-on core surface. Safe-mode profiles won't see it.
- Hindsight remains in `memory_mode: tools`, `auto_recall: false`,
  `auto_retain: false`. This plugin does NOT change those.
- The ledger is intentionally single-row. If someone wants per-tenant
  status, add a `tenant` column and key by `(tenant, id)` — but keep
  `read_status()` returning a single row at a time so the hook stays
  cheap.
