# status-sidecar

A small durable ledger that holds **current operational hints** for the
agent (active project, active sessions, active Kanban task, workspace,
last tool, recent drift signals) and appends them as a bounded
`<system_status>` block to the user message before each LLM call.

This is the HEARTBEAT recommendation from
`projects/aleph/hermes-harness/caveagent-status-ledger-design.md`. It is
*not* semantic memory and *not* canonical truth — it's a low-confidence
pointer the agent should verify with real tools before acting on.

## What it does

1. **`pre_llm_call` hook** records the current session in an
   `active_sessions` table, reads `$HERMES_HOME/state/status.db`, and
   returns a `{"context": "..."}` dict when either the current-status row
   or at least one session pointer is fresh (≤ 4 hours by default). The
   agent loop (`run_agent.py`) appends that to the user message via
   `_plugin_user_context`. **The system prompt is never touched** —
   prompt-cache prefix stays intact.

2. **`post_tool_call` hook** updates the ledger deterministically after
   observed tool calls so the status stays fresh without the agent
   having to remember to write it. Currently tracked:

   | Tool family | What lands in the ledger |
   |---|---|
   | `kanban_show`, `kanban_complete`, `kanban_block`, `kanban_comment`, `kanban_link`, `kanban_heartbeat` | `active_kanban_task_id` (from result `task.id` or args), `active_workspace` (kanban_show only), inferred `active_project_slug`, `last_tool_invocation` |
   | `kanban_create` | `last_tool_invocation`, plus a `created kanban card <id>` drift signal (no title / body) |
   | `cronjob` | `last_cron_job_id` (from result `job_id` or args), `last_tool_invocation`, plus `cronjob <action> [<id>]` drift signal (no prompt / schedule / script content) |
   | `write_file`, `patch` | `last_tool_invocation = "<tool>:<basename>"` plus inferred `active_project_slug` where possible (no full path, no content, no diff) |
   | Anything else | Ignored. `terminal`, `status_update`, `read_file`, `search_files` are explicitly excluded to keep noise out and avoid recursion. |

   **Hard invariant:** nothing from a tool's result *body* or args
   *content* lands in the ledger. The decision policy is in the pure
   function `status_sidecar.derive_status_from_tool_call`, with tests
   that grep the raw SQLite bytes to prove no prose, file content, cron
   prompts, or kanban bodies leak through.

3. **`on_session_finalize` hook** appends a single compact drift signal
   when a session ends — `"session ended: N tool calls, last=<tool>"`.
   Pure breadcrumb, no user / assistant prose captured. Skipped when a
   session had no tracked tool calls.

4. **`status_update` tool** lets the agent post a short drift signal or
   override the auto-tracked pointers, including `active_project_slug`.
   Length-bounded; the drift ring buffer keeps only the most recent 5
   entries.

## Tool exposure

The `status_update` tool is registered under the **`todo`** toolset, not
`memory`. This is deliberate: Aleph's actual DM surface is Discord, and
`platform_toolsets.discord` in the user config does NOT include
`memory`. Registering under `todo` (which IS allowlisted on Discord and
in the default safe-profile set) means the tool is reachable from
Aleph's actual chat surface.

If you want to gate the tool further (e.g. allow only the deterministic
`post_tool_call` path on locked-down profiles), remove `todo` from that
profile's `platform_toolsets` — the plugin will continue updating the
ledger via the hook even if the tool itself isn't loaded.

## Strict TTL — no laundering

Stale current-status rows (older than `STATUS_SIDECAR_TTL_SECONDS`,
default 14400s) are **dropped**, not surfaced. Fresh active-session rows
may still render, but they do not refresh the stale task/project row.
This is deliberate: it's better to show only a fresh session pointer
than to let the agent treat an abandoned task from yesterday as current
truth.

## Rollback

To disable:

1. Remove `status-sidecar` from `plugins.enabled` in
   `~/.hermes/config.yaml` (or add it to `plugins.disabled`).
2. Ask Zoe to `/restart` the gateway / CLI — plugins are discovered at
   startup.
3. The ledger file at `$HERMES_HOME/state/status.db` is harmless when
   not read. To wipe it: `rm $HERMES_HOME/state/status.db*`.

## Failure modes

- **Missing ledger:** hook can still render the current session after
  touching `active_sessions`; without a session id it returns `None`.
- **Corrupt SQLite file:** library swallows the
  `sqlite3.OperationalError`, returns `{}` / `[]`, hook returns `None`
  or a session-only block.
- **Empty status row:** current session may render; no fake task/project
  status is created.
- **Stale status row (> TTL):** stale task/project fields are skipped;
  fresh active-session rows may still render.
- **Oversized rendered block:** truncated to ~1200 chars with a
  `[truncated]` marker.
- **Hook errors:** every hook is wrapped in try/except and logs at
  debug. A failing status write must never break the agent loop.

## Configuration

| env var | default | what it does |
|---|---|---|
| `STATUS_SIDECAR_TTL_SECONDS` | `14400` | Current-status row is dropped if `status_updated_at` is older than this many seconds; active sessions use the same TTL against `last_seen_at`. |

## Tests

```bash
/Users/zmll/.hermes/hermes-agent/venv/bin/python -m pytest \
  tests/plugins/test_status_sidecar_plugin.py \
  tests/plugins/test_status_sidecar_deterministic_updates.py -v
```

53 tests total: the original ledger/inject suite plus deterministic
updates, active-session tracking, project-slug inference, and privacy
checks. The deterministic suite specifically asserts the no-content
invariant by reading raw bytes from the SQLite file after a hook run and
grepping for marker strings.

## Activation smoke plan

The plugin is **opt-in**. Activation is a two-step manual operation
because plugin discovery happens once at startup; an agent cannot
self-activate it.

### Step 1: merge / cherry-pick

```bash
# Option A: merge the branch into the installed checkout
cd /Users/zmll/.hermes/hermes-agent
git merge --ff-only feat/status-sidecar-sqlite-reconciled

# Option B: cherry-pick just the relevant commits
git cherry-pick <status-sidecar-commit-sha>
```

### Step 2: enable the plugin

Edit `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - accountability
    - message-timestamps
    - superpowers
    - status-sidecar    # add this line
```

### Step 3: restart

Have Zoe `/restart` the gateway from the chat surface (do NOT
`launchctl kickstart` from an agent tool — restart is a human-driven
operation).

### Step 4: smoke (post-restart, two checks)

**Check A — fresh status injects:**

```bash
/Users/zmll/.hermes/hermes-agent/venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "/Users/zmll/.hermes/hermes-agent/plugins/runtime/status_sidecar")
import status_sidecar as ss
ss.write_status(
    active_kanban_task_id="t_smoke",
    active_workspace="/tmp/smoke/hermes-agent",
    active_project_slug="hermes-agent",
    last_tool_invocation="smoke_check",
)
ss.touch_session(
    session_id="smoke-session",
    surface="cli",
    active_project_slug="hermes-agent",
    active_kanban_task_id="t_smoke",
    active_workspace="/tmp/smoke/hermes-agent",
    last_tool_invocation="smoke_check",
)
print(ss.render_status_block())
PY
```

Then in Aleph DM: ask a trivial question. Verify the model's first
response acknowledges (or at minimum doesn't contradict) the
`active_kanban_task_id=t_smoke` pointer. The block should be present in
the user message — confirm by inspecting the next session log under
`~/.hermes/logs/gateway.log`.

**Check B — stale status disappears:**

```bash
/Users/zmll/.hermes/hermes-agent/venv/bin/python - <<'PY'
import sys, time
sys.path.insert(0, "/Users/zmll/.hermes/hermes-agent/plugins/runtime/status_sidecar")
import status_sidecar as ss
ss._force_set_updated_at(time.time() - 24 * 3600)   # 24h ago
print("rendered:", repr(ss.render_status_block()))
PY
```

Output must be `rendered: ''` for the direct script when no fresh session
row exists. If you then send a DM, `pre_llm_call` may render a
session-only block; verify the stale `active_kanban_task_id` is absent.

### Step 5: deterministic update smoke

Send a DM that exercises a tracked tool (e.g. "show me kanban task
t_8a4caa0e"). After the turn:

```bash
/Users/zmll/.hermes/hermes-agent/venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "/Users/zmll/.hermes/hermes-agent/plugins/runtime/status_sidecar")
import status_sidecar as ss
print(ss.read_status())
PY
```

The dict should now contain `active_kanban_task_id=t_8a4caa0e`, an
`active_workspace`, and an inferred `active_project_slug`, written by
the `post_tool_call` hook without any explicit `status_update` call from
the agent. `read_active_sessions(14400)` should also include the session
that ran the tool.

Until all of the above passes on the live gateway, the plugin remains
in **"implemented, not live"** state.

## Notes for future maintainers

- The plugin is **opt-in** like every standalone Hermes plugin.
  Activation requires `plugins.enabled: [..., status-sidecar]` in
  config.yaml plus a restart.
- The `status_update` tool is registered under the `todo` toolset (see
  the "Tool exposure" section above for the rationale).
- Hindsight remains in `memory_mode: tools`, `auto_recall: false`,
  `auto_retain: false`. This plugin does NOT change those.
- The current-status row is intentionally single-row, with a separate
  bounded `active_sessions` table. If someone wants per-tenant status,
  add a `tenant` column and key by `(tenant, id)` — but keep
  `read_status()` returning a single row at a time and keep
  `read_active_sessions()` bounded so the hook stays cheap.
- The `derive_status_from_tool_call` function in `status_sidecar.py` is
  the only place that decides what's safe to land in the ledger. To
  track a new tool, add it there (and write a paired test that proves
  no result body leaks). Do not bypass it from the hook.
