---
title: "Authored per-input context"
description: "Opt-in local profile snapshots, verified-copy fallback, persistence and request-boundary checks"
---

# Authored per-input context

`agent.authored_turn_context` delivers selected local Markdown sections independently of memory providers, inferred cards, recall results and the recall budget. It is disabled by default. Existing whole-profile system-prompt delivery is unchanged.

```yaml
agent:
  authored_turn_context:
    file: user-profile.md
    sections:
      - Why we have this
      - Us
```

Paths are absolute or relative to the owning Hermes home. Relative paths cannot escape that home. Sources must be Markdown regular files, UTF-8, at most 64 KiB. Selected headings must each occur exactly once at level three, outside code fences, with nonempty content. The selector retains section bytes and line endings and joins the selected sections in configuration order. It does not summarize them.

Configuration is bound when the agent is constructed. Absent/null, `false`, or `{enabled: false}` disables admission without opening the source or its cache. Normal configuration/runtime adoption is required to change an existing agent's binding. Disabling admission is not deletion of historical context already stored in the transcript.

## Source-first admission and fallback

Each normal/staged input and each standard-loop queued steer/redirect gets a new occurrence ID. Queue capture happens before coalescing; queued snapshots become required for the request only when delivered. Delegated children inherit the immutable source binding, not the parent's expected-admission set, even without a memory provider.

Every new admission tries the current source first. A successful read computes the selected-body and complete-source SHA-256 values and a UTC verification time. The last successfully persisted complete copy is stored atomically, using a private directory and file, under:

```text
<hermes-home>/state/authored-turn-context/<binding-digest>.json
```

The digest binds the canonical home, source path and ordered section selection. Missing, unreadable or invalid current sources can use only a valid copy with that same binding. The copy's body hash, source hash and original verification time are retained; the new occurrence is marked degraded. Source recovery is checked again on the next admission. A successful current read may still proceed if cache persistence fails; this condition is reported separately. Explicit disable does not use fallback.

A degraded block identifies itself as a verified copy, including its original time and source hash. A lifecycle diagnostic and log report the source/cache fault once per continuous fault episode. Surface-callback failure does not discard an otherwise trustworthy copy. If neither source nor cache is trustworthy, admission raises `AuthoredContextUnavailable`; normal inference must not silently proceed without the required context.

This private local cache detects corruption and wrong bindings; it is not a signature or an authenticity boundary against an actor who can rewrite both the source and cache as the same OS user.

## Stable stored snapshot schema

`display_metadata.authored_turn_context` is a list. Monitoring consumers may continue to rely on `id`, `sha256` and `body`:

```json
{
  "id": "unique-admission-id",
  "sha256": "sha256-of-exact-utf8-body",
  "body": "selected Markdown sections",
  "version": 1,
  "binding": {
    "home": "/absolute/owning/home",
    "source": "/absolute/canonical/profile.md",
    "sections": ["Why we have this", "Us"]
  },
  "source_sha256": "sha256-of-complete-original-source-bytes",
  "verified_at": "ISO-8601-aware-UTC-time",
  "degraded": false,
  "cache_write_failed": false,
  "admission": {
    "timestamp": 1800000000.0,
    "message_uid": "durable-original-message-uid",
    "kind": "message",
    "turn_author": {"id": "transport-sender", "name": "Pat", "is_bot": false},
    "input_provenance": null,
    "display_kind": null,
    "source": "discord"
  }
}
```

`admission` is an optional backwards-compatible extension. New captures bind the snapshot to its own capture-time epoch, original message UID (null for nested entries), kind (`message` or `entry`), and structured author/source facts. Carrier-row timestamps can change at compaction; neither they nor `verified_at` replace this admission time. Opener `turn_author` is also persisted in surrounding display metadata, including exact-row early-flush backfill. Local TUI client submissions explicitly set `input_provenance: local_human`; platform names alone never establish this. Relay bots and hidden/internal inputs keep their distinct provenance.

`cache_write_failed` is optional. A cache fallback retains the original verification fields and sets `degraded: true`. An ID identifies an admission, not a unique body: identical bodies on separate inputs have different IDs. Replays and merges retain existing IDs. Consumers must preserve surrounding actor, display-kind, steering-entry and absorbed-message metadata when classifying human versus harness input; these snapshots are not new human speech.

String turns use the existing `api_content` sidecar. Multimodal turns receive a separate text part and retain the existing writer's storage shape (text projection for normal flush, structured JSON for in-place compaction). Early-persisted rows receive snapshot metadata through a narrow JSON merge matched by session, row ID, user role, active state and actual stored content. Unrelated metadata is preserved; unresolved snapshot backfills raise rather than guessing the latest text-matching row.

Consecutive-user merges, micro-compaction and in-flight replay/carrier absorption carry snapshots and existing message/absorbed IDs. The host summary-dispatch boundary retains every currently required snapshot, not only the most recent correction; it does not retain every historical profile copy indefinitely. SessionDB serialization renders snapshots on direct compaction writes as well as normal flushes, preserving the raw string `api_content` versus sentinel-encoded structured `content` distinction. Rendering is idempotent within each row. Historical replay uses its captured body rather than rereading today's profile.

## Request boundaries and limits

After request middleware and provider conversion, Chat Completions, Responses and native Anthropic requests are checked for the exact required blocks. Missing or altered blocks raise before inference. The check does not establish model compliance with the text or live provider delivery.

Codex app-server openers carry the rendered snapshot through the real session conversion to `turn/start`. Isolated tests cover that boundary with a fake external RPC client, not an actual Codex subprocess/backend. With this setting enabled, native mid-turn steer/redirect returns `false` rather than sending an unverified omission; callers must use their normal queued-turn fallback. The inline TUI text-queue fallback is tested through accept-time persistence, FIFO drain, real `AIAgent.run_conversation`, real session conversion, `turn/start` parameters and monitor readback using a fake external RPC client. With authored delivery enabled, repeated queued words remain separate occurrences; source/UID/timestamp bindings survive drain without recapture. Queued-only evidence remains unknown until a dispatch copy exists. A real native subprocess/backend remains untested.

MoA is explicitly held before its preparer runs while authored delivery is enabled: its inner requests do not have the verified delivery contract in this slice. Full-compaction preservation is tested for an opener plus two delivered corrections through repeated host summary dispatch, SQLite archive/compaction and resume. The summary provider is mocked. Isolated compute-host forwarding, classic terminal/one-shot id-less provenance, queue recovery after process death, media-bearing native queue fallback and other surface-specific queue implementations are excluded from this acceptance claim; legacy unbound provenance remains unknown. These are review/adoption holds, not implied coverage. If any currently required block is lost, the final request check stops the request rather than accepting partial delivery.

## Skill-context middleware host contract

Successful unchanged `skill_view` dedup responses retain `_source_path`. Both direct `model_tools` dispatch and agent-owned tool-execution middleware pass `enabled_tools` as the actual call's effective grant tuple (empty when no grant is supplied). Consumers must use that tuple, not process-global discovery/history, to decide whether memory access is permitted. A new consumer still needs an integrated test against both host paths; the host changes alone do not prove plugin enrichment.
