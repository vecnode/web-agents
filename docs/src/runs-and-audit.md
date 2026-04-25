# Runs and Audit Trail

ARP stores each execution as a run bundle under `runs/<experiment_id>/<run_id>/`.

## Run manifest

`manifest.json` captures runtime options and graph snapshot.

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunManifest {
    pub manifest_version: String,
    pub app_name: String,
    pub app_version: String,
    pub created_at: String,
    pub experiment_id: String,
    pub run_id: String,
    pub graph_signature: String,
    pub runtime: RunRuntimeSettings,
    pub graph: GraphSnapshot,
}
```

## Event ledger

`events.jsonl` is append-only and hash-aware. Every event stores input and output hashes plus payload.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventEnvelope {
    pub experiment_id: String,
    pub run_id: String,
    pub event_id: u64,
    pub event_type: String,
    pub timestamp: String,
    pub input_hash: String,
    pub output_hash: String,
    pub payload: serde_json::Value,
}
```

On finalization, a `summary.json` file is generated with:

- Event counters by type.
- HTTP transport success and failure counts.
- SHA-256 of `manifest.json` and `events.jsonl` when present.

## Overview chat persistence

The Overview tab stores conversations in SQLite through `Store` (`src/ui/overview_chat/store.rs`) and writes audit records through `AuditHandle` (`src/ui/overview_chat/audit.rs`).

This provides two persistence layers:

1. Structured room and message history for UI replay.
2. Append-only audit lines for external review.
