# Architecture

ARP is a local-first Rust desktop application built with eframe/egui. It coordinates multi-agent conversations through Ollama, persists runs as reproducible artifacts, and exposes optional HTTP endpoints for integration.

## High-level modules

- `src/main.rs`: process startup, Tokio runtime, optional embedded web server.
- `src/ui/mod.rs`: main app shell and tab routing.
- `src/agents/`: run orchestration, conversation loops, sidecars, and node graph state.
- `src/ollama/`: model listing and streaming inference.
- `src/run/`: run manifest and append-only event ledger.
- `src/python/`: managed Python virtual environments and task execution.
- `src/vault.rs`: master password gate and encrypted in-memory vault blob.
- `src/web/mod.rs`: HTTP policy guardrails and optional outbound/inbound web hooks.

## Data roots

- `runs/`: manifests, event ledgers, and per-run outputs.
- `runtimes/python/`: Python runtime registry and virtual environments.
- `metrics/timings.jsonl`: inference and turn timing events.

## Core object wiring

The app state is centralized in `AMSAgents`, created by the UI shell.

```rust
pub struct AMSAgents {
    pub(crate) rt_handle: Handle,
    pub(crate) app_state: Arc<AppState>,
    pub(crate) selected_ollama_model: String,
    pub(crate) ollama_host: String,
    pub(crate) event_ledger: Option<Arc<EventLedger>>,
    pub(crate) nodes_panel: nodes_panel::NodesPanelState,
    pub(crate) chat_turn_tx: Option<std::sync::mpsc::Sender<AgentChatEvent>>,
    pub(crate) chat_turn_rx: Option<std::sync::mpsc::Receiver<AgentChatEvent>>,
}
```

This design keeps runtime handles, run context, and UI-consumed event channels in one place, while feature modules remain separated by concern.
