# arp 

Agents Research Platform for HCI and Cognitive Sciences.


### Overview

- Multi-agent conversations
- Prompt design and injection
- Reproducibility of inference
- Local and field-first architecture



### Building

```sh
# Development: Builds to 'target/debug/'
cargo run

# Distribution: Builds to 'target/release/'
cargo build --release
```

### Current Architecture

- Agents are rows: **Manager**, **Worker**, **Evaluator**, **Researcher**, plus **Topic** presets.
- Rows wire via dropdowns (e.g. worker→manager/topic).
- **Start** saves a manifest and runs Ollama loops; workers with a topic are **paired in id order** (two workers ⇒ dialogue, one ⇒ solo loop).
- **Evaluators**: post-line sidecar on each utterance when active. **Researchers**: pre-turn only (injection + HTTP) for the worker selected under Injection—no second post-line researcher pass. **Stop** ends all loops.

### Communication

- JSON `POST` to `CONVERSATION_HTTP_ENDPOINT` (default `http://localhost:3000/`).
- Conversation events: `sender_id`, `receiver_id`, `topic`, `message`, …
- Evaluator/researcher: `evaluator_name`, sentiment (researcher: `sentiment` like `references:<topic>`).
- RFC3339 UTC timestamps; runs may include `experiment_id`, `run_id`, `manifest_version`.

### Reproducible Runs

- **Start** writes `runs/<experiment_id>/<run_id>/manifest.json` (`manifest_version = "2.0.0"`).
- Manifest: runtime settings + **flat agent snapshot** (links live in each node `config`, no edge list).
- Settings: export manifest, load manifest + run (read-only), bundle zip.


### Dependencies

- rust-adk
- eframe
- egui-phosphor

