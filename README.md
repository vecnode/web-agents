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

- `Agent` nodes: `Agent Manager`, `Agent Worker`, `Agent Evaluator`, `Agent Researcher`
- `Tool` nodes: `Topic` (provides topic preset + topic text to Workers)
- `Output` nodes: `Conversation` (terminal node that runs monologue/dialogue)

Common graph flow:
- `Manager -> Worker <- Topic`, then `Worker -> Conversation`.  

If one Worker is connected to a Conversation, it runs as monologue.
If two Workers are connected, it runs as dialogue.

Evaluator/Researcher run independently when connected to a Worker.
- `Run Graph` stops current loops and starts valid conversation paths again.
- `Stop Graph` stops all active conversation loops.


### Communication

This app sends JSON `POST` events for conversation turns, evaluator outputs, and researcher outputs.
Set `CONVERSATION_HTTP_ENDPOINT` to your receiver URL (default: `http://localhost:3000/`).

- `conversation` payload: `{ sender_id, sender_name, receiver_id, receiver_name, topic, message, timestamp }`
- `evaluator` payload: `{ evaluator_name, sentiment, message, timestamp }`
- `researcher` payload: same evaluator shape, where `sentiment = "references:<topic>"`.

To build a receiver app, expose one `POST` endpoint, parse these fields, and branch by keys:
if payload has `sender_id` => conversation event; if payload has `evaluator_name` => evaluator/researcher event.
All timestamps are RFC3339 UTC strings.

### Reproducible Runs

Each `Run Graph` execution now creates a run manifest for reproducibility:

- Path: `runs/<experiment_id>/<run_id>/manifest.json`
- Schema version: `manifest_version = "1.0.0"`
- Manifest captures runtime settings and a canonical node graph snapshot
- Outbound HTTP events include `experiment_id`, `run_id`, and `manifest_version`

Settings panel includes:

- `Export Manifest`: write the current manifest to a custom path
- `Run From Manifest`: load a manifest in read-only replay mode and execute it


### Dependencies

- rust-adk
- eframe
- egui-phosphor
- egui-snarl = { path = "crates/egui-snarl" }

