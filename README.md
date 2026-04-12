# arp

Agents Research Platform for HCI and Cognitive Sciences.

### Overview

- Multi-agent conversations
- Prompt design and injection
- Reproducibility of inference
- Local and field-first architecture


### Dependencies

- rust-adk
- eframe
- egui-phosphor


### Building

```sh
# One-time vault: interactive prompt writes `runs/.master_hash` (PHC Argon2id hash for the password gate)
cargo run --bin gen_master_hash

# Development: run the application (`target/debug/`)
cargo run

# Distribution: build the application ('target/release/')
cargo build --release
```


### Communication and Security

- **Vault:** Argon2id master-password gate with non-trivial defaults (`m=65536 KiB`, `t=3`, `p<=4`); configure with `runs/.master_hash` (first line) or `AMS_MASTER_HASH`. Tune with `AMS_ARGON2_M_KIB`, `AMS_ARGON2_T`, `AMS_ARGON2_P`. Optional: Set `AMS_SKIP_VAULT=1` only for local development (disables the gate).
- **Outbound HTTP:** JSON bodies are `POST`ed to `CONVERSATION_HTTP_ENDPOINT` (default `http://localhost:3000/`).
- **Conversation payloads** include fields such as `sender_id`, `receiver_id`, `topic`, and `message` (plus other event metadata as emitted).
- **Sidecars:** evaluators attach `evaluator_name` and sentiment; researchers use `sentiment` (e.g. `references:<topic>`) on the configured injection path.
- **Time and run identity:** timestamps are RFC3339 UTC; runs may carry `experiment_id`, `run_id`, and `manifest_version`.

