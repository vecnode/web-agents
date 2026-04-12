# arpsci-dashboard

Agents Research Platform for HCI and Cognitive Sciences (Dashboard).

### Overview

- Multi-agent conversations
- Prompt design and injection
- Reproducibility of inference
- Local and field-first architecture


### Main Dependencies

- rust-adk (adk-agent, adk-model, adk-runner, adk-session, adk-core)
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

- **Vault:** master password verification uses **Argon2id** with non-trivial defaults (`m=65536 KiB`, `t=3`, `p<=4`), persisted as PHC hash in `runs/.master_hash` (or `AMS_MASTER_HASH`). KDF tuning: `AMS_ARGON2_M_KIB`, `AMS_ARGON2_T`, `AMS_ARGON2_P`.
- **Vault key derivation + encryption:** vault payload keys are derived separately from the master password (`Argon2id + HKDF-SHA256`) and encrypted with AEAD (`ChaCha20-Poly1305`) using random salt/nonce and versioned metadata.
- **Outbound HTTP:** JSON bodies are `POST`ed to `CONVERSATION_HTTP_ENDPOINT` (default `http://localhost:3000/`) unless air-gap mode is enabled.
- **Air-gap mode:** set `AMS_AIR_GAP=1` to block non-loopback outbound HTTP; optional `AMS_ALLOW_LOCAL_OLLAMA=0` also blocks local Ollama requests. Blocked attempts are mirrored to the run ledger as `transport.http_blocked` events.
- **Conversation payloads** include fields such as `sender_id`, `receiver_id`, `topic`, and `message` (plus other event metadata as emitted).
- **Sidecars:** evaluators attach `evaluator_name` and sentiment; researchers use `sentiment` (e.g. `references:<topic>`) on the configured injection path.
- **Time and run identity:** timestamps are RFC3339 UTC; runs may carry `experiment_id`, `run_id`, and `manifest_version`.
- **Dev bypass:** set `AMS_SKIP_VAULT=1` only for local development to disable the vault gate.

### Python Runtimes

- **Contained venvs:** each runtime is an isolated Python virtual environment stored under `runtimes/python/{id}/`, created from a [`PythonRuntimeSpec`](src/python_runtime.rs) (base interpreter, requirements list, optional post-install commands).
- **Registry:** all runtimes are tracked in `runtimes/python/python_runtimes.json` with stable `pyrt_<hash>` ids, resolved Python version, and lifecycle state.
- **Reproducibility:** `pip freeze` output is written to `requirements.lock` on creation; a SHA-256 digest of that file is embedded in the run manifest as a `PythonRuntimeRef` for full dependency traceability.
- **Traceable execution:** Python tasks are launched via [`launch_task()`](src/python_runtime.rs) with `VIRTUAL_ENV`/`PATH` set by environment variables only (no global shell activation). `ARP_EXPERIMENT_ID`, `ARP_RUN_ID`, and `ARP_EVENTS_PATH` are injected automatically. Stdout/stderr are captured to `runs/{experiment_id}/{run_id}/python_tasks/{task_id}/`, alongside a `meta.json` sidecar.
- **Event ledger:** each task invocation emits `python_task.started` and `python_task.finished` events into the run ledger with entrypoint, args, exit code, and log file pointers.
- **Lifecycle:** runtimes progress through `Active` → `Deprecated` → `Deleted`. Physical deletion removes the venv directory but retains the manifest entry so historical runs remain traceable.
