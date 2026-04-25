# arpsci-dashboard

Under heavy development. 

Agents Research Platform for HCI and Cognitive Sciences (Dashboard).

- Multi-agent conversations
- Interactive prompt design
- Reproducibility of inference
- Local and field-first architecture


### Building

```sh
# One-time vault: Writes `runs/.master_hash` (PHC Argon2id hash)
# Ubuntu 22
cargo run --bin gen_master_hash

# Windows 11
$env:CARGO_TARGET_DIR="target-hash-win11"; cargo run --bin gen_master_hash

# Development: run the application (`target/debug/`)
cargo run

# Development: run with embedded web server (`target/debug/`)
AMS_WEB_ENABLED=true cargo run
# http://127.0.0.1:8000/api/health
# http://127.0.0.1:8000/api/outgoing-http-log

# Distribution: build the application ('target/release/')
cargo build --release

cargo run --bin timings_report -- metrics/timings.jsonl


```

### uv Tests

```sh
Being done
```

### Dependencies

- eframe / egui
- catppuccin-egui
- egui-phosphor
- tokio
- reqwest
- rocket
- serde / serde_json
- rusqlite
- argon2 + hkdf + chacha20poly1305


### Reproducibility

- The `./runs/` folder is the application persisted state and run history:
    1) a workspace snapshot you can load/save outside a run,
    2) per-experiment/per-run execution artifacts


### Security

- Vault: master password verification uses Argon2id.
- Vault key derivation + encryption: vault payload keys are derived separately from the master password (`Argon2id + HKDF-SHA256`) and encrypted with AEAD (`ChaCha20-Poly1305`) using random salt/nonce and versioned metadata.
- Outbound HTTP: JSON bodies are `POST`ed to `CONVERSATION_HTTP_ENDPOINT` (default `http://localhost:3000/`) unless air-gap mode is enabled.
- Air-gap mode: set `AMS_AIR_GAP=1` to block non-loopback outbound HTTP.



### Timing and Metrics

- Metrics capture is app-global and enabled by default.
- The app always records timing metrics unless you explicitly disable recording in Settings.
- Metrics are written to JSONL for offline research.
- Default output file: `metrics/timings.jsonl`.


### Python Runtimes

You can create and manage isolated Python virtual environments per experiment via the Python tab.

- Create a venv from any system interpreter, install packages into it, and destroy it when done.
- Each runtime is stored under `runtimes/python/{id}/` and tracked in `python_runtimes.json`.
- `pip freeze` is captured on creation for reproducibility; execution is fully traceable via the run ledger.

Install NumPy on custom venv:  

```python
import numpy as np
print(np.arange(6).reshape(2, 3))
```
