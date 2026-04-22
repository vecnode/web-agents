# arpsci-dashboard

Agents Research Platform for HCI and Cognitive Sciences (Dashboard).

- Multi-agent conversations
- Interactive prompt design
- Reproducibility of inference
- Local and field-first architecture



### Dependencies


- rust-adk
- eframe
- egui-phosphor


### Building

```sh
# One-time vault: interactive prompt writes `runs/.master_hash` (PHC Argon2id hash for the password gate)
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
```

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
- Captured fields include `t_start`, `t_first_token` (when streaming yields text), `t_end`, `duration_ms`, and `ttft_ms`.
- Inter-turn pacing is also recorded (`turn_timing`) with `gap_ms` between turns.
- Default output file: `metrics/timings.jsonl`.

Configuration options:

- UI: Settings > Reproducibility > Timing and Metrics
    - Enable/disable metrics capture (global app switch)
    - Set output JSONL file path

Sample JSONL records:

```json
{"event_type":"inference_timing","source":"dialogue.turn","duration_ms":1289,"ttft_ms":214}
{"event_type":"turn_timing","turn_index":4,"speaker_name":"Agent A","receiver_name":"Agent B","gap_ms":411}
```


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
