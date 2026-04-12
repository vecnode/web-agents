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



