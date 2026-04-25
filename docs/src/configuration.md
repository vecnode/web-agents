# Configuration Reference

## Core environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Base URL for Ollama API |
| `OLLAMA_MODEL` | empty | Selected model override |
| `AMS_OLLAMA_CONTEXT_WINDOW` | unset | Optional context window (`num_ctx`) |
| `AMS_WEB_ENABLED` | `false` | Enables embedded Rocket server |
| `AMS_WEBHOOKS_ENABLED` | `false` | Enables outbound webhook POSTs (independent of Rocket) |
| `AMS_WEB_ADDRESS` | `127.0.0.1` | Bind address for embedded server |
| `AMS_WEB_PORT` | `8000` | Bind port for embedded server |
| `AMS_CONVERSATION_HTTP_STREAM_ENABLED` | `false` | Enables conversation start/turn/end HTTP streaming |
| `AMS_CHAT_STREAM_ENABLED` | `true` | Enables forwarding turn-by-turn messages into Overview chat |
| `AMS_AIR_GAP` | `false` | Blocks non-loopback outbound HTTP |
| `AMS_ALLOW_LOCAL_OLLAMA` | `true` | Allows loopback Ollama in air-gap mode |
| `CONVERSATION_HTTP_ENDPOINT` | `http://localhost:3000/` | Webhook endpoint for conversation events |
| `AMS_LOG_PLAY_PLAN` | `false` | Logs resolved play plan before run |
| `AMS_CONVERSATION_GROUP_SIZE` | `2` | Number of workers per conversation loop |
| `AMS_METRICS_FILE` | `metrics/timings.jsonl` | Metrics JSONL output path |
| `AMS_MASTER_HASH` | unset | Master password Argon2id PHC hash |
| `AMS_ARGON2_M_KIB` | `65536` | Vault Argon2 memory cost |
| `AMS_ARGON2_T` | `3` | Vault Argon2 time cost |
| `AMS_ARGON2_P` | auto (`1..4`) | Vault Argon2 parallelism |
| `AMS_SKIP_VAULT` | `false` | Disables vault gate in development |

## Notes

- Metrics recording is enabled by default.
- Disabling the vault is intended only for local development.
- Air-gap mode and local Ollama allowance are independent toggles.
- `AMS_WEB_ENABLED` starts Rocket only; it no longer enables outbound conversation webhooks.
