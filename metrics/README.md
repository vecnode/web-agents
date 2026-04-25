This folder stores local metrics/tracing artifacts for offline analysis.

Current tracing output:

- `timings.jsonl` (default)

Each line is one JSON object (`JSONL`) and may include:

- `inference_timing`: per Ollama call timings (`t_start`, `t_first_token`, `t_end`, `duration_ms`, `ttft_ms`)
- `turn_timing`: per dialogue turn pacing (`gap_ms` between turns)

## Generate a Compact Report

Use the built-in reporter binary to summarize model performance and run behavior:

```sh
# default input: metrics/timings.jsonl
cargo run --bin timings_report

# custom file
cargo run --bin timings_report -- metrics/timings.jsonl
```

The report includes:

- per-model latency summary (`avg`, `p50`, `p95`)
- success, stopped, and error rates
- average token usage by model
- turn gap summary
- run-to-model mapping
