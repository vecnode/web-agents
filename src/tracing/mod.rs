use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

fn parse_bool_env(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default,
        },
        Err(_) => default,
    }
}

fn default_metrics_file() -> String {
    "metrics/timings.jsonl".to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TracingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_metrics_file")]
    pub metrics_file: String,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            metrics_file: default_metrics_file(),
        }
    }
}

impl TracingConfig {
    pub fn from_env() -> Self {
        Self {
            enabled: parse_bool_env("AMS_TRACING_ENABLED", false),
            metrics_file: std::env::var("AMS_TRACING_FILE")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(default_metrics_file),
        }
    }

    pub fn metrics_path(&self) -> PathBuf {
        PathBuf::from(self.metrics_file.trim())
    }
}

#[derive(Clone, Debug, Default)]
pub struct InferenceTraceContext {
    pub source: String,
    pub experiment_id: Option<String>,
    pub run_id: Option<String>,
    pub node_global_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceTimingEvent {
    pub event_type: String,
    pub timestamp: String,
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experiment_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_global_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub t_start: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub t_first_token: Option<String>,
    pub t_end: String,
    pub duration_ms: u128,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<u128>,
    pub input_chars: usize,
    pub output_chars: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_token_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates_token_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_token_count: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TurnTimingEvent {
    pub event_type: String,
    pub timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experiment_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    pub loop_key_node_id: usize,
    pub turn_index: usize,
    pub speaker_id: usize,
    pub speaker_name: String,
    pub receiver_id: usize,
    pub receiver_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gap_ms: Option<u128>,
}

pub trait MetricsSink: Send + Sync {
    fn emit_inference(&self, event: InferenceTimingEvent);
    fn emit_turn(&self, event: TurnTimingEvent);
}

#[derive(Default)]
pub struct NoopMetricsSink;

impl MetricsSink for NoopMetricsSink {
    fn emit_inference(&self, _event: InferenceTimingEvent) {}
    fn emit_turn(&self, _event: TurnTimingEvent) {}
}

pub struct FileMetricsSink {
    writer: Mutex<BufWriter<File>>,
}

impl FileMetricsSink {
    pub fn new(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("create metrics dir {}", parent.display()))?;
            }
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("open metrics sink at {}", path.display()))?;
        Ok(Self {
            writer: Mutex::new(BufWriter::new(file)),
        })
    }

    fn emit_json_line<T: Serialize>(&self, event: &T) -> Result<()> {
        let line = serde_json::to_string(event).context("serialize metrics event")?;
        let mut guard = self.writer.lock().unwrap();
        writeln!(guard, "{line}").context("write metrics line")?;
        guard.flush().ok();
        Ok(())
    }
}

impl MetricsSink for FileMetricsSink {
    fn emit_inference(&self, event: InferenceTimingEvent) {
        if let Err(e) = self.emit_json_line(&event) {
            eprintln!("[Tracing] failed to emit inference event: {e}");
        }
    }

    fn emit_turn(&self, event: TurnTimingEvent) {
        if let Err(e) = self.emit_json_line(&event) {
            eprintln!("[Tracing] failed to emit turn event: {e}");
        }
    }
}

pub fn build_metrics_sink(config: &TracingConfig) -> Arc<dyn MetricsSink> {
    if !config.enabled {
        return Arc::new(NoopMetricsSink);
    }
    let path = config.metrics_path();
    match FileMetricsSink::new(path.as_path()) {
        Ok(sink) => Arc::new(sink),
        Err(e) => {
            eprintln!("[Tracing] disabled because sink init failed: {e}");
            Arc::new(NoopMetricsSink)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FileMetricsSink, InferenceTimingEvent, MetricsSink, TracingConfig};
    use std::fs;

    #[test]
    fn tracing_config_default_is_disabled() {
        let cfg = TracingConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.metrics_file, "metrics/timings.jsonl");
    }

    #[test]
    fn file_sink_writes_jsonl() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("timings.jsonl");
        let sink = FileMetricsSink::new(path.as_path()).expect("sink");

        sink.emit_inference(InferenceTimingEvent {
            event_type: "inference_timing".to_string(),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            source: "test".to_string(),
            experiment_id: Some("exp_abc".to_string()),
            run_id: Some("run_123".to_string()),
            node_global_id: Some("node-1".to_string()),
            model: Some("glm-4.7-flash:latest".to_string()),
            success: true,
            error: None,
            t_start: "2026-01-01T00:00:00Z".to_string(),
            t_first_token: Some("2026-01-01T00:00:00Z".to_string()),
            t_end: "2026-01-01T00:00:01Z".to_string(),
            duration_ms: 1000,
            ttft_ms: Some(250),
            input_chars: 12,
            output_chars: 24,
            prompt_token_count: Some(10),
            candidates_token_count: Some(20),
            total_token_count: Some(30),
        });

        let raw = fs::read_to_string(path).expect("read metrics file");
        assert_eq!(raw.lines().count(), 1);
        let v: serde_json::Value = serde_json::from_str(raw.trim()).expect("valid json line");
        assert_eq!(v["event_type"], "inference_timing");
        assert_eq!(v["source"], "test");
    }
}