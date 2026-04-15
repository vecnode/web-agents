use anyhow::Result;
use std::time::SystemTime;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::run::manifest::now_rfc3339_utc;
use crate::tracing::{InferenceTimingEvent, InferenceTraceContext, MetricsSink};

mod client;

/// Sentinel error message when the user stops inference or starts a new run.
pub const OLLAMA_STOPPED_MSG: &str = "ollama inference stopped";

/// When live epoch differs from `captured_epoch`, streaming stops cooperatively (user Stop).
pub type OllamaStopEpoch = (Arc<AtomicU64>, u64);

fn format_rfc3339(ts: SystemTime) -> String {
    chrono::DateTime::<chrono::Utc>::from(ts).to_rfc3339()
}

pub async fn fetch_ollama_models(ollama_host: &str) -> Result<Vec<String>> {
    client::fetch_models(ollama_host).await
}

pub async fn send_to_ollama(
    ollama_host: &str,
    instruction: &str,
    input: &str,
    limit_token: bool,
    num_predict: &str,
    model_override: Option<&str>,
    stop_epoch: Option<OllamaStopEpoch>,
    metrics_sink: Arc<dyn MetricsSink>,
    trace_context: InferenceTraceContext,
) -> Result<String> {
    let t_start_wall = SystemTime::now();
    let t_start = std::time::Instant::now();

    if let Some((epoch, caught)) = &stop_epoch {
        if epoch.load(Ordering::SeqCst) != *caught {
            let t_end_wall = SystemTime::now();
            metrics_sink.emit_inference(InferenceTimingEvent {
                event_type: "inference_timing".to_string(),
                timestamp: now_rfc3339_utc(),
                source: trace_context.source.clone(),
                experiment_id: trace_context.experiment_id.clone(),
                run_id: trace_context.run_id.clone(),
                node_global_id: trace_context.node_global_id.clone(),
                model: model_override.map(|m| m.to_string()),
                success: false,
                error: Some(OLLAMA_STOPPED_MSG.to_string()),
                t_start: format_rfc3339(t_start_wall),
                t_first_token: None,
                t_end: format_rfc3339(t_end_wall),
                duration_ms: t_start.elapsed().as_millis(),
                ttft_ms: None,
                input_chars: input.chars().count(),
                output_chars: 0,
                prompt_token_count: None,
                candidates_token_count: None,
                total_token_count: None,
            });
            return Err(anyhow::anyhow!(OLLAMA_STOPPED_MSG));
        }
    }

    let runner_ctx = match client::build_runner_context(
        ollama_host,
        instruction,
        limit_token,
        num_predict,
        model_override,
    )
    .await
    {
        Ok(ctx) => ctx,
        Err(e) => {
            let t_end_wall = SystemTime::now();
            metrics_sink.emit_inference(InferenceTimingEvent {
                event_type: "inference_timing".to_string(),
                timestamp: now_rfc3339_utc(),
                source: trace_context.source.clone(),
                experiment_id: trace_context.experiment_id.clone(),
                run_id: trace_context.run_id.clone(),
                node_global_id: trace_context.node_global_id.clone(),
                model: model_override.map(|m| m.to_string()),
                success: false,
                error: Some(e.to_string()),
                t_start: format_rfc3339(t_start_wall),
                t_first_token: None,
                t_end: format_rfc3339(t_end_wall),
                duration_ms: t_start.elapsed().as_millis(),
                ttft_ms: None,
                input_chars: input.chars().count(),
                output_chars: 0,
                prompt_token_count: None,
                candidates_token_count: None,
                total_token_count: None,
            });
            return Err(e);
        }
    };
    client::print_context_preview(input);

    let model_name = Some(runner_ctx.model_name.clone());
    match client::run_prompt_streaming(runner_ctx, input, false, stop_epoch).await {
        Ok(streaming) => {
            let t_end_wall = SystemTime::now();
            let first_token_at = streaming
                .ttft
                .and_then(|d| t_start_wall.checked_add(d))
                .map(format_rfc3339);
            let ttft_ms = streaming.ttft.map(|d| d.as_millis());
            let output_chars = streaming.response.chars().count();
            let token_usage = streaming.usage.clone();
            metrics_sink.emit_inference(InferenceTimingEvent {
                event_type: "inference_timing".to_string(),
                timestamp: now_rfc3339_utc(),
                source: trace_context.source,
                experiment_id: trace_context.experiment_id,
                run_id: trace_context.run_id,
                node_global_id: trace_context.node_global_id,
                model: model_name,
                success: true,
                error: None,
                t_start: format_rfc3339(t_start_wall),
                t_first_token: first_token_at,
                t_end: format_rfc3339(t_end_wall),
                duration_ms: t_start.elapsed().as_millis(),
                ttft_ms,
                input_chars: input.chars().count(),
                output_chars,
                prompt_token_count: token_usage.as_ref().map(|u| u.prompt_token_count),
                candidates_token_count: token_usage
                    .as_ref()
                    .map(|u| u.candidates_token_count),
                total_token_count: token_usage.as_ref().map(|u| u.total_token_count),
            });
            Ok(streaming.response)
        }
        Err(e) => {
            let t_end_wall = SystemTime::now();
            metrics_sink.emit_inference(InferenceTimingEvent {
                event_type: "inference_timing".to_string(),
                timestamp: now_rfc3339_utc(),
                source: trace_context.source,
                experiment_id: trace_context.experiment_id,
                run_id: trace_context.run_id,
                node_global_id: trace_context.node_global_id,
                model: model_name,
                success: false,
                error: Some(e.to_string()),
                t_start: format_rfc3339(t_start_wall),
                t_first_token: None,
                t_end: format_rfc3339(t_end_wall),
                duration_ms: t_start.elapsed().as_millis(),
                ttft_ms: None,
                input_chars: input.chars().count(),
                output_chars: 0,
                prompt_token_count: None,
                candidates_token_count: None,
                total_token_count: None,
            });
            Err(e)
        }
    }
}

pub async fn test_ollama(
    ollama_host: &str,
    model_override: Option<&str>,
    metrics_sink: Arc<dyn MetricsSink>,
) -> Result<String> {
    let runner_ctx = client::build_runner_context(
        ollama_host,
        "You are a helpful assistant running locally via Ollama.",
        false,
        "",
        model_override,
    )
    .await?;
    let input = "Hello, how are you?";
    println!("Input: {}", input);
    let model_name = runner_ctx.model_name.clone();
    let t_start_wall = SystemTime::now();
    let t_start = std::time::Instant::now();
    match client::run_prompt_streaming(runner_ctx, input, true, None).await {
        Ok(streaming) => {
            let t_end_wall = SystemTime::now();
            let first_token_at = streaming
                .ttft
                .and_then(|d| t_start_wall.checked_add(d))
                .map(format_rfc3339);
            metrics_sink.emit_inference(InferenceTimingEvent {
                event_type: "inference_timing".to_string(),
                timestamp: now_rfc3339_utc(),
                source: "settings.test_ollama".to_string(),
                experiment_id: None,
                run_id: None,
                node_global_id: None,
                model: Some(model_name),
                success: true,
                error: None,
                t_start: format_rfc3339(t_start_wall),
                t_first_token: first_token_at,
                t_end: format_rfc3339(t_end_wall),
                duration_ms: t_start.elapsed().as_millis(),
                ttft_ms: streaming.ttft.map(|d| d.as_millis()),
                input_chars: input.chars().count(),
                output_chars: streaming.response.chars().count(),
                prompt_token_count: streaming.usage.as_ref().map(|u| u.prompt_token_count),
                candidates_token_count: streaming
                    .usage
                    .as_ref()
                    .map(|u| u.candidates_token_count),
                total_token_count: streaming.usage.as_ref().map(|u| u.total_token_count),
            });
            Ok(streaming.response)
        }
        Err(e) => {
            let t_end_wall = SystemTime::now();
            metrics_sink.emit_inference(InferenceTimingEvent {
                event_type: "inference_timing".to_string(),
                timestamp: now_rfc3339_utc(),
                source: "settings.test_ollama".to_string(),
                experiment_id: None,
                run_id: None,
                node_global_id: None,
                model: Some(model_name),
                success: false,
                error: Some(e.to_string()),
                t_start: format_rfc3339(t_start_wall),
                t_first_token: None,
                t_end: format_rfc3339(t_end_wall),
                duration_ms: t_start.elapsed().as_millis(),
                ttft_ms: None,
                input_chars: input.chars().count(),
                output_chars: 0,
                prompt_token_count: None,
                candidates_token_count: None,
                total_token_count: None,
            });
            Err(e)
        }
    }
}
