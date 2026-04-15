use adk_agent::LlmAgentBuilder;
use adk_core::Content;
use adk_model::ollama::{OllamaConfig, OllamaModel};
use adk_runner::{Runner, RunnerConfig};
use adk_session::{CreateRequest, InMemorySessionService, SessionService};
use anyhow::Result;
use futures_util::StreamExt;
use serde::Deserialize;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use std::time::Instant;

const APP_NAME: &str = "ams-agents";
const USER_ID: &str = "user1";
const AGENT_NAME: &str = "local-assistant";

#[derive(Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaTagModel>,
}

#[derive(Deserialize)]
struct OllamaTagModel {
    name: String,
}

pub(crate) fn normalize_ollama_host(input: &str) -> String {
    let s = input.trim();
    if s.is_empty() {
        return "http://127.0.0.1:11434".to_string();
    }
    let s = s.trim_end_matches('/').to_string();
    if s.contains("://") {
        s
    } else {
        format!("http://{}", s)
    }
}

pub(crate) struct RunnerContext {
    pub(crate) runner: Runner,
    pub(crate) session_id: String,
    pub(crate) model_name: String,
}

pub(crate) struct StreamingResult {
    pub(crate) response: String,
    pub(crate) ttft: Option<Duration>,
    pub(crate) usage: Option<TokenUsage>,
}

#[derive(Clone, Debug)]
pub(crate) struct TokenUsage {
    pub(crate) prompt_token_count: u64,
    pub(crate) candidates_token_count: u64,
    pub(crate) total_token_count: u64,
}

pub(crate) async fn fetch_models(ollama_host: &str) -> Result<Vec<String>> {
    let url = std::env::var("OLLAMA_TAGS_URL")
        .unwrap_or_else(|_| format!("{}/api/tags", normalize_ollama_host(ollama_host)));
    crate::web::guard_ollama_request(&url)?;
    let client = reqwest::Client::new();
    let response = client.get(url).send().await?.error_for_status()?;
    let tags = response.json::<OllamaTagsResponse>().await?;
    let mut models: Vec<String> = tags.models.into_iter().map(|m| m.name).collect();
    models.sort();
    models.dedup();
    Ok(models)
}

pub(crate) async fn build_runner_context(
    ollama_host: &str,
    instruction: &str,
    limit_token: bool,
    num_predict: &str,
    model_override: Option<&str>,
) -> Result<RunnerContext> {
    let model_name = resolve_model_name(model_override);
    let host = normalize_ollama_host(ollama_host);
    crate::web::guard_ollama_request(&host)?;
    let mut config = OllamaConfig::with_host(host, &model_name);
    if limit_token {
        if let Ok(num) = num_predict.parse::<u32>() {
            config.num_ctx = Some(num);
        }
    }
    let model = OllamaModel::new(config)?;

    let agent = LlmAgentBuilder::new(AGENT_NAME)
        .description("A helpful local assistant")
        .model(Arc::new(model))
        .instruction(instruction)
        .build()?;

    let session_service = Arc::new(InMemorySessionService::new());
    let session = session_service
        .create(CreateRequest {
            app_name: APP_NAME.to_string(),
            user_id: USER_ID.to_string(),
            session_id: None,
            state: std::collections::HashMap::new(),
        })
        .await?;

    let runner = Runner::new(RunnerConfig {
        app_name: APP_NAME.to_string(),
        agent: Arc::new(agent),
        session_service,
        artifact_service: None,
        memory_service: None,
        plugin_manager: None,
        run_config: None,
        compaction_config: None,
    })?;

    Ok(RunnerContext {
        runner,
        session_id: session.id().to_string(),
        model_name,
    })
}

pub(crate) async fn run_prompt_streaming(
    runner_ctx: RunnerContext,
    input: &str,
    print_response_prefix: bool,
    stop_epoch: Option<(Arc<AtomicU64>, u64)>,
) -> Result<StreamingResult> {
    let stream_started = Instant::now();
    let mut first_token_seen: Option<Duration> = None;
    let mut usage: Option<TokenUsage> = None;
    let user_content = Content::new("user").with_text(input);
    let mut stream = runner_ctx
        .runner
        .run(USER_ID.to_string(), runner_ctx.session_id, user_content)
        .await?;

    let mut response_parts = Vec::new();
    if print_response_prefix {
        print!("Response: ");
        let _ = std::io::stdout().flush();
    }

    while let Some(event_result) = stream.next().await {
        if let Some((ref epoch, caught)) = stop_epoch {
            if epoch.load(Ordering::SeqCst) != caught {
                println!("\n[Ollama inference stopped by user]");
                return Err(anyhow::anyhow!(super::OLLAMA_STOPPED_MSG));
            }
        }
        match event_result {
            Ok(event) => {
                if let Some(content) = event.llm_response.content.as_ref() {
                    for part in &content.parts {
                        if let adk_core::Part::Text { text } = part {
                            if first_token_seen.is_none() {
                                first_token_seen = Some(stream_started.elapsed());
                            }
                            print!("{}", text);
                            let _ = std::io::stdout().flush();
                            response_parts.push(text.clone());
                        }
                    }
                }

                if event.llm_response.turn_complete {
                    if let Some(usage_meta) = &event.llm_response.usage_metadata {
                        usage = Some(TokenUsage {
                            prompt_token_count: usage_meta.prompt_token_count as u64,
                            candidates_token_count: usage_meta.candidates_token_count as u64,
                            total_token_count: usage_meta.total_token_count as u64,
                        });
                        println!(
                            "\n[Tokens: prompt={}, candidates={}, total={}]",
                            usage_meta.prompt_token_count,
                            usage_meta.candidates_token_count,
                            usage_meta.total_token_count
                        );
                    }
                }
            }
            Err(e) => {
                println!("\n[Stream error: {}]", e);
                return Err(anyhow::anyhow!("Stream error: {}", e));
            }
        }
    }

    println!();
    Ok(StreamingResult {
        response: response_parts.join(""),
        ttft: first_token_seen,
        usage,
    })
}

pub(crate) fn print_context_preview(input_text: &str) {
    if input_text.is_empty() {
        return;
    }

    let max_bytes = 200;
    if input_text.len() > max_bytes {
        let mut truncate_at = max_bytes;
        while !input_text.is_char_boundary(truncate_at) && truncate_at > 0 {
            truncate_at -= 1;
        }
        println!("Context: {}...", &input_text[..truncate_at]);
    } else {
        println!("Context: {}", input_text);
    }
}

fn resolve_model_name(model_override: Option<&str>) -> String {
    if let Some(model) = model_override {
        if !model.trim().is_empty() {
            return model.to_string();
        }
    }
    std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "glm-4.7-flash:latest".to_string())
}
