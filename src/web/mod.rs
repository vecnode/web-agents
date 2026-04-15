use crate::run::event_ledger::EventLedger;
use crate::run::manifest::RunContext;
use anyhow::{Result, anyhow};
use rocket::figment::Figment;
use rocket::serde::json::Json;
use rocket::{Build, Config, Rocket, routes};
use reqwest::Url;
use serde::Serialize;
use serde::{Deserialize, Serialize as DeriveSerialize};
use std::net::IpAddr;
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::runtime::Handle;

static OUTGOING_HTTP_LOG: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
static HTTP_POLICY: OnceLock<RwLock<HttpPolicy>> = OnceLock::new();

fn outgoing_http_log() -> &'static Mutex<Vec<String>> {
    OUTGOING_HTTP_LOG.get_or_init(|| Mutex::new(Vec::new()))
}

fn trim_line(input: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for c in input.chars().take(max_chars) {
        out.push(c);
    }
    if input.chars().count() > max_chars {
        out.push_str("...");
    }
    out
}

fn push_outgoing_http_log_line(line: String) {
    let mut log = outgoing_http_log().lock().unwrap();
    log.push(line);
    if log.len() > 400 {
        let keep_from = log.len() - 400;
        log.drain(0..keep_from);
    }
}

pub fn get_outgoing_http_log_lines() -> Vec<String> {
    outgoing_http_log().lock().unwrap().clone()
}

#[derive(Clone, Copy, Debug)]
pub struct HttpPolicy {
    pub air_gap_enabled: bool,
    pub allow_local_ollama: bool,
}

impl Default for HttpPolicy {
    fn default() -> Self {
        Self {
            air_gap_enabled: false,
            allow_local_ollama: true,
        }
    }
}

#[derive(Clone, Debug)]
struct WebConfig {
    enabled: bool,
    address: String,
    port: u16,
}

impl WebConfig {
    fn from_env() -> Self {
        let enabled = parse_bool_env("AMS_WEB_ENABLED", false);
        let address = std::env::var("AMS_WEB_ADDRESS").unwrap_or_else(|_| "127.0.0.1".to_string());
        let port = std::env::var("AMS_WEB_PORT")
            .ok()
            .and_then(|v| v.parse::<u16>().ok())
            .unwrap_or(8000);

        Self {
            enabled,
            address,
            port,
        }
    }
}

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

impl HttpPolicy {
    pub fn from_env() -> HttpPolicy {
        HttpPolicy {
            air_gap_enabled: parse_bool_env("AMS_AIR_GAP", false),
            allow_local_ollama: parse_bool_env("AMS_ALLOW_LOCAL_OLLAMA", true),
        }
    }
}

pub fn set_policy(policy: HttpPolicy) {
    let cell = HTTP_POLICY.get_or_init(|| RwLock::new(policy));
    if let Ok(mut guard) = cell.write() {
        *guard = policy;
    }
}

pub fn current_policy() -> HttpPolicy {
    let cell = HTTP_POLICY.get_or_init(|| RwLock::new(HttpPolicy::from_env()));
    match cell.read() {
        Ok(guard) => *guard,
        Err(_) => HttpPolicy::default(),
    }
}

fn parse_url_like(input: &str) -> Result<Url> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("empty URL"));
    }
    if let Ok(url) = Url::parse(trimmed) {
        return Ok(url);
    }
    Url::parse(&format!("http://{trimmed}"))
        .map_err(|e| anyhow!("invalid URL '{trimmed}': {e}"))
}

fn is_loopback_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.parse::<IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(false)
}

fn log_blocked_http_event(
    attempted_url: &str,
    component: &str,
    reason: &str,
    ledger: Option<&Arc<EventLedger>>,
) {
    if let Some(l) = ledger {
        let _ = l.append_with_hashes(
            "transport.http_blocked",
            None,
            None,
            attempted_url,
            reason,
            serde_json::json!({
                "attempted_url": attempted_url,
                "component": component,
                "reason": reason,
            }),
        );
    }
}

pub fn guard_http_request(
    attempted_url: &str,
    component: &str,
    ledger: Option<&Arc<EventLedger>>,
) -> Result<()> {
    let policy = current_policy();
    if !policy.air_gap_enabled {
        return Ok(());
    }

    let parsed = parse_url_like(attempted_url)?;
    let host = parsed.host_str().unwrap_or_default().to_string();

    if is_loopback_host(&host) {
        return Ok(());
    }

    let reason = "AirGapPolicy";
    log_blocked_http_event(attempted_url, component, reason, ledger);
    Err(anyhow!(
        "air-gap mode blocked outbound HTTP to '{}' (component: {})",
        host,
        component
    ))
}

pub fn guard_ollama_request(ollama_url: &str) -> Result<()> {
    let policy = current_policy();
    if !policy.air_gap_enabled {
        return Ok(());
    }
    if !policy.allow_local_ollama {
        return Err(anyhow!(
            "air-gap mode blocked Ollama request (allow local Ollama is disabled)"
        ));
    }

    let parsed = parse_url_like(ollama_url)?;
    let host = parsed.host_str().unwrap_or_default().to_string();
    if is_loopback_host(&host) {
        return Ok(());
    }

    Err(anyhow!(
        "air-gap mode only allows loopback Ollama hosts; got '{}'",
        host
    ))
}

#[derive(DeriveSerialize, Deserialize, Debug)]
pub struct ConversationMessage {
    pub sender_id: usize,
    pub sender_name: String,
    pub receiver_id: usize,
    pub receiver_name: String,
    pub topic: String,
    pub message: String,
    pub timestamp: String,
    pub experiment_id: Option<String>,
    pub run_id: Option<String>,
    pub manifest_version: Option<String>,
}

pub async fn send_conversation_message(
    endpoint: &str,
    sender_id: usize,
    sender_name: &str,
    receiver_id: usize,
    receiver_name: &str,
    topic: &str,
    message: &str,
    run_context: Option<&RunContext>,
    ledger: Option<&Arc<EventLedger>>,
) -> Result<(), anyhow::Error> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let timestamp_str = chrono::DateTime::<chrono::Utc>::from_timestamp(timestamp as i64, 0)
        .unwrap()
        .to_rfc3339();

    let payload = ConversationMessage {
        sender_id,
        sender_name: sender_name.to_string(),
        receiver_id,
        receiver_name: receiver_name.to_string(),
        topic: topic.to_string(),
        message: message.to_string(),
        timestamp: timestamp_str,
        experiment_id: run_context.map(|c| c.experiment_id.clone()),
        run_id: run_context.map(|c| c.run_id.clone()),
        manifest_version: run_context.map(|c| c.manifest_version.clone()),
    };

    let payload_json = serde_json::to_string(&payload)?;

    if let Err(e) = guard_http_request(endpoint, "conversation", ledger) {
        push_outgoing_http_log_line(format!(
            "BLOCKED {} | conversation | {} -> {} | {}",
            endpoint,
            sender_name,
            receiver_name,
            trim_line(message, 90),
        ));
        return Err(e);
    }

    push_outgoing_http_log_line(format!(
        "POST {} | conversation | {} -> {} | {}",
        endpoint,
        sender_name,
        receiver_name,
        trim_line(message, 90),
    ));

    let client = reqwest::Client::new();
    let response = match client.post(endpoint).json(&payload).send().await {
        Ok(r) => r,
        Err(e) => {
            if let Some(l) = ledger {
                let _ = l.append_transport_http(
                    "conversation",
                    &payload_json,
                    &e.to_string(),
                    None,
                    Some(&e.to_string()),
                );
            }
            return Err(anyhow!(e));
        }
    };
    let status = response.status();
    let code = status.as_u16();
    let body_text = response.text().await.unwrap_or_default();

    if let Some(l) = ledger {
        let err = if status.is_success() {
            None
        } else {
            Some(format!("HTTP {}", code))
        };
        let _ = l.append_transport_http(
            "conversation",
            &payload_json,
            &body_text,
            Some(code),
            err.as_deref(),
        );
    }

    if !status.is_success() {
        anyhow::bail!("conversation HTTP {}: {}", code, trim_line(&body_text, 200));
    }

    Ok(())
}

#[derive(DeriveSerialize, Deserialize, Debug)]
pub struct EvaluatorResult {
    pub evaluator_name: String,
    pub sentiment: String,
    pub message: String,
    pub timestamp: String,
    pub experiment_id: Option<String>,
    pub run_id: Option<String>,
    pub manifest_version: Option<String>,
}

pub async fn send_evaluator_result(
    endpoint: &str,
    evaluator_name: &str,
    sentiment: &str,
    message: &str,
    run_context: Option<&RunContext>,
    ledger: Option<&Arc<EventLedger>>,
) -> Result<(), anyhow::Error> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let timestamp_str = chrono::DateTime::<chrono::Utc>::from_timestamp(timestamp as i64, 0)
        .unwrap()
        .to_rfc3339();
    let payload = EvaluatorResult {
        evaluator_name: evaluator_name.to_string(),
        sentiment: sentiment.to_string(),
        message: message.to_string(),
        timestamp: timestamp_str,
        experiment_id: run_context.map(|c| c.experiment_id.clone()),
        run_id: run_context.map(|c| c.run_id.clone()),
        manifest_version: run_context.map(|c| c.manifest_version.clone()),
    };
    let payload_json = serde_json::to_string(&payload)?;

    if let Err(e) = guard_http_request(endpoint, "evaluator", ledger) {
        push_outgoing_http_log_line(format!(
            "BLOCKED {} | evaluator {} [{}] | {}",
            endpoint,
            evaluator_name,
            sentiment,
            trim_line(message, 90),
        ));
        return Err(e);
    }

    push_outgoing_http_log_line(format!(
        "POST {} | evaluator {} [{}] | {}",
        endpoint,
        evaluator_name,
        sentiment,
        trim_line(message, 90),
    ));
    let client = reqwest::Client::new();
    let response = match client.post(endpoint).json(&payload).send().await {
        Ok(r) => r,
        Err(e) => {
            if let Some(l) = ledger {
                let _ = l.append_transport_http(
                    "evaluator",
                    &payload_json,
                    &e.to_string(),
                    None,
                    Some(&e.to_string()),
                );
            }
            return Err(anyhow!(e));
        }
    };
    let status = response.status();
    let code = status.as_u16();
    let body_text = response.text().await.unwrap_or_default();

    if let Some(l) = ledger {
        let err = if status.is_success() {
            None
        } else {
            Some(format!("HTTP {}", code))
        };
        let _ = l.append_transport_http(
            "evaluator",
            &payload_json,
            &body_text,
            Some(code),
            err.as_deref(),
        );
    }

    if !status.is_success() {
        anyhow::bail!("evaluator HTTP {}: {}", code, trim_line(&body_text, 200));
    }
    Ok(())
}

pub async fn send_researcher_result(
    endpoint: &str,
    researcher_name: &str,
    topic: &str,
    message: &str,
    run_context: Option<&RunContext>,
    ledger: Option<&Arc<EventLedger>>,
) -> Result<(), anyhow::Error> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let timestamp_str = chrono::DateTime::<chrono::Utc>::from_timestamp(timestamp as i64, 0)
        .unwrap()
        .to_rfc3339();
    let payload = EvaluatorResult {
        evaluator_name: researcher_name.to_string(),
        sentiment: format!("references:{}", topic.to_lowercase()),
        message: message.to_string(),
        timestamp: timestamp_str,
        experiment_id: run_context.map(|c| c.experiment_id.clone()),
        run_id: run_context.map(|c| c.run_id.clone()),
        manifest_version: run_context.map(|c| c.manifest_version.clone()),
    };
    let payload_json = serde_json::to_string(&payload)?;

    if let Err(e) = guard_http_request(endpoint, "researcher", ledger) {
        push_outgoing_http_log_line(format!(
            "BLOCKED {} | researcher {} [{}] | {}",
            endpoint,
            researcher_name,
            topic,
            trim_line(message, 90),
        ));
        return Err(e);
    }

    push_outgoing_http_log_line(format!(
        "POST {} | researcher {} [{}] | {}",
        endpoint,
        researcher_name,
        topic,
        trim_line(message, 90),
    ));
    let client = reqwest::Client::new();
    let response = match client.post(endpoint).json(&payload).send().await {
        Ok(r) => r,
        Err(e) => {
            if let Some(l) = ledger {
                let _ = l.append_transport_http(
                    "researcher",
                    &payload_json,
                    &e.to_string(),
                    None,
                    Some(&e.to_string()),
                );
            }
            return Err(anyhow!(e));
        }
    };
    let status = response.status();
    let code = status.as_u16();
    let body_text = response.text().await.unwrap_or_default();

    if let Some(l) = ledger {
        let err = if status.is_success() {
            None
        } else {
            Some(format!("HTTP {}", code))
        };
        let _ = l.append_transport_http(
            "researcher",
            &payload_json,
            &body_text,
            Some(code),
            err.as_deref(),
        );
    }

    if !status.is_success() {
        anyhow::bail!("researcher HTTP {}: {}", code, trim_line(&body_text, 200));
    }
    Ok(())
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    service: &'static str,
}

#[derive(Serialize)]
struct OutgoingHttpLogResponse {
    lines: Vec<String>,
}

#[rocket::get("/health")]
fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "ams-agents",
    })
}

#[rocket::get("/outgoing-http-log")]
fn outgoing_http_log_route() -> Json<OutgoingHttpLogResponse> {
    Json(OutgoingHttpLogResponse {
        lines: get_outgoing_http_log_lines(),
    })
}

fn build_rocket(config: &WebConfig) -> Rocket<Build> {
    let figment: Figment = Config::figment()
        .merge(("address", config.address.clone()))
        .merge(("port", config.port));

    rocket::custom(figment).mount("/api", routes![health, outgoing_http_log_route])
}

pub fn start_embedded_server_if_enabled(rt_handle: &Handle) -> bool {
    let config = WebConfig::from_env();
    if !config.enabled {
        return false;
    }

    rt_handle.spawn(async move {
        if let Err(err) = build_rocket(&config).launch().await {
            eprintln!("Embedded Rocket server failed: {err}");
        }
    });

    true
}

#[cfg(test)]
mod tests {
    use super::parse_bool_env;

    #[test]
    fn parse_bool_env_honors_default_when_unset() {
        let key = "AMS_TEST_WEB_BOOL_MISSING";
        unsafe { std::env::remove_var(key) };
        assert!(!parse_bool_env(key, false));
        assert!(parse_bool_env(key, true));
    }

    #[test]
    fn parse_bool_env_supports_truthy_values() {
        let key = "AMS_TEST_WEB_BOOL_TRUE";
        unsafe { std::env::set_var(key, "yes") };
        assert!(parse_bool_env(key, false));
        unsafe { std::env::remove_var(key) };
    }

    #[test]
    fn parse_bool_env_supports_falsey_values() {
        let key = "AMS_TEST_WEB_BOOL_FALSE";
        unsafe { std::env::set_var(key, "off") };
        assert!(!parse_bool_env(key, true));
        unsafe { std::env::remove_var(key) };
    }
}
