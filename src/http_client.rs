use serde::{Serialize, Deserialize};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};
use crate::reproducibility::RunContext;

static OUTGOING_HTTP_LOG: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

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

#[derive(Serialize, Deserialize, Debug)]
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

    push_outgoing_http_log_line(format!(
        "POST {} | conversation | {} -> {} | {}",
        endpoint,
        sender_name,
        receiver_name,
        trim_line(message, 90),
    ));
    
    let client = reqwest::Client::new();
    let response = client
        .post(endpoint)
        .json(&payload)
        .send()
        .await?;
    
    if !response.status().is_success() {
        eprintln!("HTTP request failed: {}", response.status());
    }
    
    Ok(())
}

#[derive(Serialize, Deserialize, Debug)]
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
    push_outgoing_http_log_line(format!(
        "POST {} | evaluator {} [{}] | {}",
        endpoint,
        evaluator_name,
        sentiment,
        trim_line(message, 90),
    ));
    let client = reqwest::Client::new();
    let response = client.post(endpoint).json(&payload).send().await?;
    if !response.status().is_success() {
        eprintln!("HTTP evaluator request failed: {}", response.status());
    }
    Ok(())
}

pub async fn send_researcher_result(
    endpoint: &str,
    researcher_name: &str,
    topic: &str,
    message: &str,
    run_context: Option<&RunContext>,
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
    push_outgoing_http_log_line(format!(
        "POST {} | researcher {} [{}] | {}",
        endpoint,
        researcher_name,
        topic,
        trim_line(message, 90),
    ));
    let client = reqwest::Client::new();
    let response = client.post(endpoint).json(&payload).send().await?;
    if !response.status().is_success() {
        eprintln!("HTTP researcher request failed: {}", response.status());
    }
    Ok(())
}
