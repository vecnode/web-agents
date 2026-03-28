use crate::event_ledger::EventLedger;
use crate::reproducibility::RunContext;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

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
            return Err(anyhow::anyhow!(e));
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
            return Err(anyhow::anyhow!(e));
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
            return Err(anyhow::anyhow!(e));
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
