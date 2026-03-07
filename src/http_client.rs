use serde::{Serialize, Deserialize};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Serialize, Deserialize, Debug)]
pub struct ConversationMessage {
    pub sender_id: usize,
    pub sender_name: String,
    pub receiver_id: usize,
    pub receiver_name: String,
    pub topic: String,
    pub message: String,
    pub timestamp: String,
}

pub async fn send_conversation_message(
    endpoint: &str,
    sender_id: usize,
    sender_name: &str,
    receiver_id: usize,
    receiver_name: &str,
    topic: &str,
    message: &str,
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
    };
    
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
