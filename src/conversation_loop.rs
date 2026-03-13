use std::sync::{Arc, Mutex};
use tokio::time::{sleep, Duration};

// Conversation history entry
#[derive(Clone)]
struct ConversationMessage {
    #[allow(dead_code)]
    agent_id: usize,
    agent_name: String,
    message: String,
    #[allow(dead_code)]
    turn: usize,
}

// Conversation history manager
struct ConversationHistory {
    messages: Vec<ConversationMessage>,
    max_history: usize,
}

impl ConversationHistory {
    fn new(max_history: usize) -> Self {
        Self {
            messages: Vec::new(),
            max_history,
        }
    }
    
    fn add_message(&mut self, agent_id: usize, agent_name: String, message: String, turn: usize) {
        self.messages.push(ConversationMessage {
            agent_id,
            agent_name,
            message,
            turn,
        });
        
        // Keep only the last max_history messages
        if self.messages.len() > self.max_history {
            self.messages.remove(0);
        }
    }
    
    fn format_history(&self, _current_agent_name: &str, partner_name: &str, topic: &str) -> String {
        if self.messages.is_empty() {
            return format!("You are discussing \"{}\" with {}. Please start the conversation.", topic, partner_name);
        }
        
        let mut formatted = format!("You are discussing \"{}\" with {}. Here's the conversation so far:\n\n", topic, partner_name);
        
        for msg in &self.messages {
            formatted.push_str(&format!("{}: {}\n\n", msg.agent_name, msg.message));
        }
        
        formatted.push_str(&format!("Your turn: Respond to {}'s last message.", partner_name));
        formatted
    }
    
}

pub async fn start_conversation_loop(
    agent_a_id: usize,
    agent_a_name: String,
    agent_a_instruction: String,
    agent_b_id: usize,
    agent_b_name: String,
    agent_b_instruction: String,
    topic: String,
    endpoint: String,
    active_flag: Arc<Mutex<bool>>,
    last_message_in_chat: Arc<Mutex<Option<String>>>,
    selected_model: Option<String>,
) {
    let mut turn = 0;
    let mut is_agent_a_turn = true;
    let mut history = ConversationHistory::new(5); // Keep last 5 messages
    
    // Print conversation header
    let start_message = format!(
        "Conversation Started: {} ↔ {}\nTopic: {}",
        agent_a_name, agent_b_name, topic
    );
    println!("\n{}", start_message);
    
    // Send to chat app
    let endpoint_clone = endpoint.clone();
    let topic_clone = topic.clone();
    tokio::spawn(async move {
        if let Err(e) = crate::http_client::send_conversation_message(
            &endpoint_clone,
            0, // System message ID
            "Agent Manager",
            0,
            "System",
            &topic_clone,
            &start_message,
        ).await {
            eprintln!("[HTTP] Failed to send conversation start message: {}", e);
        }
    });
    
    loop {
        // Check if conversation is still active
        let active = {
            let flag = active_flag.lock().unwrap();
            *flag
        };
        
        if !active {
            println!("\n[Conversation stopped by user]");
            break;
        }
        
        // Determine which agent is speaking
        let (sender_id, sender_name, sender_instruction, receiver_name) = if is_agent_a_turn {
            (agent_a_id, agent_a_name.clone(), agent_a_instruction.clone(), agent_b_name.clone())
        } else {
            (agent_b_id, agent_b_name.clone(), agent_b_instruction.clone(), agent_a_name.clone())
        };
        
        // Enhance instruction with conversation context
        let enhanced_instruction = format!(
            "{}\n\nYou are now in a conversation with {} about \"{}\". Keep your responses concise and engaging (2-3 sentences preferred).",
            sender_instruction,
            receiver_name,
            topic
        );
        
        // Build conversation context
        let conversation_context = history.format_history(&sender_name, &receiver_name, &topic);
        
        // Print turn header
        let turn_message = format!(
            "Turn {}: {} -> {}",
            turn + 1, sender_name, receiver_name
        );
        println!("{}", turn_message);
        
        // Send turn message to chat app
        let endpoint_clone = endpoint.clone();
        let topic_clone = topic.clone();
        let turn_message_clone = turn_message.clone();
        tokio::spawn(async move {
            if let Err(e) = crate::http_client::send_conversation_message(
                &endpoint_clone,
                0, // System message ID
                "Agent Manager",
                0,
                "System",
                &topic_clone,
                &turn_message_clone,
            ).await {
                eprintln!("[HTTP] Failed to send turn message: {}", e);
            }
        });
        
        // Send message to Ollama
        match crate::adk_integration::send_to_ollama_with_context(
            &enhanced_instruction,
            &conversation_context,
            false,
            "",
            selected_model.as_deref(),
        ).await {
            Ok(response) => {
                // Add to history
                history.add_message(sender_id, sender_name.clone(), response.clone(), turn);
                *last_message_in_chat.lock().unwrap() = Some(response.clone());
                // Print formatted message
                println!("\n[{}]: {}", sender_name, response);
                println!();
                // Send via HTTP (non-blocking, log errors but continue)
                let endpoint_clone = endpoint.clone();
                let topic_clone = topic.clone();
                let receiver_id = if is_agent_a_turn { agent_b_id } else { agent_a_id };
                tokio::spawn(async move {
                    if let Err(e) = crate::http_client::send_conversation_message(
                        &endpoint_clone,
                        sender_id,
                        &sender_name,
                        receiver_id,
                        &receiver_name,
                        &topic_clone,
                        &response,
                    ).await {
                        eprintln!("[HTTP] Failed to send message: {}", e);
                    }
                });
                
                // Switch turns
                is_agent_a_turn = !is_agent_a_turn;
                turn += 1;
            }
            Err(e) => {
                eprintln!("[Error] Ollama error in conversation loop: {}", e);
                break;
            }
        }
        
        // Wait before next turn
        sleep(Duration::from_secs(3)).await;
        
        // Safety limit
        if turn > 50 {
            println!("\n[Conversation reached safety limit of 50 turns]");
            break;
        }
    }
    
    let end_message = format!(
        "Conversation Ended: {} ↔ {}\nTotal turns: {}",
        agent_a_name, agent_b_name, turn
    );
    println!("\n{}", end_message);
    
    // Send to chat app
    let endpoint_clone = endpoint.clone();
    let topic_clone = topic.clone();
    tokio::spawn(async move {
        if let Err(e) = crate::http_client::send_conversation_message(
            &endpoint_clone,
            0, // System message ID
            "Agent Manager",
            0,
            "System",
            &topic_clone,
            &end_message,
        ).await {
            eprintln!("[HTTP] Failed to send conversation end message: {}", e);
        }
    });
}
