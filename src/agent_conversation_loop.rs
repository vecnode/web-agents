use std::sync::{Arc, Mutex};
use tokio::time::{sleep, Duration};
use crate::reproducibility::RunContext;

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
    agent_a_topic: String,
    agent_a_topic_source: String,
    agent_b_id: usize,
    agent_b_name: String,
    agent_b_instruction: String,
    agent_b_topic: String,
    agent_b_topic_source: String,
    endpoint: String,
    active_flag: Arc<Mutex<bool>>,
    last_message_in_chat: Arc<Mutex<Option<String>>>,
    message_events: Arc<Mutex<Vec<String>>>,
    selected_model: Option<String>,
    history_size: usize,
    turn_delay_secs: u64,
    run_context: Option<RunContext>,
) {
    let mut turn = 0;
    let mut is_agent_a_turn = true;
    let mut history = ConversationHistory::new(history_size.max(1));
    let topics_summary = format!(
        "Topics => {}: \"{}\" [{}] | {}: \"{}\" [{}]",
        agent_a_name,
        agent_a_topic,
        agent_a_topic_source,
        agent_b_name,
        agent_b_topic,
        agent_b_topic_source
    );
    
    // Print conversation header
    let start_message = format!(
        "Conversation Started: {} ↔ {}\n{}",
        agent_a_name, agent_b_name, topics_summary
    );
    println!("\n{}", start_message);
    
    // Send to chat app
    let endpoint_clone = endpoint.clone();
    let topic_clone = topics_summary.clone();
    let run_context_for_start = run_context.clone();
    tokio::spawn(async move {
        if let Err(e) = crate::http_client::send_conversation_message(
            &endpoint_clone,
            0, // System message ID
            "Agent Manager",
            0,
            "System",
            &topic_clone,
            &start_message,
            run_context_for_start.as_ref(),
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
        let (sender_id, sender_name, sender_instruction, sender_topic, sender_topic_source, receiver_id, receiver_name, receiver_topic) = if is_agent_a_turn {
            (
                agent_a_id,
                agent_a_name.clone(),
                agent_a_instruction.clone(),
                agent_a_topic.clone(),
                agent_a_topic_source.clone(),
                agent_b_id,
                agent_b_name.clone(),
                agent_b_topic.clone(),
            )
        } else {
            (
                agent_b_id,
                agent_b_name.clone(),
                agent_b_instruction.clone(),
                agent_b_topic.clone(),
                agent_b_topic_source.clone(),
                agent_a_id,
                agent_a_name.clone(),
                agent_a_topic.clone(),
            )
        };
        let effective_topic = if sender_topic_source == "Follow Partner" {
            receiver_topic.clone()
        } else {
            sender_topic.clone()
        };
        
        // Enhance instruction with conversation context
        let enhanced_instruction = format!(
            "{}\n\nYou are now in a conversation with {} about \"{}\". Keep your responses concise and engaging (2-3 sentences preferred).",
            sender_instruction,
            receiver_name,
            effective_topic
        );
        
        // Build conversation context
        let conversation_context = history.format_history(&sender_name, &receiver_name, &effective_topic);
        
        // Print turn header
        let turn_message = format!(
            "Turn {}: {} -> {}",
            turn + 1, sender_name, receiver_name
        );
        println!("{}", turn_message);
        
        // Send turn message to chat app
        let endpoint_clone = endpoint.clone();
        let topic_clone = effective_topic.clone();
        let turn_message_clone = turn_message.clone();
        let run_context_for_turn = run_context.clone();
        tokio::spawn(async move {
            if let Err(e) = crate::http_client::send_conversation_message(
                &endpoint_clone,
                0, // System message ID
                "Agent Manager",
                0,
                "System",
                &topic_clone,
                &turn_message_clone,
                run_context_for_turn.as_ref(),
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
                // Include a monotonic turn marker so downstream nodes can react
                // once per turn even if model text repeats exactly.
                let event = format!("TURN:{}::MSG::{}", turn, response);
                *last_message_in_chat.lock().unwrap() = Some(event.clone());
                message_events.lock().unwrap().push(event);
                // Print formatted message
                println!("\n[{}]: {}", sender_name, response);
                println!();
                // Send via HTTP (non-blocking, log errors but continue)
                let endpoint_clone = endpoint.clone();
                let topic_clone = effective_topic.clone();
                let run_context_for_message = run_context.clone();
                tokio::spawn(async move {
                    if let Err(e) = crate::http_client::send_conversation_message(
                        &endpoint_clone,
                        sender_id,
                        &sender_name,
                        receiver_id,
                        &receiver_name,
                        &topic_clone,
                        &response,
                        run_context_for_message.as_ref(),
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
        if turn_delay_secs > 0 {
            sleep(Duration::from_secs(turn_delay_secs)).await;
        }
        
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
    let topic_clone = topics_summary;
    let run_context_for_end = run_context;
    tokio::spawn(async move {
        if let Err(e) = crate::http_client::send_conversation_message(
            &endpoint_clone,
            0, // System message ID
            "Agent Manager",
            0,
            "System",
            &topic_clone,
            &end_message,
            run_context_for_end.as_ref(),
        ).await {
            eprintln!("[HTTP] Failed to send conversation end message: {}", e);
        }
    });
}


