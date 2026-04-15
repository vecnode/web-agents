use crate::agents::conversation_sidecars::{
    ConversationSidecarConfig, ResearchMessageGrounding, apply_research_injection,
    run_evaluator_sidecars_for_message, run_researchers_before_worker_turn,
    DEFAULT_RESEARCH_INJECTION_PLACEMENT,
};
use crate::run::event_ledger::EventLedger;
use crate::run::manifest::RunContext;
use crate::ollama::OllamaStopEpoch;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// ─── Conversation history ─────────────────────────────────────────────────

#[derive(Clone)]
struct ConversationMessage {
    agent_id: usize,
    agent_name: String,
    message: String,
    #[allow(dead_code)]
    turn: usize,
}

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
        if self.messages.len() > self.max_history {
            self.messages.remove(0);
        }
    }

    /// Latest utterance from a participant (e.g. partner's last line before your turn).
    fn last_message_from_agent(&self, agent_id: usize) -> Option<&str> {
        self.messages
            .iter()
            .rev()
            .find(|m| m.agent_id == agent_id)
            .map(|m| m.message.as_str())
    }

    fn format_history(&self, _current_agent_name: &str, partner_name: &str, topic: &str) -> String {
        if self.messages.is_empty() {
            return format!(
                "You are discussing \"{}\" with {}. Please start the conversation.",
                topic, partner_name
            );
        }

        let mut formatted = format!(
            "You are discussing \"{}\" with {}. Here's the conversation so far:\n\n",
            topic, partner_name
        );

        for msg in &self.messages {
            formatted.push_str(&format!("{}: {}\n\n", msg.agent_name, msg.message));
        }

        formatted.push_str(&format!(
            "Your turn: Respond to {}'s last message.",
            partner_name
        ));
        formatted
    }
}

// ─── Conversation loop entry point ────────────────────────────────────────

/// `message_event_source_id` namespaces evaluator/researcher event keys (e.g. conversation loop id)
/// so parallel loops never share duplicate `TURN:n` prefixes.
pub async fn start_conversation_loop(
    message_event_source_id: usize,
    ollama_stop_epoch: Option<OllamaStopEpoch>,
    sidecars: Arc<ConversationSidecarConfig>,
    agent_a_id: usize,
    agent_a_name: String,
    agent_a_instruction: String,
    agent_a_topic: String,
    agent_a_topic_source: String,
    agent_a_global_id: String,
    agent_b_id: usize,
    agent_b_name: String,
    agent_b_instruction: String,
    agent_b_topic: String,
    agent_b_topic_source: String,
    agent_b_global_id: String,
    ollama_host: String,
    endpoint: String,
    active_flag: Arc<Mutex<bool>>,
    last_message_in_chat: Arc<Mutex<Option<String>>>,
    message_events: Arc<Mutex<Vec<String>>>,
    selected_model: Option<String>,
    history_size: usize,
    run_context: Option<RunContext>,
    run_generation: u64,
    run_generation_counter: Arc<AtomicU64>,
    loops_remaining_in_run: Arc<AtomicUsize>,
    conversation_graph_running: Arc<AtomicBool>,
    ledger: Option<Arc<EventLedger>>,
) {
    let mut turn = 0;
    let mut is_agent_a_turn = true;
    let mut history = ConversationHistory::new(history_size.max(1));
    let topics_summary = format!(
        "Topics => {}: \"{}\" | {}: \"{}\"",
        agent_a_name, agent_a_topic, agent_b_name, agent_b_topic,
    );

    let start_message = format!(
        "Conversation Started: {} ↔ {}\n{}",
        agent_a_name, agent_b_name, topics_summary
    );
    println!("\n{}", start_message);

    if let Some(ref l) = ledger {
        let _ = l.append_with_hashes(
            "dialogue.start",
            None,
            selected_model.clone(),
            "",
            &start_message,
            serde_json::json!({ "topics_summary": topics_summary }),
        );
    }

    if let Err(e) = crate::web::send_conversation_message(
        &endpoint,
        0,
        "Agent Manager",
        0,
        "System",
        &topics_summary,
        &start_message,
        run_context.as_ref(),
        ledger.as_ref(),
    )
    .await
    {
        eprintln!("[HTTP] Failed to send conversation start message: {}", e);
    }

    loop {
        let active = {
            let flag = active_flag.lock().unwrap();
            *flag
        };

        if !active {
            println!("\n[Conversation stopped by user]");
            break;
        }

        let (
            sender_id,
            sender_name,
            sender_instruction,
            sender_topic,
            sender_topic_source,
            receiver_id,
            receiver_name,
            receiver_topic,
        ) = if is_agent_a_turn {
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

        // Pre-turn: ground on the tied worker's last line when it exists; else partner line (first turn).
        let research_injection = if let Some((line, grounding)) = history
            .last_message_from_agent(sender_id)
            .map(|t| (t, ResearchMessageGrounding::TiedWorkerLastMessage))
            .or_else(|| {
                history
                    .last_message_from_agent(receiver_id)
                    .map(|p| (p, ResearchMessageGrounding::PartnerFallbackFirstTurn))
            }) {
            match run_researchers_before_worker_turn(
                sidecars.as_ref(),
                sender_id,
                sender_name.as_str(),
                line,
                grounding,
                ollama_host.as_str(),
                endpoint.as_str(),
                run_context.as_ref(),
                selected_model.as_deref(),
                ollama_stop_epoch.clone(),
                false,
                ledger.as_ref(),
            )
            .await
            {
                Ok(s) => s,
                Err(()) => break,
            }
        } else {
            String::new()
        };

        let enhanced_instruction = format!(
            "{}\n\nYou are now in a conversation with {} about \"{}\". Keep your responses concise and engaging (2-3 sentences preferred).",
            sender_instruction, receiver_name, effective_topic
        );

        let conversation_context =
            history.format_history(&sender_name, &receiver_name, &effective_topic);
        let (enhanced_instruction, conversation_context) = apply_research_injection(
            DEFAULT_RESEARCH_INJECTION_PLACEMENT,
            enhanced_instruction,
            conversation_context,
            &research_injection,
        );

        let turn_message = format!("Turn {}: {} -> {}", turn + 1, sender_name, receiver_name);
        println!("{}", turn_message);

        let endpoint_clone = endpoint.clone();
        let topic_clone = effective_topic.clone();
        let turn_message_clone = turn_message.clone();
        let run_context_for_turn = run_context.clone();
        let ledger_turn = ledger.clone();
        tokio::spawn(async move {
            if let Err(e) = crate::web::send_conversation_message(
                &endpoint_clone,
                0,
                "Agent Manager",
                0,
                "System",
                &topic_clone,
                &turn_message_clone,
                run_context_for_turn.as_ref(),
                ledger_turn.as_ref(),
            )
            .await
            {
                eprintln!("[HTTP] Failed to send turn message: {}", e);
            }
        });

        let dialogue_input = format!("{}\n---\n{}", enhanced_instruction, conversation_context);
        match crate::ollama::send_to_ollama(
            ollama_host.as_str(),
            &enhanced_instruction,
            &conversation_context,
            false,
            "",
            selected_model.as_deref(),
            ollama_stop_epoch.clone(),
        )
        .await
        {
            Ok(response) => {
                let sender_gid = if sender_id == agent_a_id {
                    agent_a_global_id.clone()
                } else {
                    agent_b_global_id.clone()
                };
                if let Some(ref l) = ledger {
                    let _ = l.append_with_hashes(
                        "dialogue.turn",
                        Some(sender_gid),
                        selected_model.clone(),
                        &dialogue_input,
                        &response,
                        serde_json::json!({
                            "turn": turn,
                            "receiver_name": receiver_name,
                        }),
                    );
                }
                history.add_message(sender_id, sender_name.clone(), response.clone(), turn);
                let event = format!(
                    "SRC{}:TURN:{}::MSG::{}",
                    message_event_source_id, turn, response
                );
                *last_message_in_chat.lock().unwrap() = Some(event.clone());
                message_events.lock().unwrap().push(event);
                println!("\n[{}]: {}", sender_name, response);
                println!();

                let message_for_chat = if research_injection.is_empty() {
                    response.clone()
                } else {
                    format!(
                        "{}\n\n---\nResearch (used for this turn)\n{}",
                        response, research_injection
                    )
                };

                if let Err(e) = crate::web::send_conversation_message(
                    &endpoint,
                    sender_id,
                    &sender_name,
                    receiver_id,
                    &receiver_name,
                    &effective_topic,
                    &message_for_chat,
                    run_context.as_ref(),
                    ledger.as_ref(),
                )
                .await
                {
                    eprintln!("[HTTP] Failed to send message: {}", e);
                }

                if run_evaluator_sidecars_for_message(
                    sidecars.as_ref(),
                    &response,
                    ollama_host.as_str(),
                    &endpoint,
                    run_context.as_ref(),
                    selected_model.as_deref(),
                    ollama_stop_epoch.clone(),
                    true,
                    ledger.as_ref(),
                )
                .await
                .is_err()
                {
                    break;
                }

                is_agent_a_turn = !is_agent_a_turn;
                turn += 1;
            }
            Err(e) => {
                if e.to_string() == crate::ollama::OLLAMA_STOPPED_MSG {
                    break;
                }
                if let Some(ref l) = ledger {
                    let _ = l.append_with_hashes(
                        "dialogue.ollama_error",
                        None,
                        selected_model.clone(),
                        &dialogue_input,
                        "",
                        serde_json::json!({ "error": e.to_string(), "turn": turn }),
                    );
                }
                eprintln!("[Error] Ollama error in conversation loop: {}", e);
                break;
            }
        }

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

    if let Some(ref l) = ledger {
        let _ = l.append_with_hashes(
            "dialogue.end",
            None,
            selected_model.clone(),
            "",
            &end_message,
            serde_json::json!({ "total_turns": turn }),
        );
    }

    if let Err(e) = crate::web::send_conversation_message(
        &endpoint,
        0,
        "Agent Manager",
        0,
        "System",
        &topics_summary,
        &end_message,
        run_context.as_ref(),
        ledger.as_ref(),
    )
    .await
    {
        eprintln!("[HTTP] Failed to send conversation end message: {}", e);
    }

    let prev_remaining = loops_remaining_in_run.fetch_sub(1, Ordering::SeqCst);
    if prev_remaining == 1 && run_generation_counter.load(Ordering::SeqCst) == run_generation {
        conversation_graph_running.store(false, Ordering::Release);
        if let Some(ref l) = ledger {
            let _ = l.try_finalize_run_stopped("conversation_loops_finished");
        }
    }
}
