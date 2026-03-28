use crate::adk_integration::OllamaStopEpoch;
use crate::event_ledger::EventLedger;
use crate::http_client::{send_evaluator_result, send_researcher_result};
use crate::reproducibility::RunContext;
use futures_util::future::join_all;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// --- Sidecars: evaluators run after each line; researchers run pre-turn only (see below) --

#[derive(Clone, Default)]
pub struct ConversationSidecarConfig {
    pub evaluators: Vec<SidecarEvaluator>,
    pub researchers: Vec<SidecarResearcher>,
}

#[derive(Clone)]
pub struct SidecarEvaluator {
    pub global_id: String,
    pub instruction: String,
    pub analysis_mode: String,
    pub limit_token: bool,
    pub num_predict: String,
}

#[derive(Clone)]
pub struct SidecarResearcher {
    pub global_id: String,
    pub topic_mode: String,
    pub instruction: String,
    pub limit_token: bool,
    pub num_predict: String,
    /// Worker row id this researcher is wired to ("Injection"); pre-turn context uses the partner's last line.
    pub target_worker_id: usize,
}

// --- Pre-turn research injection (tweak wording / placement here) -----------------------

const RESEARCH_INJECTION_HEADER: &str =
    "\n\n---\nResearch references for your turn (consider when responding):\n";
const RESEARCH_INJECTION_FOOTER: &str = "\n---\n";

/// Where pre-turn research text is merged into the Ollama prompt for this worker turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResearchInjectionPlacement {
    /// After formatted history + “Your turn…” (default).
    ConversationContext,
    /// After the system instruction block, before the transcript (stronger emphasis on references).
    EnhancedInstruction,
}

pub const DEFAULT_RESEARCH_INJECTION_PLACEMENT: ResearchInjectionPlacement =
    ResearchInjectionPlacement::ConversationContext;

/// Merge non-empty `research_blocks` into the turn prompt per `placement`.
pub fn apply_research_injection(
    placement: ResearchInjectionPlacement,
    mut enhanced_instruction: String,
    mut conversation_context: String,
    research_blocks: &str,
) -> (String, String) {
    if research_blocks.is_empty() {
        return (enhanced_instruction, conversation_context);
    }
    let mut injection = String::with_capacity(
        RESEARCH_INJECTION_HEADER.len() + research_blocks.len() + RESEARCH_INJECTION_FOOTER.len(),
    );
    injection.push_str(RESEARCH_INJECTION_HEADER);
    injection.push_str(research_blocks);
    injection.push_str(RESEARCH_INJECTION_FOOTER);
    match placement {
        ResearchInjectionPlacement::ConversationContext => {
            conversation_context.push_str(&injection)
        }
        ResearchInjectionPlacement::EnhancedInstruction => {
            enhanced_instruction.push_str(&injection)
        }
    }
    (enhanced_instruction, conversation_context)
}

/// Which utterance grounds pre-turn research (for ledger / debugging).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResearchMessageGrounding {
    /// `message_for_research` is the tied worker's own last line in this dialogue.
    TiedWorkerLastMessage,
    /// Tied worker has not spoken yet; we use the partner's last line so first turn still gets references.
    PartnerFallbackFirstTurn,
}

/// Reference **category** from the Topics dropdown (Movies, Music, Articles, …) plus explicit grounding.
fn researcher_pre_turn_instruction(
    rs: &SidecarResearcher,
    tied_worker_name: &str,
    category: &str,
    grounding: ResearchMessageGrounding,
) -> String {
    let ground_line = match grounding {
        ResearchMessageGrounding::TiedWorkerLastMessage => format!(
            "The text below is the **last message from worker \"{name}\"** (the agent this researcher is tied to via Injection). \
             Base your suggestions only on that content.",
            name = tied_worker_name,
        ),
        ResearchMessageGrounding::PartnerFallbackFirstTurn => format!(
            "The tied worker \"{name}\" has not spoken yet in this dialogue. \
             The text below is the **partner's last message**—use it as the only ground for your {category} suggestions until the tied worker has their own prior line.",
            name = tied_worker_name,
            category = category,
        ),
    };
    format!(
        "{}\n\nReference category: **{category}** (e.g. films, albums, papers—stay in this category).\n\
         {ground_line}\n\
         Suggest exactly 3 items in that category. Bullet points: title/name + one line why it fits.",
        rs.instruction,
        category = category,
        ground_line = ground_line,
    )
}

/// Run researchers targeting `speaking_worker_id` **before** that worker's reply.
///
/// **Latency:** This always runs **before** the dialogue model. Each researcher row is one extra Ollama
/// round-trip; we run those in **parallel** when multiple researchers share the same injection worker.
/// Expect roughly `(research time) + (dialogue time)` per turn when research is active—not a bug, but
/// strictly more work than a plain dialogue turn.
pub async fn run_researchers_before_worker_turn(
    sidecars: &ConversationSidecarConfig,
    speaking_worker_id: usize,
    tied_worker_name: &str,
    message_for_research: &str,
    grounding: ResearchMessageGrounding,
    ollama_host: &str,
    endpoint: &str,
    run_context: Option<&RunContext>,
    selected_model: Option<&str>,
    ollama_stop_epoch: Option<OllamaStopEpoch>,
    post_http: bool,
    ledger: Option<&Arc<EventLedger>>,
) -> Result<String, ()> {
    let researchers: Vec<SidecarResearcher> = sidecars
        .researchers
        .iter()
        .filter(|rs| rs.target_worker_id == speaking_worker_id)
        .cloned()
        .collect();

    if researchers.is_empty() {
        return Ok(String::new());
    }

    let msg = message_for_research.to_string();
    let host = ollama_host.to_string();
    let model = selected_model.map(|s| s.to_string());
    let epoch = ollama_stop_epoch.clone();

    let futures = researchers.into_iter().map(|rs| {
        let msg = msg.clone();
        let host = host.clone();
        let model = model.clone();
        let epoch = epoch.clone();
        let tied = tied_worker_name.to_string();
        let category = if rs.topic_mode.trim().is_empty() {
            "Articles".to_string()
        } else {
            rs.topic_mode.clone()
        };
        let instruction = researcher_pre_turn_instruction(&rs, &tied, &category, grounding);
        async move {
            let ollama_input = format!("{}\n{}", instruction, msg);
            let out = crate::adk_integration::send_to_ollama(
                host.as_str(),
                &instruction,
                &msg,
                rs.limit_token,
                &rs.num_predict,
                model.as_deref(),
                epoch,
            )
            .await;
            (rs, category, ollama_input, out)
        }
    });

    let joined = join_all(futures).await;
    let mut blocks = Vec::new();

    for (rs, topic, ollama_input, out) in joined {
        match out {
            Ok(response) => {
                if let Some(l) = ledger {
                    let _ = l.append_with_hashes(
                        "sidecar.researcher",
                        Some(rs.global_id.clone()),
                        selected_model.map(|s| s.to_string()),
                        &ollama_input,
                        &response,
                        serde_json::json!({
                            "topic": topic,
                            "phase": "pre_turn_injection",
                            "grounding": match grounding {
                                ResearchMessageGrounding::TiedWorkerLastMessage => "tied_worker_last",
                                ResearchMessageGrounding::PartnerFallbackFirstTurn => "partner_fallback",
                            },
                        }),
                    );
                }
                if post_http {
                    if let Err(e) = send_researcher_result(
                        endpoint,
                        "Agent Researcher",
                        &topic,
                        &response,
                        run_context,
                        ledger,
                    )
                    .await
                    {
                        eprintln!("[Researcher] Failed to send to ams-chat: {}", e);
                    }
                }
                blocks.push(format!(
                    "### References ({topic})\n{response}",
                    topic = topic,
                    response = response
                ));
            }
            Err(e) => {
                if e.to_string() == crate::adk_integration::OLLAMA_STOPPED_MSG {
                    return Err(());
                }
                if let Some(l) = ledger {
                    let _ = l.append_with_hashes(
                        "sidecar.researcher",
                        Some(rs.global_id.clone()),
                        selected_model.map(|s| s.to_string()),
                        &ollama_input,
                        "",
                        serde_json::json!({
                            "topic": topic,
                            "phase": "pre_turn_injection",
                            "stage": "ollama",
                            "error": e.to_string(),
                        }),
                    );
                }
                eprintln!("[Researcher] Ollama error (pre-turn): {}", e);
            }
        }
    }
    Ok(blocks.join("\n\n"))
}

fn evaluator_sentiment(analysis_mode: &str, response: &str) -> &'static str {
    let response_lower = response.to_lowercase();
    match analysis_mode {
        "Topic Extraction" => "topic",
        "Decision Analysis" => "decision",
        "Sentiment Classification" => {
            if response_lower.contains("positive") || response_lower.contains("happy") {
                "sentiment"
            } else if response_lower.contains("negative")
                || response_lower.contains("sad")
                || response_lower.contains("angry")
                || response_lower.contains("frustrated")
            {
                "sentiment"
            } else if response_lower.contains("neutral") {
                "sentiment"
            } else {
                "unknown"
            }
        }
        _ => {
            if response_lower.contains("happy") {
                "happy"
            } else if response_lower.contains("sad") {
                "sad"
            } else {
                "analysis"
            }
        }
    }
}

/// Evaluator sidecars after each dialogue line (`agent_message` = the line just spoken).
/// Researchers are **not** run here; they run pre-turn via [`run_researchers_before_worker_turn`].
pub async fn run_evaluator_sidecars_for_message(
    sidecars: &ConversationSidecarConfig,
    agent_message: &str,
    ollama_host: &str,
    endpoint: &str,
    run_context: Option<&RunContext>,
    selected_model: Option<&str>,
    ollama_stop_epoch: Option<OllamaStopEpoch>,
    post_http: bool,
    ledger: Option<&Arc<EventLedger>>,
) -> Result<(), ()> {
    for ev in &sidecars.evaluators {
        let ollama_input = format!("{}\n{}", ev.instruction, agent_message);
        match crate::adk_integration::send_to_ollama(
            ollama_host,
            &ev.instruction,
            agent_message,
            ev.limit_token,
            &ev.num_predict,
            selected_model,
            ollama_stop_epoch.clone(),
        )
        .await
        {
            Ok(response) => {
                if let Some(l) = ledger {
                    let _ = l.append_with_hashes(
                        "sidecar.evaluator",
                        Some(ev.global_id.clone()),
                        selected_model.map(|s| s.to_string()),
                        &ollama_input,
                        &response,
                        serde_json::json!({ "analysis_mode": ev.analysis_mode }),
                    );
                }
                let sentiment = evaluator_sentiment(ev.analysis_mode.as_str(), &response);
                if post_http {
                    if let Err(e) = send_evaluator_result(
                        endpoint,
                        "Agent Evaluator",
                        sentiment,
                        &response,
                        run_context,
                        ledger,
                    )
                    .await
                    {
                        eprintln!("[Evaluator] Failed to send to ams-chat: {}", e);
                    }
                }
            }
            Err(e) => {
                if e.to_string() == crate::adk_integration::OLLAMA_STOPPED_MSG {
                    return Err(());
                }
                if let Some(l) = ledger {
                    let _ = l.append_with_hashes(
                        "sidecar.evaluator",
                        Some(ev.global_id.clone()),
                        selected_model.map(|s| s.to_string()),
                        &ollama_input,
                        "",
                        serde_json::json!({
                            "analysis_mode": ev.analysis_mode,
                            "stage": "ollama",
                            "error": e.to_string(),
                        }),
                    );
                }
                eprintln!("[Evaluator] Ollama error: {}", e);
            }
        }
    }

    Ok(())
}

// --- Conversation history ---------------------------------------------------------------

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

    // Print conversation header
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

    // Send to chat app (await so later lines follow this in order).
    if let Err(e) = crate::http_client::send_conversation_message(
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

        // Pre-turn: ground on the **tied worker's** last line when it exists; else partner line (first turn).
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

        // Enhance instruction with conversation context
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

        // Print turn header
        let turn_message = format!("Turn {}: {} -> {}", turn + 1, sender_name, receiver_name);
        println!("{}", turn_message);

        let endpoint_clone = endpoint.clone();
        let topic_clone = effective_topic.clone();
        let turn_message_clone = turn_message.clone();
        let run_context_for_turn = run_context.clone();
        let ledger_turn = ledger.clone();
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
                ledger_turn.as_ref(),
            )
            .await
            {
                eprintln!("[HTTP] Failed to send turn message: {}", e);
            }
        });

        // Send message to Ollama
        let dialogue_input = format!("{}\n---\n{}", enhanced_instruction, conversation_context);
        match crate::adk_integration::send_to_ollama_with_context(
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
                // Add to history
                history.add_message(sender_id, sender_name.clone(), response.clone(), turn);
                // Include a monotonic turn marker so downstream nodes can react
                // once per turn even if model text repeats exactly.
                let event = format!(
                    "SRC{}:TURN:{}::MSG::{}",
                    message_event_source_id, turn, response
                );
                *last_message_in_chat.lock().unwrap() = Some(event.clone());
                message_events.lock().unwrap().push(event);
                // Print formatted message
                println!("\n[{}]: {}", sender_name, response);
                println!();

                // Chat transcript: include pre-turn research in the same worker line (ledger still has full dialogue_input).
                let message_for_chat = if research_injection.is_empty() {
                    response.clone()
                } else {
                    format!(
                        "{}\n\n---\nResearch (used for this turn)\n{}",
                        response, research_injection
                    )
                };

                // Agent line to chat before sidecars so chat order matches dialogue flow.
                if let Err(e) = crate::http_client::send_conversation_message(
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

                // Evaluators only (researchers run pre-turn). Must finish before the next dialogue Ollama call.
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

                // Switch turns
                is_agent_a_turn = !is_agent_a_turn;
                turn += 1;
            }
            Err(e) => {
                if e.to_string() == crate::adk_integration::OLLAMA_STOPPED_MSG {
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

    // End line after all turns and their sidecars; await so nothing is still in flight here.
    if let Err(e) = crate::http_client::send_conversation_message(
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

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_RESEARCH_INJECTION_PLACEMENT, ResearchInjectionPlacement, apply_research_injection,
    };

    #[test]
    fn apply_research_injection_empty_is_noop() {
        let (e, c) = apply_research_injection(
            DEFAULT_RESEARCH_INJECTION_PLACEMENT,
            "sys".into(),
            "ctx".into(),
            "",
        );
        assert_eq!(e, "sys");
        assert_eq!(c, "ctx");
    }

    #[test]
    fn apply_research_injection_appends_to_context_by_default() {
        let (e, c) = apply_research_injection(
            ResearchInjectionPlacement::ConversationContext,
            "sys".into(),
            "ctx".into(),
            "refs",
        );
        assert_eq!(e, "sys");
        assert!(c.contains("refs"));
        assert!(c.contains("Research references"));
    }

    #[test]
    fn apply_research_injection_can_target_enhanced_instruction() {
        let (e, c) = apply_research_injection(
            ResearchInjectionPlacement::EnhancedInstruction,
            "sys".into(),
            "ctx".into(),
            "refs",
        );
        assert!(e.contains("refs"));
        assert_eq!(c, "ctx");
    }
}
