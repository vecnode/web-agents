# Conversation Engine

This module is the execution core for agent-to-agent dialogue.

## Run trigger

A run starts when `run_graph()` is called. It stops previous loops, creates a new agent-to-chat channel, writes a manifest, opens an event ledger, and spawns async conversation loops.

```rust
pub(crate) fn run_graph(&mut self) -> String {
    self.stop_graph();
    let (chat_tx, chat_rx) = std::sync::mpsc::channel::<crate::agents::AgentChatEvent>();
    self.chat_turn_tx = Some(chat_tx);
    self.chat_turn_rx = Some(chat_rx);

    let manifest = self.build_run_manifest(None, self.read_only_replay_mode)?;
    let manifest_path = self.persist_active_manifest(manifest)?;

    self.event_ledger = Some(Arc::new(EventLedger::open(
        manifest_path.parent().unwrap().to_path_buf(),
        ctx.experiment_id.clone(),
        ctx.run_id.clone(),
    )?));

    self.start_conversation_from_node_worker_resolved(...);
    "ok".to_string()
}
```

## Turn loop

Each async loop rotates speakers, assembles prompts, calls Ollama, records metrics and ledger events, and forwards completed messages to the Overview chat room.

```rust
loop {
    if !*active_flag.lock().unwrap() {
        break;
    }

    let sender = &participants[current_speaker_idx];
    let receiver = &participants[(current_speaker_idx + 1) % participants.len()];

    let memory_block = session.memory_block(&receiver.name, &effective_topic);
    let prompt = PromptAssembler::assemble(PromptBuildInput {
        base_instruction: &sender.instruction,
        manager_name: &sender.manager_name,
        turn_index: turn,
        sender_name: &sender.name,
        receiver_name: &receiver.name,
        topic: &effective_topic,
        memory_block: &memory_block,
        sidecar_augmentation: &research_injection,
    });

    let inference = crate::ollama::send_to_ollama_with_result(...).await?;
    session.record_turn(sender.id, sender.name.clone(), inference.response.clone(), inference.usage.as_ref());
    turn += 1;
}
```

## Prompt memory strategy

`DialogueSessionState` keeps a bounded recent window plus a rolling summary. This controls prompt growth while preserving continuity.

```rust
pub fn record_turn(&mut self, agent_id: usize, agent_name: String, message: String, usage: Option<&TokenUsage>) {
    self.per_agent_last.insert(agent_id, message.clone());
    self.recent_exchanges.push_back(DialogueMessage { agent_id, agent_name, message });

    while self.recent_exchanges.len() > self.max_recent {
        if let Some(old) = self.recent_exchanges.pop_front() {
            self.rolling_summary.push_str(&format!("{} said: {}", old.agent_name, truncate_for_summary(&old.message)));
        }
    }

    self.token_budget.record_usage(usage);
}
```

## Stop semantics

Run stop is cooperative. An epoch counter is incremented in `stop_graph()`, and in-flight streaming checks the epoch to terminate quickly.
