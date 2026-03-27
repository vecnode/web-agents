use eframe::egui;
use tokio::runtime::Handle;
use std::sync::{Arc, Mutex};
use crate::agent_entities::{Agent, AgentManager, Evaluator, Researcher};
use crate::reproducibility::{RunContext, RunManifest};

mod settings_panel;
mod nodes_panel;
mod agent_worker_node;
mod agent_evaluator_node;
mod agent_researcher_node;

pub struct AMSAgents {
    rt_handle: Handle,
    ollama_models: Arc<Mutex<Vec<String>>>,
    ollama_models_loading: Arc<Mutex<bool>>,
    selected_ollama_model: String,
    managers: Vec<AgentManager>,
    agents: Vec<Agent>,
    evaluators: Vec<Evaluator>,
    researchers: Vec<Researcher>,
    conversation_turn_delay_secs: u64,
    conversation_history_size: usize,
    http_endpoint: String,
    last_message_in_chat: Arc<Mutex<Option<String>>>,
    conversation_message_events: Arc<Mutex<Vec<String>>>,
    last_evaluated_message_by_evaluator: Arc<Mutex<std::collections::HashMap<usize, String>>>,
    last_researched_message_by_researcher: Arc<Mutex<std::collections::HashMap<usize, String>>>,
    evaluator_event_queues: Arc<Mutex<std::collections::HashMap<usize, std::collections::VecDeque<(String, String)>>>>,
    researcher_event_queues: Arc<Mutex<std::collections::HashMap<usize, std::collections::VecDeque<(String, String)>>>>,
    evaluator_inflight_nodes: Arc<Mutex<std::collections::HashSet<usize>>>,
    researcher_inflight_nodes: Arc<Mutex<std::collections::HashSet<usize>>>,
    conversation_loop_handles: Vec<(usize, Arc<Mutex<bool>>, tokio::task::JoinHandle<()>)>, // (agent_id, active_flag, handle)
    current_run_context: Option<RunContext>,
    current_manifest: Option<RunManifest>,
    manifest_export_path: String,
    manifest_import_path: String,
    manifest_status_message: String,
    read_only_replay_mode: bool,
    theme_applied: bool,
    nodes_panel: nodes_panel::NodesPanelState,
}

impl AMSAgents {
    pub fn new(rt_handle: Handle) -> Self {
        Self { 
            rt_handle,
            ollama_models: Arc::new(Mutex::new(Vec::new())),
            ollama_models_loading: Arc::new(Mutex::new(false)),
            selected_ollama_model: std::env::var("OLLAMA_MODEL").unwrap_or_default(),
            managers: Vec::new(),
            agents: Vec::new(),
            evaluators: Vec::new(),
            researchers: Vec::new(),
            conversation_turn_delay_secs: 3,
            conversation_history_size: 5,
            http_endpoint: std::env::var("CONVERSATION_HTTP_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:3000/".to_string()),
            last_message_in_chat: Arc::new(Mutex::new(None)),
            conversation_message_events: Arc::new(Mutex::new(Vec::new())),
            last_evaluated_message_by_evaluator: Arc::new(Mutex::new(std::collections::HashMap::new())),
            last_researched_message_by_researcher: Arc::new(Mutex::new(std::collections::HashMap::new())),
            evaluator_event_queues: Arc::new(Mutex::new(std::collections::HashMap::new())),
            researcher_event_queues: Arc::new(Mutex::new(std::collections::HashMap::new())),
            evaluator_inflight_nodes: Arc::new(Mutex::new(std::collections::HashSet::new())),
            researcher_inflight_nodes: Arc::new(Mutex::new(std::collections::HashSet::new())),
            conversation_loop_handles: Vec::new(),
            current_run_context: None,
            current_manifest: None,
            manifest_export_path: "runs/exported-manifest.json".to_string(),
            manifest_import_path: "runs/import-manifest.json".to_string(),
            manifest_status_message: String::new(),
            read_only_replay_mode: false,
            theme_applied: false,
            nodes_panel: nodes_panel::NodesPanelState::default(),
        }
    }
}

impl eframe::App for AMSAgents {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.theme_applied {
            catppuccin_egui::set_theme(ctx, catppuccin_egui::LATTE);
            self.theme_applied = true;
        }
        // Auto-refresh model list on startup
        if self.ollama_models.lock().unwrap().is_empty() && !*self.ollama_models_loading.lock().unwrap() {
            *self.ollama_models_loading.lock().unwrap() = true;
            let models_arc = self.ollama_models.clone();
            let loading_arc = self.ollama_models_loading.clone();
            let ctx = ctx.clone();
            let handle = self.rt_handle.clone();
            handle.spawn(async move {
                let models = crate::adk_integration::fetch_ollama_models().await.unwrap_or_default();
                *models_arc.lock().unwrap() = models;
                *loading_arc.lock().unwrap() = false;
                ctx.request_repaint();
            });
        }
        let any_evaluator_active = self.evaluators.iter().any(|e| e.active);
        let any_researcher_active = self.researchers.iter().any(|r| r.active);
        if any_evaluator_active || any_researcher_active {
            ctx.request_repaint();
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.set_min_height(ui.available_height());
                self.render_nodes_panel(ui);
            });
        });
    }
}

pub struct AMSAgentsApp {
    ams_agents: AMSAgents,
}

impl AMSAgentsApp {
    pub fn new(rt_handle: Handle) -> Self {
        Self {
            ams_agents: AMSAgents::new(rt_handle),
        }
    }
}

impl eframe::App for AMSAgentsApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        eframe::App::update(&mut self.ams_agents, ctx, frame);
    }
}

