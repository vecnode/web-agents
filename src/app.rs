use crate::agent_entities::{Evaluator, Researcher};
use crate::event_ledger::EventLedger;
use crate::reproducibility::{RunContext, RunManifest};
use eframe::egui;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, Mutex};
use tokio::runtime::Handle;

mod nodes_panel;
mod settings_panel;

use crate::vault::MasterVault;

pub struct AMSAgents {
    rt_handle: Handle,
    ollama_models: Arc<Mutex<Vec<String>>>,
    ollama_models_loading: Arc<Mutex<bool>>,
    selected_ollama_model: String,
    evaluators: Vec<Evaluator>,
    researchers: Vec<Researcher>,
    conversation_history_size: usize,
    /// Base URL for the Ollama API (e.g. http://127.0.0.1:11434).
    ollama_host: String,
    http_endpoint: String,
    last_message_in_chat: Arc<Mutex<Option<String>>>,
    conversation_message_events: Arc<Mutex<Vec<String>>>,
    last_evaluated_message_by_evaluator: Arc<Mutex<std::collections::HashMap<usize, String>>>,
    last_researched_message_by_researcher: Arc<Mutex<std::collections::HashMap<usize, String>>>,
    evaluator_event_queues:
        Arc<Mutex<std::collections::HashMap<usize, std::collections::VecDeque<(String, String)>>>>,
    researcher_event_queues:
        Arc<Mutex<std::collections::HashMap<usize, std::collections::VecDeque<(String, String)>>>>,
    evaluator_inflight_nodes: Arc<Mutex<std::collections::HashSet<usize>>>,
    researcher_inflight_nodes: Arc<Mutex<std::collections::HashSet<usize>>>,
    conversation_loop_handles: Vec<(usize, Arc<Mutex<bool>>, tokio::task::JoinHandle<()>)>,
    /// Incremented on Stop (and at each run restart) so in-flight Ollama streams can exit promptly.
    pub(super) ollama_run_epoch: Arc<AtomicU64>,
    current_run_context: Option<RunContext>,
    current_manifest: Option<RunManifest>,
    manifest_export_path: String,
    manifest_import_path: String,
    /// JSON path for Agents tab Load / Save (graph + runtime).
    agents_workspace_path: String,
    /// Target path for "Download Run Bundle" (zip).
    pub(super) bundle_export_path: String,
    /// Append-only ledger for the active run (`events.jsonl`), if any.
    pub(super) event_ledger: Option<Arc<EventLedger>>,
    manifest_status_message: String,
    read_only_replay_mode: bool,
    /// True while a started play is active (cleared on Stop or when all conversation tasks finish).
    conversation_graph_running: Arc<AtomicBool>,
    /// Bumped on each `run_graph`; stale conversation tasks ignore completion.
    conversation_run_generation: Arc<AtomicU64>,
    theme_applied: bool,
    phosphor_fonts_installed: bool,
    nodes_panel: nodes_panel::NodesPanelState,
}

impl AMSAgents {
    pub fn new(rt_handle: Handle) -> Self {
        Self {
            rt_handle,
            ollama_models: Arc::new(Mutex::new(Vec::new())),
            ollama_models_loading: Arc::new(Mutex::new(false)),
            selected_ollama_model: std::env::var("OLLAMA_MODEL").unwrap_or_default(),
            evaluators: Vec::new(),
            researchers: Vec::new(),
            conversation_history_size: 5,
            ollama_host: std::env::var("OLLAMA_HOST")
                .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string()),
            http_endpoint: std::env::var("CONVERSATION_HTTP_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:3000/".to_string()),
            last_message_in_chat: Arc::new(Mutex::new(None)),
            conversation_message_events: Arc::new(Mutex::new(Vec::new())),
            last_evaluated_message_by_evaluator: Arc::new(Mutex::new(
                std::collections::HashMap::new(),
            )),
            last_researched_message_by_researcher: Arc::new(Mutex::new(
                std::collections::HashMap::new(),
            )),
            evaluator_event_queues: Arc::new(Mutex::new(std::collections::HashMap::new())),
            researcher_event_queues: Arc::new(Mutex::new(std::collections::HashMap::new())),
            evaluator_inflight_nodes: Arc::new(Mutex::new(std::collections::HashSet::new())),
            researcher_inflight_nodes: Arc::new(Mutex::new(std::collections::HashSet::new())),
            conversation_loop_handles: Vec::new(),
            ollama_run_epoch: Arc::new(AtomicU64::new(0)),
            current_run_context: None,
            current_manifest: None,
            manifest_export_path: "runs/exported-manifest.json".to_string(),
            manifest_import_path: "runs/import-manifest.json".to_string(),
            agents_workspace_path: "runs/agents-workspace.json".to_string(),
            bundle_export_path: "runs/run-bundle.zip".to_string(),
            event_ledger: None,
            manifest_status_message: String::new(),
            read_only_replay_mode: false,
            conversation_graph_running: Arc::new(AtomicBool::new(false)),
            conversation_run_generation: Arc::new(AtomicU64::new(0)),
            theme_applied: false,
            phosphor_fonts_installed: false,
            nodes_panel: nodes_panel::NodesPanelState::default(),
        }
    }

    pub(crate) fn prepare_shell(&mut self, ctx: &egui::Context) {
        if !self.theme_applied {
            catppuccin_egui::set_theme(ctx, catppuccin_egui::LATTE);
            self.theme_applied = true;
        }
        if !self.phosphor_fonts_installed {
            let mut fonts = ctx.fonts(|f| f.definitions().clone());
            egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Regular);
            ctx.set_fonts(fonts);
            self.phosphor_fonts_installed = true;
        }
    }
}

impl eframe::App for AMSAgents {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.prepare_shell(ctx);
        // Auto-refresh model list on startup
        if self.ollama_models.lock().unwrap().is_empty()
            && !*self.ollama_models_loading.lock().unwrap()
        {
            *self.ollama_models_loading.lock().unwrap() = true;
            let models_arc = self.ollama_models.clone();
            let loading_arc = self.ollama_models_loading.clone();
            let ctx = ctx.clone();
            let handle = self.rt_handle.clone();
            let ollama_host = self.ollama_host.clone();
            handle.spawn(async move {
                let models = crate::adk_integration::fetch_ollama_models(&ollama_host)
                    .await
                    .unwrap_or_default();
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
    vault: MasterVault,
    ams_agents: AMSAgents,
}

impl AMSAgentsApp {
    pub fn new(rt_handle: Handle) -> Self {
        Self {
            vault: MasterVault::new(),
            ams_agents: AMSAgents::new(rt_handle),
        }
    }
}

impl eframe::App for AMSAgentsApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.ams_agents.prepare_shell(ctx);

        if !self.vault.is_unlocked() {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(40.0);
                    self.vault.show_unlock_ui(ui);
                });
            });
            return;
        }

        egui::TopBottomPanel::top("master_vault_lock_bar")
            .frame(egui::Frame::NONE.inner_margin(egui::Margin::same(6)))
            .show(ctx, |ui| {
                if self.vault.show_lock_bar(ui) {
                    self.vault.lock();
                }
            });

        eframe::App::update(&mut self.ams_agents, ctx, frame);
    }
}
