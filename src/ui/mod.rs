use crate::agents::AMSAgents;
use crate::vault::MasterVault;
use eframe::egui;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use tokio::runtime::Handle;

mod nodes_panel;
mod python_panel;
mod settings_panel;

pub(crate) struct OllamaUiState {
	pub(crate) models: Arc<Mutex<Vec<String>>>,
	pub(crate) models_loading: Arc<Mutex<bool>>,
	pub(crate) test_status: Arc<Mutex<String>>,
	pub(crate) test_running: Arc<AtomicBool>,
}

impl Default for OllamaUiState {
	fn default() -> Self {
		Self {
			models: Arc::new(Mutex::new(Vec::new())),
			models_loading: Arc::new(Mutex::new(false)),
			test_status: Arc::new(Mutex::new(String::new())),
			test_running: Arc::new(AtomicBool::new(false)),
		}
	}
}

pub(crate) struct PythonPanelUiState {
	pub(crate) label_input: String,
	pub(crate) interpreter_input: String,
	pub(crate) pkg_input: String,
	pub(crate) active_runtime: Option<crate::python::PythonRuntime>,
	pub(crate) op_running: Arc<AtomicBool>,
	pub(crate) status: String,
	pub(crate) bg_new_runtime:
		Arc<Mutex<Option<Result<crate::python::PythonRuntime, String>>>>,
	pub(crate) bg_msg: Arc<Mutex<Option<String>>>,
	pub(crate) bg_destroyed: Arc<AtomicBool>,
}

impl Default for PythonPanelUiState {
	fn default() -> Self {
		Self {
			label_input: String::new(),
			interpreter_input: "python3".to_string(),
			pkg_input: String::new(),
			active_runtime: None,
			op_running: Arc::new(AtomicBool::new(false)),
			status: String::new(),
			bg_new_runtime: Arc::new(Mutex::new(None)),
			bg_msg: Arc::new(Mutex::new(None)),
			bg_destroyed: Arc::new(AtomicBool::new(false)),
		}
	}
}

pub(crate) struct AMSAgentsUiState {
	pub(crate) ollama: OllamaUiState,
	pub(crate) agents_workspace_path: String,
	pub(crate) manifest_status_message: String,
	pub(crate) python: PythonPanelUiState,
	pub(crate) inbox_messages: Vec<(String, String)>, // (timestamp, message)
	pub(crate) inbox_input: String,
}

impl Default for AMSAgentsUiState {
	fn default() -> Self {
		Self {
			ollama: OllamaUiState::default(),
			agents_workspace_path: String::new(),
			manifest_status_message: String::new(),
			python: PythonPanelUiState::default(),
			inbox_messages: vec![(
				chrono::Local::now().format("%H:%M:%S").to_string(),
				"Welcome to your inbox!".to_string(),
			)],
			inbox_input: String::new(),
		}
	}
}

fn prepare_shell(
	ctx: &egui::Context,
	theme_applied: &mut bool,
	phosphor_fonts_installed: &mut bool,
) {
	if !*theme_applied {
		catppuccin_egui::set_theme(ctx, catppuccin_egui::LATTE);
		*theme_applied = true;
	}
	if !*phosphor_fonts_installed {
		let mut fonts = ctx.fonts(|f| f.definitions().clone());
		egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Regular);
		ctx.set_fonts(fonts);
		*phosphor_fonts_installed = true;
	}
}

fn refresh_ollama_models_on_startup(
	ams_agents: &AMSAgents,
	ui_state: &mut AMSAgentsUiState,
	ctx: &egui::Context,
) {
	if ui_state.ollama.models.lock().unwrap().is_empty()
		&& !*ui_state.ollama.models_loading.lock().unwrap()
	{
		*ui_state.ollama.models_loading.lock().unwrap() = true;
		let models_arc = ui_state.ollama.models.clone();
		let loading_arc = ui_state.ollama.models_loading.clone();
		let ctx = ctx.clone();
		let handle = ams_agents.rt_handle.clone();
		let ollama_host = ams_agents.ollama_host.clone();
		handle.spawn(async move {
			let models = crate::ollama::fetch_ollama_models(&ollama_host)
				.await
				.unwrap_or_default();
			*models_arc.lock().unwrap() = models;
			*loading_arc.lock().unwrap() = false;
			ctx.request_repaint();
		});
	}
}

pub struct AMSAgentsApp {
	vault: MasterVault,
	ams_agents: AMSAgents,
	ui_state: AMSAgentsUiState,
	theme_applied: bool,
	phosphor_fonts_installed: bool,
}

impl AMSAgentsApp {
	pub fn new(rt_handle: Handle) -> Self {
		Self {
			vault: MasterVault::new(),
			ams_agents: AMSAgents::new(rt_handle),
			ui_state: AMSAgentsUiState {
				agents_workspace_path: "runs/agents-workspace.json".to_string(),
				..Default::default()
			},
			theme_applied: false,
			phosphor_fonts_installed: false,
		}
	}
}

impl eframe::App for AMSAgentsApp {
	fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
		prepare_shell(
			ctx,
			&mut self.theme_applied,
			&mut self.phosphor_fonts_installed,
		);

		if !self.vault.is_unlocked() {
			egui::CentralPanel::default().show(ctx, |ui| {
				ui.vertical_centered(|ui| {
					ui.add_space(40.0);
					self.vault.show_unlock_ui(ui);
				});
			});
			return;
		}

		egui::TopBottomPanel::top("master_vault_lock_bar").show(ctx, |ui| {
			if self.vault.show_lock_bar(ui) {
				self.vault.lock();
			}
		});

		refresh_ollama_models_on_startup(&self.ams_agents, &mut self.ui_state, ctx);
		egui::CentralPanel::default().show(ctx, |ui| {
			ui.vertical(|ui| {
				ui.set_min_height(ui.available_height());
				self.ams_agents.render_nodes_panel(ui, &mut self.ui_state);
			});
		});
	}
}
