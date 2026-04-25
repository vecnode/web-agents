# Startup and UI Flow

## Startup path

The binary initializes Tokio, optionally starts the embedded web server, then launches the egui app.

```rust
fn main() -> eframe::Result<()> {
    let rt = Arc::new(Runtime::new().expect("Failed to create Tokio runtime"));
    let rt_handle = rt.handle().clone();

    std::thread::spawn(move || {
        rt.block_on(async {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
            }
        });
    });

    if web::start_embedded_server_if_enabled(&rt_handle) {
        eprintln!("Embedded Rocket server enabled (AMS_WEB_ENABLED=true)");
    }

    eframe::run_native(
        "arp-cogsci",
        eframe::NativeOptions::default(),
        Box::new(move |_cc| Ok(Box::new(AMSAgentsApp::new(rt_handle.clone())))),
    )
}
```

## Frame loop responsibilities

`AMSAgentsApp` performs three key tasks on each frame:

1. Prepare visual shell (theme and fonts).
2. Enforce vault unlock before rendering the workspace.
3. Render the nodes panel and feature tabs.

```rust
impl eframe::App for AMSAgentsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        prepare_shell(ctx, &mut self.theme_applied, &mut self.phosphor_fonts_installed);

        if !self.vault.is_unlocked() {
            egui::CentralPanel::default().show(ctx, |ui| {
                self.vault.show_unlock_ui(ui);
            });
            return;
        }

        refresh_ollama_models_on_startup(&self.ams_agents, &mut self.ui_state, ctx);
        egui::CentralPanel::default().show(ctx, |ui| {
            self.ams_agents.render_nodes_panel(ui, &mut self.ui_state);
        });
    }
}
```

The UI is intentionally thin: orchestration and business logic live in `AMSAgents` and related modules.
