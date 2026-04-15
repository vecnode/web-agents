use ams_agents::{AMSAgentsApp, web};
use std::sync::Arc;
use tokio::runtime::Runtime;

fn main() -> eframe::Result<()> {
    // Create a Tokio runtime for async operations
    let rt = Arc::new(Runtime::new().expect("Failed to create Tokio runtime"));
    let rt_handle = rt.handle().clone();

    // Keep the runtime alive by storing it
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

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([900.0, 840.0]),
        ..Default::default()
    };

    eframe::run_native(
        "ams-agents",
        options,
        Box::new(move |_cc| Ok(Box::new(AMSAgentsApp::new(rt_handle.clone())))),
    )
}
