mod app;
mod adk_integration;
mod http_client;
mod conversation_loop;

use app::MyApp;
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

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([720.0, 860.0]),
        ..Default::default()
    };

    eframe::run_native(
        "web-agents",
        options,
        Box::new(move |_cc| Ok(Box::new(MyApp::new(rt_handle.clone())))),
    )
}
