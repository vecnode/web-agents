use eframe::egui;
use tokio::runtime::Handle;

pub struct MyApp {
    rt_handle: Handle,
}

impl MyApp {
    pub fn new(rt_handle: Handle) -> Self {
        Self { rt_handle }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.add_space(20.0);
                
                if ui.button("Hello").clicked() {
                    println!("Hello");
                }
                
                ui.add_space(10.0);
                
                if ui.button("Test Ollama").clicked() {
                    println!("Testing Ollama integration");
                    let ctx = ctx.clone();
                    let handle = self.rt_handle.clone();
                    handle.spawn(async move {
                        match crate::adk_integration::test_ollama().await {
                            Ok(_response) => {
                                // Response is already printed during streaming in test_ollama()
                            }
                            Err(e) => {
                                eprintln!("Ollama error: {}", e);
                            }
                        }
                        ctx.request_repaint();
                    });
                }
            });
        });
    }
}
