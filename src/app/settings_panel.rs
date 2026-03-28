use super::AMSAgents;
use eframe::egui;
use std::path::PathBuf;

impl AMSAgents {
    pub(super) fn render_ollama_settings_widgets(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
    ) {
        ui.vertical(|ui| {
            let settings_fold = egui::collapsing_header::CollapsingState::load_with_default_open(
                ui.ctx(),
                ui.make_persistent_id("ollama_section_settings"),
                true,
            )
            .show_header(ui, |ui| {
                ui.label(egui::RichText::new("Settings Ollama").strong());
            });
            let _ = settings_fold.body(|ui| {
                ui.horizontal(|ui| {
                    ui.label("API host / URL:");
                    ui.add(egui::TextEdit::singleline(&mut self.ollama_host).desired_width(300.0));
                });
            });

            let test_fold = egui::collapsing_header::CollapsingState::load_with_default_open(
                ui.ctx(),
                ui.make_persistent_id("ollama_section_test"),
                true,
            )
            .show_header(ui, |ui| {
                ui.label(egui::RichText::new("Test Ollama").strong());
            });
            let _ = test_fold.body(|ui| {
                ui.horizontal(|ui| {
                    ui.label("Ollama Model:");
                    let models = self.ollama_models.lock().unwrap().clone();
                    if self.selected_ollama_model.is_empty() {
                        if let Some(first) = models.first() {
                            self.selected_ollama_model = first.clone();
                        }
                    }
                    egui::ComboBox::from_id_salt("ollama_model_selector")
                        .selected_text(if self.selected_ollama_model.is_empty() {
                            "Select model".to_string()
                        } else {
                            self.selected_ollama_model.clone()
                        })
                        .show_ui(ui, |ui| {
                            for model in &models {
                                ui.selectable_value(
                                    &mut self.selected_ollama_model,
                                    model.clone(),
                                    model,
                                );
                            }
                        });

                    let loading = *self.ollama_models_loading.lock().unwrap();
                    if ui
                        .button(if loading { "Loading" } else { "Refresh" })
                        .clicked()
                        && !loading
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
                });
                ui.add_space(5.0);

                ui.horizontal(|ui| {
                    ui.label("Chat HTTP Endpoint:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.http_endpoint).desired_width(260.0),
                    );
                });
                ui.add_space(5.0);

                ui.horizontal(|ui| {
                    if ui.button("Test API").clicked() {
                        println!("Pinging Ollama");
                        let ctx = ctx.clone();
                        let handle = self.rt_handle.clone();
                        let model = self.selected_ollama_model.clone();
                        let ollama_host = self.ollama_host.clone();
                        handle.spawn(async move {
                            match crate::adk_integration::test_ollama(
                                ollama_host.as_str(),
                                if model.trim().is_empty() {
                                    None
                                } else {
                                    Some(model.as_str())
                                },
                            )
                            .await
                            {
                                Ok(_) => {}
                                Err(e) => eprintln!("Ollama error: {}", e),
                            }
                            ctx.request_repaint();
                        });
                    }
                });
                ui.add_space(5.0);
                ui.horizontal(|ui| {
                    ui.label("Turn Delay (s):");
                    ui.add(
                        egui::DragValue::new(&mut self.conversation_turn_delay_secs)
                            .range(0..=60)
                            .speed(0.1),
                    );
                    ui.label("History:");
                    ui.add(
                        egui::DragValue::new(&mut self.conversation_history_size)
                            .range(1..=50)
                            .speed(0.1),
                    );
                });
            });
        });
    }

    pub(super) fn render_reproducibility_settings_widgets(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.label(egui::RichText::new("Reproducibility").strong().size(12.0));
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Export Manifest Path:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.manifest_export_path).desired_width(300.0),
                );
                if ui.button("Export Manifest").clicked() {
                    let export_path = PathBuf::from(self.manifest_export_path.clone());
                    if let Err(e) = self.export_manifest_to_path(export_path) {
                        self.manifest_status_message = format!("Manifest export failed: {e}");
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("Run From Manifest Path:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.manifest_import_path).desired_width(300.0),
                );
                if ui.button("Run From Manifest").clicked() {
                    let import_path = PathBuf::from(self.manifest_import_path.clone());
                    if let Err(e) = self.run_from_manifest_path(import_path) {
                        self.manifest_status_message = format!("Manifest run failed: {e}");
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("Run bundle (zip) path:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.bundle_export_path).desired_width(300.0),
                );
                if ui
                    .button("Download Run Bundle")
                    .on_hover_text(
                        "Zip manifest.json, events.jsonl, and summary.json for the current run.",
                    )
                    .clicked()
                {
                    let p = PathBuf::from(self.bundle_export_path.trim());
                    if let Err(e) = self.download_run_bundle_to_path(p) {
                        self.manifest_status_message = format!("Run bundle failed: {e}");
                    }
                }
            });
            if !self.manifest_status_message.trim().is_empty() {
                ui.label(&self.manifest_status_message);
            }
            if self.read_only_replay_mode {
                ui.label("Replay mode is active.");
            }
        });
    }
}
