//! Python runtime management panel — issue #16.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use eframe::egui;

use crate::python::{
    create_runtime, default_registry_path, default_runtimes_dir, delete_runtime,
    install_packages_in_runtime, PythonRuntimeSpec, RuntimeRegistry,
};
use crate::agents::AMSAgents;
use crate::ui::AMSAgentsUiState;

impl AMSAgents {
    pub(crate) fn render_python_panel(
        &mut self,
        ui: &mut egui::Ui,
        ui_state: &mut AMSAgentsUiState,
    ) {
        let panel = &mut ui_state.python;

        if let Some(result) = panel.bg_new_runtime.lock().unwrap().take() {
            match result {
                Ok(rt) => {
                    panel.status = format!(
                        "Runtime '{}' created (Python {}).",
                        rt.label, rt.python_version
                    );
                    panel.active_runtime = Some(rt);
                }
                Err(e) => {
                    panel.status = format!("Error: {e}");
                }
            }
            panel.op_running.store(false, Ordering::Relaxed);
        }
        if let Some(msg) = panel.bg_msg.lock().unwrap().take() {
            panel.status = msg;
            panel.op_running.store(false, Ordering::Relaxed);
        }
        if panel.bg_destroyed.swap(false, Ordering::Relaxed) {
            panel.active_runtime = None;
            panel.status = "Runtime destroyed and removed.".to_string();
            panel.op_running.store(false, Ordering::Relaxed);
        }

        let running = panel.op_running.load(Ordering::Relaxed);
        if running {
            ui.ctx().request_repaint();
        }

        egui::ScrollArea::vertical().show(ui, |ui| {

            if panel.active_runtime.is_none() {
                ui.label(egui::RichText::new("Python Environment").strong());
                ui.separator();
                ui.add_space(4.0);

                egui::Grid::new("py_create_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Label:");
                        ui.add_enabled(
                            !running,
                            egui::TextEdit::singleline(&mut panel.label_input)
                                .desired_width(220.0)
                                .hint_text("StroopTask-py3.11"),
                        );
                        ui.end_row();

                        ui.label("Interpreter:");
                        ui.add_enabled(
                            !running,
                            egui::TextEdit::singleline(&mut panel.interpreter_input)
                                .desired_width(220.0),
                        );
                        ui.end_row();
                    });

                ui.add_space(6.0);

                if running {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Creating environment");
                    });
                } else {
                    let label_ok = !panel.label_input.trim().is_empty();
                    if ui
                        .add_enabled(label_ok, egui::Button::new("  Create env  "))
                        .on_disabled_hover_text("Enter a label first")
                        .clicked()
                    {
                        let label = panel.label_input.trim().to_string();
                        let interpreter = panel.interpreter_input.trim().to_string();
                        let bg_rt = Arc::clone(&panel.bg_new_runtime);
                        let runtimes_dir = default_runtimes_dir();
                        panel.op_running.store(true, Ordering::Relaxed);
                        panel.status = "Creating…".to_string();
                        self.rt_handle.spawn_blocking(move || {
                            let spec = PythonRuntimeSpec {
                                base_interpreter: interpreter,
                                requirements: vec![],
                                post_install_commands: vec![],
                            };
                            let result = create_runtime(spec, &label, "ui", &runtimes_dir)
                                .and_then(|rt| {
                                    let reg_path = default_registry_path();
                                    let mut reg =
                                        RuntimeRegistry::load(&reg_path).unwrap_or_default();
                                    reg.runtimes.push(rt.clone());
                                    reg.save(&reg_path)?;
                                    Ok(rt)
                                })
                                .map_err(|e| e.to_string());
                            *bg_rt.lock().unwrap() = Some(result);
                        });
                    }
                }
            } else {
                // ── Active runtime → info + install + destroy ─────────────
                // Clone display data upfront to release the borrow on python_active_runtime
                // so button click handlers can freely borrow other self fields.
                let (rt_id, rt_label, rt_version, rt_path, rt_state, rt_cloned) = {
                    let Some(rt) = panel.active_runtime.as_ref() else {
                        return;
                    };
                    (
                        rt.id.clone(),
                        rt.label.clone(),
                        rt.python_version.clone(),
                        rt.root_path
                            .as_ref()
                            .map_or_else(|| "—".to_string(), |p| p.to_string_lossy().to_string()),
                        format!("{:?}", rt.state),
                        rt.clone(),
                    )
                };

                ui.label(egui::RichText::new("Active Runtime").strong().size(16.0));
                ui.separator();
                egui::Grid::new("py_rt_info")
                    .num_columns(2)
                    .spacing([8.0, 2.0])
                    .show(ui, |ui| {
                        ui.label("ID:");
                        ui.label(egui::RichText::new(&rt_id).monospace());
                        ui.end_row();
                        ui.label("Label:");
                        ui.label(&rt_label);
                        ui.end_row();
                        ui.label("Python:");
                        ui.label(&rt_version);
                        ui.end_row();
                        ui.label("Path:");
                        ui.label(&rt_path);
                        ui.end_row();
                        ui.label("State:");
                        ui.label(&rt_state);
                        ui.end_row();
                    });

                ui.add_space(8.0);

                // ── Install packages ──────────────────────────────────────
                ui.label(egui::RichText::new("Install Packages").strong().size(16.0));
                ui.separator();
                ui.label(
                    egui::RichText::new("One package per line, e.g.  numpy>=1.26")
                        .small()
                        .weak(),
                );
                ui.add_space(2.0);
                ui.add_enabled(
                    !running,
                    egui::TextEdit::multiline(&mut panel.pkg_input)
                        .desired_width(f32::INFINITY)
                        .desired_rows(4)
                        .hint_text("numpy>=1.26\npsychopy==2024.1.0"),
                );
                ui.add_space(4.0);

                if running {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Working…");
                    });
                } else {
                    ui.horizontal(|ui| {
                        // Install
                        if ui.button("Install").clicked() {
                            let packages: Vec<String> = panel
                                .pkg_input
                                .lines()
                                .map(str::trim)
                                .filter(|l| !l.is_empty())
                                .map(String::from)
                                .collect();
                            if !packages.is_empty() {
                                let rt = rt_cloned.clone();
                                let bg_msg = Arc::clone(&panel.bg_msg);
                                panel.op_running.store(true, Ordering::Relaxed);
                                panel.status = "Installing".to_string();
                                self.rt_handle.spawn_blocking(move || {
                                    let result =
                                        install_packages_in_runtime(&rt, &packages)
                                            .map_or_else(
                                                |e| format!("Error: {e}"),
                                                |out| format!("Install complete.\n{}", out.trim()),
                                            );
                                    *bg_msg.lock().unwrap() = Some(result);
                                });
                            }
                        }

                        ui.add_space(8.0);

                        // Open interactive terminal with venv activated
                        if ui.button("Open Env").on_hover_text("Open a terminal with this venv activated").clicked() {
                            open_runtime_terminal(&rt_cloned, &mut panel.status);
                        }

                        ui.add_space(8.0);

                        // Destroy
                        let destroy_btn = ui.add(egui::Button::new(
                            egui::RichText::new("Destroy env")
                                .color(ui.visuals().error_fg_color),
                        ));
                        if destroy_btn.clicked() {
                            let rt_id_del = rt_id.clone();
                            let bg_msg = Arc::clone(&panel.bg_msg);
                            let bg_destroyed = Arc::clone(&panel.bg_destroyed);
                            panel.op_running.store(true, Ordering::Relaxed);
                            panel.status = "Destroying".to_string();
                            self.rt_handle.spawn_blocking(move || {
                                let reg_path = default_registry_path();
                                let outcome =
                                    RuntimeRegistry::load(&reg_path).and_then(|mut reg| {
                                        delete_runtime(&mut reg, &rt_id_del)?;
                                        reg.save(&reg_path)
                                    });
                                match outcome {
                                    Ok(()) => bg_destroyed.store(true, Ordering::Relaxed),
                                    Err(e) => {
                                        *bg_msg.lock().unwrap() = Some(format!("Error: {e}"));
                                    }
                                }
                            });
                        }
                    });
                }
            }

            // ── Status bar ────────────────────────────────────────────────
            if !panel.status.is_empty() {
                ui.add_space(8.0);
                ui.separator();
                ui.label(egui::RichText::new(&panel.status).small().weak());
            }
        });
    }
}

// ─── Terminal launcher ──────────────────────────────────────────────────────

/// Launch a terminal emulator running the venv's Python interpreter interactively.
///
/// Each emulator receives the venv's `python` binary as the command to execute,
/// so the user lands directly in a Python REPL with that environment active.
fn open_runtime_terminal(
    runtime: &crate::python::PythonRuntime,
    status: &mut String,
) {
    let root = match &runtime.root_path {
        Some(p) => p.clone(),
        None => {
            *status = "Error: runtime has no path (deleted?).".to_string();
            return;
        }
    };

    let bin_dir = if cfg!(windows) { root.join("Scripts") } else { root.join("bin") };
    let python_bin = bin_dir.join(if cfg!(windows) { "python.exe" } else { "python" });
    let python_str  = python_bin.to_string_lossy().to_string();
    let venv_str    = root.to_string_lossy().to_string();
    let system_path = std::env::var("PATH").unwrap_or_default();
    let path_sep = if cfg!(windows) { ';' } else { ':' };
    let venv_path   = format!("{}{}{}", bin_dir.display(), path_sep, system_path);

    #[cfg(target_os = "macos")]
    {
        // Tell Terminal.app to open a new window running the venv python directly.
        let script = format!("tell application \"Terminal\" to do script \"'{python_str}'\"");
        match std::process::Command::new("osascript").args(["-e", &script]).spawn() {
            Ok(_) => *status = "Opened Python REPL in Terminal.app.".to_string(),
            Err(e) => *status = format!("Could not open Terminal.app: {e}"),
        }
    }

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;

        // Windows 11 reports NT 10.0 with build >= 22000.
        fn is_windows_11() -> bool {
            let out = std::process::Command::new("cmd")
                .args(["/C", "ver"])
                .output();
            let text = match out {
                Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
                Err(_) => return false,
            };
            let version = match text
                .split("Version")
                .nth(1)
                .map(|s| s.trim())
                .and_then(|s| s.trim_end_matches(']').split_whitespace().next())
            {
                Some(v) => v,
                None => return false,
            };
            let mut parts = version.split('.');
            let _major = parts.next();
            let _minor = parts.next();
            let build = parts.next().and_then(|p| p.parse::<u32>().ok()).unwrap_or(0);
            build >= 22_000
        }

        let mut cmd = std::process::Command::new("cmd");
        cmd.args(["/k", &python_str])
            .env("VIRTUAL_ENV", &venv_str)
            .env("PATH", &venv_path);

        let win11 = is_windows_11();

        if win11 {
            const CREATE_NEW_CONSOLE: u32 = 0x0000_0010;
            cmd.creation_flags(CREATE_NEW_CONSOLE);
        }

        match cmd.spawn() {
            Ok(_) => {
                if win11 {
                    *status = "Opened Python REPL in a new cmd.exe window (Windows 11).".to_string();
                } else {
                    *status = "Opened Python REPL in cmd.exe.".to_string();
                }
            }
            Err(e) => *status = format!("Could not open cmd.exe: {e}"),
        }
    }

    #[cfg(not(any(target_os = "macos", windows)))]
    {
        // Most Linux/BSD terminal emulators accept `-e <cmd>` to run a command
        // instead of the default shell. gnome-terminal uses `--` as separator.
        let user_term = std::env::var("TERMINAL").unwrap_or_default();

        // (emulator binary, args that come before the python path)
        let mut candidates: Vec<(String, Vec<String>)> = Vec::new();
        if !user_term.is_empty() {
            candidates.push((user_term, vec!["-e".into()]));
        }
        candidates.extend([
            ("x-terminal-emulator".into(), vec!["-e".into()]),
            ("gnome-terminal".into(),      vec!["--".into()]),
            ("xfce4-terminal".into(),      vec!["-e".into()]),
            ("konsole".into(),             vec!["-e".into()]),
            ("xterm".into(),               vec!["-e".into()]),
        ]);

        for (prog, pre_args) in &candidates {
            let mut args = pre_args.clone();
            args.push(python_str.clone());
            let result = std::process::Command::new(prog)
                .args(&args)
                .env("VIRTUAL_ENV", &venv_str)
                .env("PATH", &venv_path)
                .env("VIRTUAL_ENV_DISABLE_PROMPT", "1")
                .spawn();
            match result {
                Ok(_) => {
                    *status = "Opened Python REPL in terminal.".to_string();
                    return;
                }
                Err(_) => continue,
            }
        }
        *status = "Could not find a terminal emulator. Set $TERMINAL or install xterm.".to_string();
    }
}
