//! Master-password gate (Argon2) and a small [`Vault`] stub for future encrypted data.
//!
//! Configure with `AMS_MASTER_HASH` (PHC string) or a file at `runs/.master_hash` (first line).
//! Set `AMS_SKIP_VAULT=1` to disable the gate (development only).

use argon2::{Argon2, PasswordHash, PasswordVerifier};
use eframe::egui::{self, TextEdit};
use secrecy::{ExposeSecret, SecretString};

const DEFAULT_HASH_FILE: &str = "runs/.master_hash";

/// Placeholder store; extend with encrypted blobs, entries, etc.
#[derive(Default)]
pub struct Vault {
    // Future: encrypted payload handle, key material wrapped after unlock, etc.
}

/// Loads when the app starts; verify with Argon2 against a stored PHC hash.
pub struct MasterVault {
    /// PHC-format Argon2 hash string, or empty if misconfigured.
    stored_hash: String,
    master_input: SecretString,
    pub unlocked: bool,
    status: String,
    /// When true, [`Self::unlocked`] is always treated as true.
    skip_vault: bool,
    /// Future hook: populate after successful unlock.
    pub vault: Vault,
}

impl MasterVault {
    pub fn new() -> Self {
        let skip_vault = std::env::var("AMS_SKIP_VAULT").unwrap_or_default() == "1";
        let stored_hash = std::env::var("AMS_MASTER_HASH")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| {
                std::fs::read_to_string(DEFAULT_HASH_FILE)
                    .ok()
                    .map(|s| s.lines().next().unwrap_or("").trim().to_string())
                    .filter(|s| !s.is_empty())
            })
            .unwrap_or_default();

        Self {
            stored_hash,
            master_input: SecretString::new(String::new().into_boxed_str()),
            unlocked: false,
            status: String::new(),
            skip_vault,
            vault: Vault::default(),
        }
    }

    pub fn is_unlocked(&self) -> bool {
        self.skip_vault || self.unlocked
    }

    pub fn has_configured_hash(&self) -> bool {
        self.skip_vault || !self.stored_hash.is_empty()
    }

    fn try_unlock(&mut self) {
        self.status.clear();
        if self.skip_vault {
            self.unlocked = true;
            return;
        }
        if self.stored_hash.is_empty() {
            self.status =
                "No master hash configured. Set AMS_MASTER_HASH or create runs/.master_hash."
                    .to_string();
            self.unlocked = false;
            return;
        }

        let argon2 = Argon2::default();
        let parsed_hash = match PasswordHash::new(&self.stored_hash) {
            Ok(h) => h,
            Err(e) => {
                self.status = format!("Invalid stored hash: {e}");
                return;
            }
        };

        let password = self.master_input.expose_secret();
        match argon2.verify_password(password.as_bytes(), &parsed_hash) {
            Ok(()) => {
                self.unlocked = true;
                self.status = "Unlocked.".to_string();
            }
            Err(_) => {
                self.unlocked = false;
                self.status = "Wrong master password.".to_string();
            }
        }
    }

    pub fn lock(&mut self) {
        self.unlocked = false;
        self.master_input = SecretString::new(String::new().into_boxed_str());
        self.status.clear();
    }

    /// Full-screen unlock UI (CentralPanel).
    pub fn show_unlock_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Unlock");
        if self.skip_vault {
            ui.label("Vault gate is disabled (AMS_SKIP_VAULT=1).");
            return;
        }
        if self.stored_hash.is_empty() {
            ui.label(self.status.clone());
            ui.label(format!(
                "Create `{}` with one line (PHC Argon2 hash), or set AMS_MASTER_HASH.",
                DEFAULT_HASH_FILE
            ));
            return;
        }

        ui.label("Master password:");
        let mut plain: String = self.master_input.expose_secret().to_owned();
        let resp = ui.add(
            TextEdit::singleline(&mut plain)
                .password(true)
                .hint_text("••••••••"),
        );
        if resp.changed() {
            self.master_input = SecretString::new(plain.into_boxed_str());
        }

        if ui.button("Unlock").clicked()
            || (resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)))
        {
            self.try_unlock();
        }

        if !self.status.is_empty() {
            ui.label(egui::RichText::new(&self.status).weak());
        }
    }

    /// Thin top strip with Lock; call from a `TopBottomPanel::top`. Returns true if user locked.
    pub fn show_lock_bar(&mut self, ui: &mut egui::Ui) -> bool {
        if self.skip_vault {
            return false;
        }
        let mut clicked = false;
        ui.horizontal(|ui| {
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), 0.0),
                egui::Layout::right_to_left(egui::Align::Center),
                |ui| {
                    if ui.button("Lock").clicked() {
                        clicked = true;
                    }
                },
            );
        });
        clicked
    }
}
