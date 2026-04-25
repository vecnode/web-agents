#[derive(Debug, Clone)]
pub struct Room {
    pub id: String,
    pub name: String,
}

impl Room {
    pub fn new(id: String, name: String) -> Self {
        Room { id, name }
    }
}

use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{SystemTime, UNIX_EPOCH};

use egui::{ScrollArea, Ui, Vec2};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use super::incoming::MessageSource;

pub struct UiInbox<T> {
    tx: Sender<T>,
    rx: Receiver<T>,
}

impl<T> UiInbox<T> {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self { tx, rx }
    }

    pub fn sender(&self) -> Sender<T> {
        self.tx.clone()
    }

    pub fn drain(&self) -> Vec<T> {
        let mut out = Vec::new();
        while let Ok(item) = self.rx.try_recv() {
            out.push(item);
        }
        out
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MessageCorrelation {
    pub conversation_id: String,
    pub event_id: String,
    pub request_id: String,
    pub timestamp_rfc3339: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AssistantGeneration {
    pub model: String,
    pub num_predict: Option<i32>,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub content: String,
    pub from: Option<String>,
    pub correlation: Option<MessageCorrelation>,
    pub source: MessageSource,
    pub api_auto_respond: bool,
    pub assistant_generation: Option<AssistantGeneration>,
}

pub type MessageHandler = Box<dyn Fn(String) + Send + Sync>;
pub type MessageCommitHook = Box<dyn Fn(&ChatMessage, &str) + Send + Sync>;

pub struct ChatExample {
    pub messages: Vec<ChatMessage>,
    pub message_timestamps: Vec<String>,
    pub inbox: UiInbox<ChatMessage>,
    pub input_text: String,
    pub message_handler: Option<MessageHandler>,
    pub message_commit_hook: Option<MessageCommitHook>,
    pub waiting_for_response: Arc<std::sync::Mutex<bool>>,
    pub picked_file_path: Option<String>,
    pub main_input_enabled: bool,

    pub rooms: Vec<Room>,
    pub sidebar_open: bool,
    pub selected_room: Option<String>,
}

impl Default for ChatExample {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatExample {
    pub fn set_rooms(&mut self, rooms: Vec<Room>) {
        self.rooms = rooms;
    }

    pub fn add_room(&mut self, name: String, store: &mut super::store::Store) {
        if let Ok(id) = store.create_conversation() {
            let _ = store.rename_conversation(&id, &name);
            self.rooms.push(Room::new(id.clone(), name));
            self.selected_room = Some(id);
        }
    }

    pub fn rename_room(&mut self, id: &str, new_name: String, store: &mut super::store::Store) {
        if let Some(room) = self.rooms.iter_mut().find(|r| r.id == id) {
            room.name = new_name.clone();
        }
        let _ = store.rename_conversation(id, &new_name);
    }

    pub fn sidebar_ui(&mut self, ui: &mut Ui) {
        let mut store = match super::store::Store::open("metrics/overview_chat.sqlite") {
            Ok(s) => s,
            Err(_) => return,
        };

        if ui.button("New room").clicked() {
            let name = format!("Room {}", self.rooms.len() + 1);
            self.add_room(name, &mut store);
        }

        ui.separator();
        let mut delete_room_id: Option<String> = None;
        for room in &self.rooms {
            let selected = self
                .selected_room
                .as_ref()
                .map_or(false, |id| id == &room.id);
            ui.horizontal(|ui| {
                let row_height = 24.0;
                let selectable_width =
                    (ui.available_width() - 20.0 - ui.spacing().item_spacing.x).max(0.0);
                let label = egui::Button::new(&room.name).selected(selected);
                if ui
                    .add_sized(egui::vec2(selectable_width, row_height), label)
                    .clicked()
                {
                    self.selected_room = Some(room.id.clone());
                }
                if ui
                    .add_sized(egui::vec2(20.0, row_height), egui::Button::new("x"))
                    .on_hover_text("Delete room")
                    .clicked()
                {
                    delete_room_id = Some(room.id.clone());
                }
            });
        }

        if let Some(id) = delete_room_id {
            self.rooms.retain(|r| r.id != id);
            let _ = store.delete_conversation(&id);
            if self.selected_room.as_deref() == Some(id.as_str()) {
                self.selected_room = self.rooms.first().map(|r| r.id.clone());
            }
        }
    }

    pub fn current_timestamp_string() -> String {
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let day_secs = now_secs % 86_400;
        let hours = day_secs / 3_600;
        let minutes = (day_secs % 3_600) / 60;
        let seconds = day_secs % 60;
        format!("{hours:02}:{minutes:02}:{seconds:02}")
    }

    pub fn display_time_for_message(msg: &ChatMessage) -> String {
        if let Some(c) = &msg.correlation {
            if let Ok(odt) = OffsetDateTime::parse(&c.timestamp_rfc3339, &Rfc3339) {
                let t = odt.time();
                return format!("{:02}:{:02}:{:02}", t.hour(), t.minute(), t.second());
            }
        }
        Self::current_timestamp_string()
    }

    pub fn default_welcome() -> (ChatMessage, String) {
        let msg = ChatMessage {
            content: "openchat-cogsci Started".to_string(),
            from: Some("System".to_string()),
            correlation: None,
            source: MessageSource::System,
            api_auto_respond: false,
            assistant_generation: None,
        };
        let ts = Self::display_time_for_message(&msg);
        (msg, ts)
    }

    pub fn new() -> Self {
        let inbox = UiInbox::new();
        let (welcome, ts) = Self::default_welcome();
        ChatExample {
            messages: vec![welcome],
            message_timestamps: vec![ts],
            inbox,
            input_text: String::new(),
            message_handler: None,
            message_commit_hook: None,
            waiting_for_response: Arc::new(std::sync::Mutex::new(false)),
            picked_file_path: None,
            main_input_enabled: true,
            rooms: Vec::new(),
            sidebar_open: true,
            selected_room: None,
        }
    }

    pub fn hydrate(&mut self, messages: Vec<ChatMessage>, message_timestamps: Vec<String>) {
        self.messages = messages;
        self.message_timestamps = message_timestamps;
    }

    pub fn set_message_commit_hook(&mut self, hook: Option<MessageCommitHook>) {
        self.message_commit_hook = hook;
    }

    fn commit_message(&mut self, msg: &ChatMessage, display_ts: &str) {
        if let Some(hook) = &self.message_commit_hook {
            hook(msg, display_ts);
        }
    }

    pub fn inbox(&self) -> &UiInbox<ChatMessage> {
        &self.inbox
    }

    pub fn set_message_handler(&mut self, handler: MessageHandler) {
        self.message_handler = Some(handler);
    }

    pub fn waiting_for_response(&self) -> &Arc<std::sync::Mutex<bool>> {
        &self.waiting_for_response
    }

    pub fn set_main_input_enabled(&mut self, enabled: bool) {
        self.main_input_enabled = enabled;
    }

    pub fn export_rows(&self) -> Vec<(String, String, String)> {
        self.messages
            .iter()
            .enumerate()
            .map(|(idx, msg)| {
                let ts = if let Some(c) = &msg.correlation {
                    c.timestamp_rfc3339.clone()
                } else {
                    self.message_timestamps
                        .get(idx)
                        .cloned()
                        .unwrap_or_else(|| "--:--:--".to_string())
                };
                let from = msg.from.clone().unwrap_or_else(|| "Unknown".to_string());
                (ts, from, msg.content.clone())
            })
            .collect()
    }

    #[allow(dead_code)]
    pub fn clear_messages(&mut self) {
        self.messages.clear();
        self.message_timestamps.clear();
    }

    pub fn reset_to_welcome(&mut self) {
        let (w, t) = Self::default_welcome();
        self.messages = vec![w];
        self.message_timestamps = vec![t];
    }

    pub fn ui(&mut self, ui: &mut Ui) {
        for item in self.inbox.drain() {
            let ts = Self::display_time_for_message(&item);
            self.message_timestamps.push(ts);
            self.messages.push(item);
        }

        egui::TopBottomPanel::top("sidebar_toggle").show_inside(ui, |ui| {
            if ui
                .button(if self.sidebar_open { "Hide Rooms" } else { "Show Rooms" })
                .clicked()
            {
                self.sidebar_open = !self.sidebar_open;
            }
        });

        self.sidebar_ui(ui);

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.vertical(|ui| {
                let list_height = (ui.available_height() - 72.0).max(0.0);
                ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .max_height(list_height)
                    .show(ui, |ui| {
                        for (idx, item) in self.messages.iter().enumerate() {
                            let timestamp = self
                                .message_timestamps
                                .get(idx)
                                .map(|s| s.as_str())
                                .unwrap_or("--:--:--");
                            ui.horizontal(|ui| {
                                ui.label(timestamp);
                                ui.separator();
                                ui.vertical(|ui| {
                                    if let Some(from) = &item.from {
                                        ui.label(egui::RichText::new(from).strong());
                                    }
                                    ui.label(&item.content);
                                });
                            });
                        }

                        if *self.waiting_for_response.lock().unwrap() {
                            ui.horizontal(|ui| {
                                ui.spinner();
                                ui.label("Waiting for response...");
                            });
                        }
                    });

                ui.add_enabled_ui(self.main_input_enabled, |ui| {
                    ui.horizontal(|ui| {
                        let response = ui.add_sized(
                            Vec2::new((ui.available_width() - 120.0).max(120.0), 26.0),
                            egui::TextEdit::singleline(&mut self.input_text).hint_text("Type a message"),
                        );
                        let send_button = ui.add_sized(Vec2::new(52.0, 26.0), egui::Button::new("Send"));
                        let send_clicked = send_button.clicked()
                            || (response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)));
                        if send_clicked && !self.input_text.trim().is_empty() {
                            let message_text = self.input_text.trim().to_string();
                            self.input_text.clear();

                            let user_message = ChatMessage {
                                content: message_text.clone(),
                                from: Some("Human".to_string()),
                                correlation: None,
                                source: MessageSource::Human,
                                api_auto_respond: false,
                                assistant_generation: None,
                            };
                            let ts = Self::current_timestamp_string();
                            let persisted = user_message.clone();
                            self.messages.push(user_message);
                            self.message_timestamps.push(ts.clone());
                            self.commit_message(&persisted, &ts);

                            if let Some(room_id) = &self.selected_room {
                                if let Ok(store) = super::store::Store::open("metrics/overview_chat.sqlite") {
                                    let _ = store.append_message(room_id, &persisted, &ts);
                                }
                            }

                            if let Some(handler) = &self.message_handler {
                                handler(message_text);
                            } else {
                                let tx = self.inbox.sender();
                                let bot_message = ChatMessage {
                                    content: "Please select a model".to_string(),
                                    from: Some("System".to_string()),
                                    correlation: None,
                                    source: MessageSource::System,
                                    api_auto_respond: false,
                                    assistant_generation: None,
                                };
                                let _ = tx.send(bot_message);
                            }
                        }
                    });

                    ui.horizontal(|ui| {
                        if ui
                            .add_sized(Vec2::new(28.0, 26.0), egui::Button::new("+"))
                            .clicked()
                        {
                            self.picked_file_path = Some("Attachment selected".to_string());
                        }
                        if let Some(file_path) = &self.picked_file_path {
                            ui.label(egui::RichText::new(format!("File: {file_path}")).small().weak());
                        }
                    });
                });
            });
        });
    }
}
