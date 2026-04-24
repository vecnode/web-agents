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
// Ported from openchat/src/chat.rs
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use egui::{Align, Frame, Layout, ScrollArea, Ui, Vec2};

use egui_inbox::UiInbox;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use super::incoming::MessageSource;

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
	pub source: crate::ui::overview_chat::MessageSource, // Will need to port or alias MessageSource
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

	// Sidebar state
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
				// Persist name
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
			use egui::{RichText, Sense};
			let sidebar_width = 180.0;
			let mut open = &mut self.sidebar_open;
			let mut new_room_name = String::new();
			let mut rename_id: Option<String> = None;
			let mut rename_text = String::new();
			let mut store = super::store::Store::open("metrics/overview_chat.sqlite").unwrap();
			egui::SidePanel::left("rooms_sidebar")
				.resizable(false)
				.min_width(sidebar_width)
				.show_animated(ui.ctx(), *open, |ui| {
					ui.vertical_centered(|ui| {
						ui.heading("Rooms");
					});
					ui.separator();
					// Add new room
					ui.horizontal(|ui| {
						ui.label("New:");
						let resp = ui.text_edit_singleline(&mut new_room_name);
						if ui.button("+").clicked() && !new_room_name.trim().is_empty() {
							self.add_room(new_room_name.clone(), &mut store);
							new_room_name.clear();
						}
					});
					ui.separator();
					for room in &mut self.rooms {
						let selected = self.selected_room.as_ref().map_or(false, |id| id == &room.id);
						let mut editing = false;
						// Double-click to rename
						let resp = ui.selectable_label(selected, RichText::new(&room.name));
						if resp.double_clicked() {
							rename_id = Some(room.id.clone());
							rename_text = room.name.clone();
							editing = true;
						}
						if resp.clicked() {
							self.selected_room = Some(room.id.clone());
						}
						if editing {
							let mut text = rename_text.clone();
							if ui.text_edit_singleline(&mut text).lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
								self.rename_room(&room.id, text.clone(), &mut store);
								rename_id = None;
							}
						}
					}
				});
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

	fn display_time_for_message(msg: &ChatMessage) -> String {
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
			source: crate::ui::overview_chat::MessageSource::System,
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
		// Sidebar and chat main area
		egui::TopBottomPanel::top("sidebar_toggle").show_inside(ui, |ui| {
			if ui.button(if self.sidebar_open { "Hide Rooms" } else { "Show Rooms" }).clicked() {
				self.sidebar_open = !self.sidebar_open;
			}
		});
		self.sidebar_ui(ui);
		egui::CentralPanel::default().show_inside(ui, |ui| {
			ui.vertical(|ui| {
				// Chat messages area - takes remaining space
				let available_height = ui.available_height();
				let input_upward_spacing = 0.0;
				let input_height = 26.0;
				let input_margin = 4.0;
				let extra_scroll_padding = 80.0;
				let input_panel_height = input_upward_spacing + input_height + input_margin + extra_scroll_padding;
				let top_padding = 22.0;
				let messages_area_height = (available_height - input_panel_height - top_padding - 20.0).max(0.0);

				Frame::NONE
					.inner_margin(egui::Margin {
						left: 0,
						right: 0,
						top: 22,
						bottom: 0,
					})
					.show(ui, |ui| {
						ScrollArea::vertical()
							.animated(false)
							.auto_shrink([false, false])
							.stick_to_bottom(true)
							.max_height(messages_area_height)
							.show(ui, |ui| {
							let row_width = ui.available_width();
							let left_margin = 10.0;
							let right_margin = 10.0;
							for (idx, item) in self.messages.iter().enumerate() {
								let timestamp = self
									.message_timestamps
									.get(idx)
									.map(|s| s.as_str())
									.unwrap_or("--:--:--");
								ui.allocate_ui_with_layout(
									egui::vec2(row_width, 0.0),
									Layout::left_to_right(Align::Min),
									|ui| {
										ui.spacing_mut().item_spacing.x = 8.0;
										let separator_color = ui.visuals().widgets.noninteractive.bg_stroke.color;
										Frame::default()
											.fill(egui::Color32::TRANSPARENT)
											.stroke(egui::Stroke::new(1.0, separator_color))
											.corner_radius(4.0)
											.inner_margin(egui::Margin::same(6))
											.outer_margin(egui::Margin {
												left: 12,
												right: 0,
												top: 0,
												bottom: 4,
											})
											.show(ui, |ui| {
												ui.label(
													egui::RichText::new(timestamp)
														.size(12.0)
														.color(ui.visuals().strong_text_color()),
												);
											});
										ui.separator();
										let messages_middle_width = ui.available_width().max(0.0);
										ui.allocate_ui_with_layout(
											egui::vec2(messages_middle_width, 0.0),
											Layout::top_down(Align::Min),
											|ui| {
												let max_msg_width =
													(messages_middle_width - left_margin - right_margin).max(0.0);
												ui.set_min_width(messages_middle_width);
												ui.set_max_width(messages_middle_width);
												let layout = Layout::top_down(Align::Min);
												ui.with_layout(layout, |ui| {
													ui.set_max_width(max_msg_width);
													let msg_color = ui.visuals().widgets.noninteractive.bg_fill;
													let border_color = match item.from.as_deref() {
														Some("Human") => egui::Color32::from_rgb(0, 255, 0),
														Some(from) if from.starts_with("Ollama") => {
															egui::Color32::from_rgb(255, 255, 0)
														}
														Some("Agent Evaluator") => egui::Color32::from_rgb(255, 105, 180),
														Some("Agent Manager") => egui::Color32::from_rgb(255, 0, 0),
														Some("Agent Researcher") => {
															egui::Color32::from_rgb(128, 0, 255)
														}
														Some(from) if from.starts_with("Agent") => {
															egui::Color32::from_rgb(255, 255, 0)
														}
														Some("System") | Some("API") => {
															egui::Color32::from_rgb(204, 85, 0)
														}
														_ => egui::Color32::TRANSPARENT,
													};
													let border_width = if border_color != egui::Color32::TRANSPARENT {
														1.0
													} else {
														0.0
													};
													let rounding = 4.0;
													let margin = 8.0;
													let outer_margin = egui::Margin {
														left: left_margin as i8,
														right: right_margin as i8,
														top: 0,
														bottom: 4,
													};
													let content_max_width = max_msg_width - margin * 2.0;
													if item.from.as_deref() == Some("Agent Manager") {
														ui.add_space(4.0);
													}
													Frame::default()
														// Sidebar and chat main area
														use crate::ui::overview_chat::store::Store;
														let mut store = Store::open("metrics/overview_chat.sqlite").expect("Failed to open chat store");
														egui::TopBottomPanel::top("sidebar_toggle").show_inside(ui, |ui| {
															if ui.button(if self.sidebar_open { "Hide Rooms" } else { "Show Rooms" }).clicked() {
																self.sidebar_open = !self.sidebar_open;
															}
														});
														self.sidebar_ui(ui);
														egui::CentralPanel::default().show_inside(ui, |ui| {
															ui.vertical(|ui| {
																// Chat messages area - takes remaining space
																let available_height = ui.available_height();
																let input_upward_spacing = 0.0;
																let input_height = 26.0;
																let input_margin = 4.0;
																let extra_scroll_padding = 80.0;
																let input_panel_height = input_upward_spacing + input_height + input_margin + extra_scroll_padding;
																let top_padding = 22.0;
																let messages_area_height = (available_height - input_panel_height - top_padding - 20.0).max(0.0);

																Frame::NONE
																	.inner_margin(egui::Margin {
																		left: 0,
																		right: 0,
																		top: 22,
																		bottom: 0,
																	})
																	.show(ui, |ui| {
																		ScrollArea::vertical()
																			.animated(false)
																			.auto_shrink([false, false])
																			.stick_to_bottom(true)
																			.max_height(messages_area_height)
																			.show(ui, |ui| {
																				ui.vertical(|ui| {
																					ui.set_width(ui.available_width());
																					let len = self.messages.len();
																					for (i, item) in self.messages.iter().enumerate() {
																						ui.allocate_ui_with_layout(
																							egui::vec2(ui.available_width(), 0.0),
																							Layout::left_to_right(Align::Min),
																							|ui| {
																								ui.spacing_mut().item_spacing.x = 8.0;
																								let separator_color = ui.visuals().widgets.noninteractive.bg_stroke.color;
																								Frame::default()
																									.fill(egui::Color32::TRANSPARENT)
																									.stroke(egui::Stroke::new(1.0, separator_color))
																									.corner_radius(4.0)
																									.inner_margin(egui::Margin::same(6))
																									.outer_margin(egui::Margin {
																										left: 12,
																										right: 0,
																										top: 0,
																										bottom: 4,
																									})
																									.show(ui, |ui| {
																										ui.label(
																											egui::RichText::new(Self::display_time_for_message(item))
																												.size(12.0)
																												.color(ui.visuals().strong_text_color()),
																										);
																									});
																								ui.separator();
																								let messages_middle_width = ui.available_width().max(0.0);
																								ui.allocate_ui_with_layout(
																									egui::vec2(messages_middle_width, 0.0),
																									Layout::top_down(Align::Min),
																									|ui| {
																										let max_msg_width =
																											(messages_middle_width - 10.0 - 10.0).max(0.0);
																										ui.set_min_width(messages_middle_width);
																										ui.set_max_width(messages_middle_width);
																										let layout = Layout::top_down(Align::Min);
																										ui.with_layout(layout, |ui| {
																											ui.set_max_width(max_msg_width);
																											let msg_color = ui.visuals().widgets.noninteractive.bg_fill;
																											let border_color = match item.from.as_deref() {
																												Some("Human") => egui::Color32::from_rgb(0, 255, 0),
																												Some(from) if from.starts_with("Ollama") => {
																													egui::Color32::from_rgb(255, 255, 0)
																												}
																												Some("Agent Evaluator") => egui::Color32::from_rgb(255, 105, 180),
																												Some("Agent Manager") => egui::Color32::from_rgb(255, 0, 0),
																												Some("Agent Researcher") => {
																													egui::Color32::from_rgb(128, 0, 255)
																												}
																												Some(from) if from.starts_with("Agent") => {
																													egui::Color32::from_rgb(255, 255, 0)
																												}
																												Some("System") | Some("API") => {
																													egui::Color32::from_rgb(204, 85, 0)
																												}
																												_ => egui::Color32::TRANSPARENT,
																											};
																											let border_width = if border_color != egui::Color32::TRANSPARENT {
																												1.0
																											} else {
																												0.0
																											};
																											let rounding = 4.0;
																											let margin = 8.0;
																											let outer_margin = egui::Margin {
																												left: 10,
																												right: 10,
																												top: 0,
																												bottom: 4,
																											};
																											let content_max_width = max_msg_width - margin * 2.0;
																											if item.from.as_deref() == Some("Agent Manager") {
																												ui.add_space(4.0);
																											}
																											Frame::default()
																												.inner_margin(egui::Margin::same(margin as i8))
																												.outer_margin(outer_margin)
																												.fill(msg_color)
																												.corner_radius(rounding)
																												.stroke(egui::Stroke::new(border_width, border_color))
																												.show(ui, |ui| {
																													ui.set_max_width(content_max_width);
																													ui.with_layout(Layout::top_down(Align::Min), |ui| {
																														let header_color = ui.visuals().strong_text_color();
																														let content_color = ui.visuals().weak_text_color();
																														if let Some(from) = &item.from {
																															if from.starts_with("Ollama ") {
																																let parts: Vec<&str> = from.splitn(2, ' ').collect();
																																if parts.len() == 2 {
																																	ui.horizontal(|ui| {
																																		ui.label(egui::RichText::new("Ollama").strong().color(header_color));
																																		ui.label(egui::RichText::new(parts[1]).color(ui.visuals().weak_text_color()));
																																	});
																																} else {
																																	ui.label(egui::RichText::new(from).strong().color(header_color));
																																}
																															} else {
																																ui.label(egui::RichText::new(from).strong().color(header_color));
																															}
																														}
																														ui.label(egui::RichText::new(&item.content).color(content_color));
																													});
																												});
																										});
																									},
																								);
																							},
																						);
																					}
																					let is_waiting = *self.waiting_for_response.lock().unwrap();
																					if is_waiting {
																						ui.add_space(4.0);
																						ui.horizontal(|ui| {
																							ui.add_space(70.0 + 10.0);
																							ui.spinner();
																						});
																					}
																				});
																		});
																});
																ui.add_space(14.0);
																ui.add_space(-input_upward_spacing);
																ui.with_layout(Layout::top_down(Align::Center), |ui| {
																	ui.set_max_width(ui.available_width() * 0.8);
																	let rounding = 8.0;
																	ui.add_enabled_ui(self.main_input_enabled, |ui| {
																		ui.horizontal(|ui| {
																			let control_height = 26.0;
																			let available_for_input = ui.available_width() - 80.0;
																			let input_frame = Frame::NONE
																				.fill(ui.visuals().widgets.inactive.bg_fill)
																				.corner_radius(rounding)
																				.inner_margin(egui::Margin::symmetric(10, 3));
																			let response = input_frame
																				.show(ui, |ui| {
																					ui.set_height(control_height);
																					ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
																						ui.add_sized(
																							Vec2::new(available_for_input, control_height),
																							egui::TextEdit::singleline(&mut self.input_text)
																								.hint_text("Type a message")
																								.frame(false),
																						)
																					})
																					.inner
																				})
																				.inner;
																			ui.add_space(2.0);
																			let button_frame = Frame::NONE
																				.fill(ui.visuals().widgets.active.bg_fill)
																				.corner_radius(rounding)
																				.inner_margin(egui::Margin::symmetric(12, 3));
																			let send_button_response = button_frame
																				.show(ui, |ui| {
																					ui.set_height(control_height);
																					ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
																						ui.add_sized(
																							Vec2::new(40.0, control_height),
																							egui::Button::new("Send").frame(false),
																						)
																					})
																					.inner
																				})
																				.inner;
																			let send_button_clicked = send_button_response.clicked();
																			let send_clicked = send_button_clicked
																				|| (response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
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
																				// Persist to SQLite for the selected room
																				if let Some(room_id) = &self.selected_room {
																					let _ = store.append_message(room_id, &persisted, &ts);
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
																					tx.send(bot_message).ok();
																				}
																			}
																		});
																	});
																	ui.add_space(4.0);
																	ui.add_enabled_ui(self.main_input_enabled, |ui| {
																		ui.with_layout(Layout::left_to_right(Align::Min), |ui| {
																			let plus_button_height = 26.0;
																			let plus_button_frame = Frame::NONE
																				.fill(ui.visuals().widgets.inactive.bg_fill)
																				.corner_radius(rounding)
																				.inner_margin(egui::Margin::symmetric(12, 3));
																			let plus_button_response = plus_button_frame
																				.show(ui, |ui| {
																					ui.set_height(plus_button_height);
																					ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
																						ui.add_sized(
																							Vec2::new(10.0, plus_button_height),
																							egui::Button::new("+").frame(false),
																						)
																					})
																					.inner
																				})
																				.inner;
																			if plus_button_response.clicked() {
																				if let Some(path) = rfd::FileDialog::new().pick_file() {
																					self.picked_file_path = Some(path.display().to_string());
																					println!("Selected file: {}", self.picked_file_path.as_ref().unwrap());
																				}
																			}
																			if let Some(ref file_path) = self.picked_file_path {
																				ui.add_space(4.0);
																				ui.label(egui::RichText::new(format!("File: {}", file_path)).small().weak());
																			}
																		});
																	});
																});
															});
														});
