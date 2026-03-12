use eframe::egui;
use tokio::runtime::Handle;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct Agent {
    id: usize,
    manager_id: usize,
    name: String,
    selected: bool,
    instruction: String,
    input: String,
    limit_token: bool,
    num_predict: String,
    // Conversation fields
    in_conversation: bool,
    conversation_topic: String,
    conversation_partner_id: Option<usize>,
    loop_chat: bool,
    conversation_active: bool,
}

#[derive(Clone)]
struct Evaluator {
    id: usize,
    manager_id: usize,
    name: String,
    instruction: String,
    limit_token: bool,
    num_predict: String,
    active: bool,
}

#[derive(Clone)]
struct AgentManager {
    id: usize,
    name: String,
}

pub struct MyApp {
    rt_handle: Handle,
    managers: Vec<AgentManager>,
    next_manager_id: usize,
    agents: Vec<Agent>,
    next_agent_id: usize,
    evaluators: Vec<Evaluator>,
    next_evaluator_id: usize,
    http_endpoint: String,
    last_message_in_chat: Arc<Mutex<Option<String>>>,
    last_evaluated_message_by_evaluator: Arc<Mutex<std::collections::HashMap<usize, String>>>,
    conversation_loop_handles: Vec<(usize, Arc<Mutex<bool>>, tokio::task::JoinHandle<()>)>, // (agent_id, active_flag, handle)
}

impl MyApp {
    pub fn new(rt_handle: Handle) -> Self {
        Self { 
            rt_handle,
            managers: Vec::new(),
            next_manager_id: 1,
            agents: Vec::new(),
            next_agent_id: 1,
            evaluators: Vec::new(),
            next_evaluator_id: 1,
            http_endpoint: std::env::var("CONVERSATION_HTTP_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:3000/".to_string()),
            last_message_in_chat: Arc::new(Mutex::new(None)),
            last_evaluated_message_by_evaluator: Arc::new(Mutex::new(std::collections::HashMap::new())),
            conversation_loop_handles: Vec::new(),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let any_evaluator_active = self.evaluators.iter().any(|e| e.active);
        if any_evaluator_active {
            ctx.request_repaint();
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui| {
                // Top row with HTTP Endpoint and buttons - green border
                let available_width = ui.available_width() - 12.0;
                let settings_bg_color = egui::Color32::from_rgb(40, 40, 40);

                egui::Frame::default()
                    .fill(settings_bg_color)
                    //.stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0, 255, 0)))
                    .inner_margin(egui::Margin { left: 6.0, right: 6.0, top: 6.0, bottom: 6.0 })
                    .rounding(4.0)
                    .show(ui, |ui| {
                        ui.set_width(available_width);
                        ui.vertical(|ui| {

                            ui.label(egui::RichText::new("Settings").strong().size(12.0));
                            ui.separator();
                            
                            // HTTP Endpoint configuration
                            ui.horizontal(|ui| {
                                ui.label("Chat HTTP Endpoint:");
                                ui.add(egui::TextEdit::singleline(&mut self.http_endpoint)
                                    .desired_width(200.0));
                            });
                            ui.add_space(5.0);
                            
                            ui.horizontal(|ui| {
                                //ui.add_space(20.0);
                                if ui.button("Agent Manager").clicked() {
                                    let used_ids: std::collections::HashSet<usize> =
                                        self.managers.iter().map(|m| m.id).collect();
                                    let mut new_id = 1;
                                    while used_ids.contains(&new_id) {
                                        new_id += 1;
                                    }
                                    self.managers.push(AgentManager {
                                        id: new_id,
                                        name: format!("Agent Manager {}", new_id),
                                    });
                                    if new_id >= self.next_manager_id {
                                        self.next_manager_id = new_id + 1;
                                    }
                                }
                                ui.add_space(6.0);

                                if ui.button("Ping Ollama").clicked() {
                                    println!("Pinging Ollama");
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
                    });
                
                ui.separator();
                
                // Scrollable area for agents with green border - full width
                let available_width = ui.available_width() - 12.0;
                egui::Frame::default()
                    .fill(egui::Color32::from_rgb(40, 40, 40))
                    //.stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0, 255, 0)))
                    .inner_margin(egui::Margin { left: 6.0, right: 6.0, top: 6.0, bottom: 6.0 })
                    .rounding(4.0)
                    .show(ui, |ui| {
                        ui.set_width(available_width);
                        egui::ScrollArea::vertical().show(ui, |ui| {
                        
                // Collect IDs of managers/agents/evaluators to remove
                let mut managers_to_remove = Vec::new();
                let mut agents_to_remove = Vec::new();
                let mut evaluators_to_remove = Vec::new();
                
                let evaluator_names: Vec<(usize, usize, String)> = self.evaluators.iter()
                    .map(|e| (e.id, e.manager_id, e.name.clone()))
                    .collect();
                
                // Collect agent info for partner dropdown (before mutable borrow)
                let agent_names: Vec<(usize, usize, String)> = self.agents.iter()
                    .map(|a| (a.id, a.manager_id, a.name.clone()))
                    .collect();
                
                // Collect full agent info for conversation setup (before mutable borrow)
                let agents_info: Vec<(usize, usize, String, String)> = self.agents.iter()
                    .map(|a| (a.id, a.manager_id, a.name.clone(), a.instruction.clone()))
                    .collect();
                
                let panel_border_color = ui.visuals().widgets.noninteractive.bg_stroke.color;
                let manager_bg_color = egui::Color32::from_rgb(40, 40, 40);
                let manager_frame = egui::Frame::default()
                    .fill(manager_bg_color)
                    .stroke(egui::Stroke::new(1.0, panel_border_color))
                    .rounding(4.0)
                    .inner_margin(egui::Margin::same(5.0))
                    .outer_margin(egui::Margin::same(0.0));

                ui.label(egui::RichText::new("Workspace").strong().size(12.0));
                ui.separator();

                if self.managers.is_empty() {
                    ui.label("No Agent Manager. Click \"Agent Manager\" above to create one.");
                }

                for manager in &self.managers {
                    manager_frame.show(ui, |ui| {
                        let manager_width = ui.available_width();
                        ui.set_width(manager_width);
                        ui.set_max_width(manager_width);
                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                ui.label(egui::RichText::new(&manager.name).strong().size(12.0));
                            });
                            ui.separator();
                            ui.vertical(|ui| {
                                ui.spacing_mut().item_spacing = egui::Vec2::new(5.0, 2.0);
                                ui.horizontal(|ui| {
                                    if ui.button("Create Worker").clicked() {
                                        let used_ids: std::collections::HashSet<usize> =
                                            self.agents.iter().map(|a| a.id).collect();
                                        let mut new_id = 1;
                                        while used_ids.contains(&new_id) {
                                            new_id += 1;
                                        }
                                        self.agents.push(Agent {
                                            id: new_id,
                                            manager_id: manager.id,
                                            name: format!("Agent {}", new_id),
                                            selected: false,
                                            instruction: "You are an assistant".to_string(),
                                            input: String::new(),
                                            limit_token: false,
                                            num_predict: String::new(),
                                            in_conversation: false,
                                            conversation_topic: String::new(),
                                            conversation_partner_id: None,
                                            loop_chat: false,
                                            conversation_active: false,
                                        });
                                        if new_id >= self.next_agent_id {
                                            self.next_agent_id = new_id + 1;
                                        }
                                    }
                                    if ui.button("Create Evaluator").clicked() {
                                        let used_ids: std::collections::HashSet<usize> =
                                            self.evaluators.iter().map(|e| e.id).collect();
                                        let mut new_id = 1;
                                        while used_ids.contains(&new_id) {
                                            new_id += 1;
                                        }
                                        self.evaluators.push(Evaluator {
                                            id: new_id,
                                            manager_id: manager.id,
                                            name: format!("Evaluator {}", new_id),
                                            instruction: "Answer with only one word: happy or sad. Evaluate the sentiment of this message.".to_string(),
                                            limit_token: false,
                                            num_predict: String::new(),
                                            active: false,
                                        });
                                        if new_id >= self.next_evaluator_id {
                                            self.next_evaluator_id = new_id + 1;
                                        }
                                    }
                                    if ui.button("Erase Manager").clicked() {
                                        managers_to_remove.push(manager.id);
                                    }
                                });
                                ui.separator();
                                for (agent_id, manager_id, agent_name) in &agent_names {
                                    if *manager_id != manager.id {
                                        continue;
                                    }
                                    ui.horizontal(|ui| {
                                        if ui.button("Status").clicked() {
                                            if let Some(agent) = self.agents.iter().find(|a| a.id == *agent_id) {
                                                println!("=== Agent {} Status ===", agent.id);
                                                println!("Manager: {}", agent.manager_id);
                                                println!("Name: {}", agent.name);
                                                println!("Instruction: {}", agent.instruction);
                                                println!("Limit Token: {}", agent.limit_token);
                                                if agent.limit_token {
                                                    println!("num_predict: {}", agent.num_predict);
                                                }
                                                println!("Selected: {}", agent.selected);
                                                println!("In Conversation: {}", agent.in_conversation);
                                                if agent.in_conversation {
                                                    println!("Topic: {}", agent.conversation_topic);
                                                    if let Some(pid) = agent.conversation_partner_id {
                                                        println!("Partner: Agent {}", pid);
                                                    }
                                                }
                                                println!("======================");
                                            }
                                        }
                                        if ui.button("Erase").clicked() {
                                            agents_to_remove.push(*agent_id);
                                        }
                                        ui.label(agent_name);
                                    });
                                }
                                for (eval_id, manager_id, eval_name) in &evaluator_names {
                                    if *manager_id != manager.id {
                                        continue;
                                    }
                                    ui.horizontal(|ui| {
                                        if ui.button("Status").clicked() {
                                            if let Some(e) = self.evaluators.iter().find(|x| x.id == *eval_id) {
                                                println!("=== Evaluator {} Status ===", e.id);
                                                println!("Manager: {}", e.manager_id);
                                                println!("Name: {}", e.name);
                                                println!("Instruction: {}", e.instruction);
                                                println!("========================");
                                            }
                                        }
                                        if ui.button("Erase").clicked() {
                                            evaluators_to_remove.push(*eval_id);
                                        }
                                        ui.label(eval_name);
                                    });
                                }

                                ui.separator();
                                ui.add_space(6.0);

                                // Display child blocks inside this manager rectangle
                                for agent in &mut self.agents {
                                    if agent.manager_id != manager.id {
                                        continue;
                                    }
                                    let agent_id = agent.id;
                                    
                                    let bg_color = if agent.selected {
                                        egui::Color32::from_rgb(50, 50, 50)
                                    } else {
                                        egui::Color32::from_rgb(45, 45, 45)
                                    };
                                    
                                    let frame = egui::Frame::default()
                                        .fill(bg_color)
                                        .stroke(egui::Stroke::new(1.0, panel_border_color))
                                        .rounding(4.0)
                                        .inner_margin(egui::Margin::same(5.0))
                                        .outer_margin(egui::Margin { left: 0.0, right: 0.0, top: 0.0, bottom: 0.0 });
                                    
                                    ui.horizontal(|ui| {
                                        ui.set_max_width(ui.available_width());
                                        let _frame_response = frame.show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                ui.vertical(|ui| {
                                                    ui.label(egui::RichText::new("Agent Worker").strong().size(12.0));
                                                });
                                                ui.separator();
                                                ui.vertical(|ui| {
                                                    ui.horizontal(|ui| {
                                                        ui.vertical(|ui| {
                                                            ui.set_width(ui.available_width() * 0.5);
                                                            ui.spacing_mut().item_spacing = egui::Vec2::new(5.0, 2.0);

                                                            ui.horizontal(|ui| {
                                                                ui.label("Name:");
                                                                ui.add(egui::TextEdit::singleline(&mut agent.name));
                                                            });
                                                            
                                                            ui.horizontal(|ui| {
                                                                ui.label("Instruction:");
                                                                ui.add(egui::TextEdit::singleline(&mut agent.instruction));
                                                            });

                                                            ui.horizontal(|ui| {
                                                                ui.label("Topic:");
                                                                ui.add(egui::TextEdit::singleline(&mut agent.conversation_topic));
                                                            });

                                                            ui.horizontal(|ui| {
                                                                ui.label("With:");
                                                                let selected_text = if let Some(pid) = agent.conversation_partner_id {
                                                                    format!("Agent {}", pid)
                                                                } else {
                                                                    "None".to_string()
                                                                };
                                                                
                                                                egui::ComboBox::from_id_source(ui.id().with(agent_id).with("partner"))
                                                                    .selected_text(selected_text)
                                                                    .show_ui(ui, |ui| {
                                                                        ui.selectable_value(&mut agent.conversation_partner_id, None, "None");
                                                                        for (other_id, other_manager_id, other_name) in &agent_names {
                                                                            if *other_id != agent_id && *other_manager_id == agent.manager_id {
                                                                                ui.selectable_value(
                                                                                    &mut agent.conversation_partner_id,
                                                                                    Some(*other_id),
                                                                                    other_name,
                                                                                );
                                                                            }
                                                                        }
                                                                    });
                                                            });
                                                            
                                                            ui.checkbox(&mut agent.loop_chat, "Loop Chat");

                                                            let button_text = if agent.conversation_active {
                                                                "Stop Conversation"
                                                            } else {
                                                                "Start Conversation"
                                                            };
                                                            let button = if agent.conversation_active {
                                                                egui::Button::new(button_text)
                                                                    .fill(egui::Color32::from_rgb(200, 50, 50))
                                                            } else {
                                                                egui::Button::new(button_text)
                                                            };

                                                            ui.separator();
                                                            
                                                            if ui.add(button).clicked() {
                                                                if agent.conversation_active {
                                                                    agent.conversation_active = false;
                                                                    agent.in_conversation = false;
                                                                    self.conversation_loop_handles.retain(|(aid, flag, _)| {
                                                                        if *aid == agent_id {
                                                                            *flag.lock().unwrap() = false;
                                                                            false
                                                                        } else {
                                                                            true
                                                                        }
                                                                    });
                                                                } else {
                                                                    if let Some(partner_id) = agent.conversation_partner_id {
                                                                        if !agent.conversation_topic.is_empty() {
                                                                            if let Some((_, _, partner_name, partner_instruction)) = agents_info.iter().find(|(id, _, _, _)| *id == partner_id) {
                                                                                agent.conversation_active = true;
                                                                                agent.in_conversation = true;
                                                                                let active_flag = Arc::new(Mutex::new(true));
                                                                                let flag_clone = active_flag.clone();
                                                                                let endpoint = self.http_endpoint.clone();
                                                                                let handle = self.rt_handle.clone();
                                                                                let agent_a_id = agent.id;
                                                                                let agent_a_name = agent.name.clone();
                                                                                let agent_a_instruction = agent.instruction.clone();
                                                                                let agent_b_id = partner_id;
                                                                                let agent_b_name = partner_name.clone();
                                                                                let agent_b_instruction = partner_instruction.clone();
                                                                                let topic = agent.conversation_topic.clone();
                                                                                let last_msg = self.last_message_in_chat.clone();
                                                                                let loop_handle = handle.spawn(async move {
                                                                                    crate::conversation_loop::start_conversation_loop(
                                                                                        agent_a_id,
                                                                                        agent_a_name,
                                                                                        agent_a_instruction,
                                                                                        agent_b_id,
                                                                                        agent_b_name,
                                                                                        agent_b_instruction,
                                                                                        topic,
                                                                                        endpoint,
                                                                                        flag_clone,
                                                                                        last_msg,
                                                                                    ).await;
                                                                                });
                                                                                self.conversation_loop_handles.push((agent_id, active_flag, loop_handle));
                                                                            } else {
                                                                                println!("Partner agent not found");
                                                                            }
                                                                        } else {
                                                                            println!("Cannot start conversation: need topic");
                                                                        }
                                                                    } else {
                                                                        println!("Cannot start conversation: need partner");
                                                                    }
                                                                }
                                                            }
                                                        });
                                                        
                                                        ui.separator();
                                                        
                                                        ui.vertical(|ui| {
                                                            ui.set_width(ui.available_width());
                                                            ui.spacing_mut().item_spacing = egui::Vec2::new(5.0, 5.0);
                                                            
                                                            ui.horizontal(|ui| {
                                                                ui.label("Input:");
                                                                let send_clicked = ui.button("Send").clicked();
                                                                ui.add_sized(
                                                                    [ui.available_width(), 0.0],
                                                                    egui::TextEdit::singleline(&mut agent.input),
                                                                );
                                                                if send_clicked {
                                                                    let agent_clone = agent.clone();
                                                                    let endpoint = self.http_endpoint.clone();
                                                                    let ctx = ctx.clone();
                                                                    let handle = self.rt_handle.clone();
                                                                    let last_msg = self.last_message_in_chat.clone();
                                                                    handle.spawn(async move {
                                                                        match crate::adk_integration::send_to_ollama(
                                                                            &agent_clone.instruction,
                                                                            &agent_clone.input,
                                                                            agent_clone.limit_token,
                                                                            &agent_clone.num_predict,
                                                                        ).await {
                                                                            Ok(response) => {
                                                                                *last_msg.lock().unwrap() = Some(response.clone());
                                                                                if agent_clone.in_conversation && agent_clone.conversation_partner_id.is_some() {
                                                                                    if let Some(partner_id) = agent_clone.conversation_partner_id {
                                                                                        let partner_name = format!("Agent {}", partner_id);
                                                                                        if let Err(e) = crate::http_client::send_conversation_message(
                                                                                            &endpoint,
                                                                                            agent_clone.id,
                                                                                            &agent_clone.name,
                                                                                            partner_id,
                                                                                            &partner_name,
                                                                                            &agent_clone.conversation_topic,
                                                                                            &response,
                                                                                        ).await {
                                                                                            eprintln!("Failed to send HTTP message: {}", e);
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                            Err(e) => {
                                                                                eprintln!("Ollama error: {}", e);
                                                                            }
                                                                        }
                                                                        ctx.request_repaint();
                                                                    });
                                                                }
                                                            });
                                                            
                                                            ui.horizontal(|ui| {
                                                                if ui.checkbox(&mut agent.limit_token, "Limit Token").changed() {
                                                                    if !agent.limit_token {
                                                                        agent.num_predict.clear();
                                                                    }
                                                                }
                                                                
                                                                if agent.limit_token {
                                                                    ui.label("num_predict:");
                                                                    ui.add(egui::TextEdit::singleline(&mut agent.num_predict)
                                                                        .desired_width(80.0));
                                                                }
                                                            });
                                                            
                                                            if agent.conversation_active {
                                                                if let Some(pid) = agent.conversation_partner_id {
                                                                    ui.label(format!("Chatting with Agent {}", pid));
                                                                }
                                                            }
                                                        });
                                                    });
                                                });
                                            });
                                        });
                                    });
                                    ui.add_space(6.0);
                                }

                                for evaluator in &mut self.evaluators {
                                    if evaluator.manager_id != manager.id {
                                        continue;
                                    }
                                    let eval_id = evaluator.id;
                                    let last_msg = self.last_message_in_chat.lock().unwrap().clone();
                                    let last_eval = self.last_evaluated_message_by_evaluator.lock().unwrap().get(&eval_id).cloned();
                                    let should_run = evaluator.active
                                        && last_msg.as_ref().map_or(false, |s| !s.is_empty())
                                        && last_eval.as_ref() != last_msg.as_ref();
                                    if should_run {
                                        let message = last_msg.clone().unwrap_or_default();
                                        println!("[Evaluator] Analyzing last message ({} chars), sending to Ollama...", message.len());
                                        self.last_evaluated_message_by_evaluator.lock().unwrap().insert(eval_id, message.clone());
                                        let eval_clone = evaluator.clone();
                                        let endpoint = self.http_endpoint.clone();
                                        let ctx = ctx.clone();
                                        let handle = self.rt_handle.clone();
                                        handle.spawn(async move {
                                            match crate::adk_integration::send_to_ollama(
                                                &eval_clone.instruction,
                                                &message,
                                                eval_clone.limit_token,
                                                &eval_clone.num_predict,
                                            ).await {
                                                Ok(response) => {
                                                    let response_lower = response.to_lowercase();
                                                    let sentiment = if response_lower.contains("happy") {
                                                        "happy"
                                                    } else if response_lower.contains("sad") {
                                                        "sad"
                                                    } else {
                                                        "unknown"
                                                    };
                                                    if let Err(e) = crate::http_client::send_evaluator_result(
                                                        &endpoint,
                                                        "Agent Evaluator",
                                                        sentiment,
                                                        &response,
                                                    ).await {
                                                        eprintln!("[Evaluator] Failed to send to web-chat: {}", e);
                                                    } else {
                                                        println!("[Evaluator] Sent to web-chat: {} -> {}", sentiment, &response[..response.len().min(60)]);
                                                    }
                                                }
                                                Err(e) => eprintln!("Ollama error: {}", e),
                                            }
                                            ctx.request_repaint();
                                        });
                                    }

                                    let bg_color = egui::Color32::from_rgb(45, 45, 45);
                                    let frame = egui::Frame::default()
                                        .fill(bg_color)
                                        .stroke(egui::Stroke::new(1.0, panel_border_color))
                                        .rounding(4.0)
                                        .inner_margin(egui::Margin::same(5.0))
                                        .outer_margin(egui::Margin { left: 0.0, right: 0.0, top: 0.0, bottom: 0.0 });
                                    ui.horizontal(|ui| {
                                        ui.set_max_width(ui.available_width());
                                        frame.show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                ui.vertical(|ui| {
                                                    ui.label(egui::RichText::new("Agent Evaluator").strong().size(12.0));
                                                });
                                                ui.separator();
                                                ui.vertical(|ui| {
                                                    ui.spacing_mut().item_spacing = egui::Vec2::new(5.0, 2.0);
                                                    ui.horizontal(|ui| {
                                                        ui.label("Name:");
                                                        ui.add(egui::TextEdit::singleline(&mut evaluator.name));
                                                    });
                                                    ui.horizontal(|ui| {
                                                        ui.label("Instruction:");
                                                        ui.add(egui::TextEdit::singleline(&mut evaluator.instruction));
                                                    });
                                                    ui.horizontal(|ui| {
                                                        if ui.checkbox(&mut evaluator.limit_token, "Limit Token").changed() {
                                                            if !evaluator.limit_token {
                                                                evaluator.num_predict.clear();
                                                            }
                                                        }
                                                        if evaluator.limit_token {
                                                            ui.label("num_predict:");
                                                            ui.add(egui::TextEdit::singleline(&mut evaluator.num_predict).desired_width(80.0));
                                                        }
                                                    });
                                                    ui.separator();
                                                    let button_text = if evaluator.active { "Stop Evaluating" } else { "Evaluate" };
                                                    let button = if evaluator.active {
                                                        egui::Button::new(button_text)
                                                            .fill(egui::Color32::from_rgb(200, 50, 50))
                                                            .min_size(egui::Vec2::new(140.0, 20.0))
                                                    } else {
                                                        egui::Button::new(button_text).min_size(egui::Vec2::new(140.0, 20.0))
                                                    };
                                                    if ui.add(button).clicked() {
                                                        evaluator.active = !evaluator.active;
                                                        if evaluator.active {
                                                            println!("[Evaluator] ON");
                                                        } else {
                                                            println!("[Evaluator] OFF");
                                                        }
                                                        if evaluator.active {
                                                            if let Some(message) = last_msg.clone() {
                                                                if last_eval.as_ref() != Some(&message) {
                                                                    println!("[Evaluator] Manual trigger for last message ({} chars)", message.len());
                                                                    self.last_evaluated_message_by_evaluator.lock().unwrap().insert(eval_id, message.clone());
                                                                    let eval_clone = evaluator.clone();
                                                                    let endpoint = self.http_endpoint.clone();
                                                                    let ctx = ctx.clone();
                                                                    let handle = self.rt_handle.clone();
                                                                    handle.spawn(async move {
                                                                        match crate::adk_integration::send_to_ollama(
                                                                            &eval_clone.instruction,
                                                                            &message,
                                                                            eval_clone.limit_token,
                                                                            &eval_clone.num_predict,
                                                                        ).await {
                                                                            Ok(response) => {
                                                                                let response_lower = response.to_lowercase();
                                                                                let sentiment = if response_lower.contains("happy") {
                                                                                    "happy"
                                                                                } else if response_lower.contains("sad") {
                                                                                    "sad"
                                                                                } else {
                                                                                    "unknown"
                                                                                };
                                                                                if let Err(e) = crate::http_client::send_evaluator_result(
                                                                                    &endpoint,
                                                                                    "Agent Evaluator",
                                                                                    sentiment,
                                                                                    &response,
                                                                                ).await {
                                                                                    eprintln!("[Evaluator] Failed to send to web-chat: {}", e);
                                                                                } else {
                                                                                    println!("[Evaluator] Sent to web-chat: {} -> {}", sentiment, &response[..response.len().min(60)]);
                                                                                }
                                                                            }
                                                                            Err(e) => eprintln!("[Evaluator] Ollama error: {}", e),
                                                                        }
                                                                        ctx.request_repaint();
                                                                    });
                                                                }
                                                            }
                                                        }
                                                    }
                                                });
                                            });
                                        });
                                    });
                                    ui.add_space(6.0);
                                }
                            });
                        });
                    });
                    ui.add_space(6.0);
                }
                
                // Remove managers and all their owned workers/evaluators
                for manager_id in managers_to_remove {
                    let manager_agent_ids: Vec<usize> = self.agents.iter()
                        .filter(|a| a.manager_id == manager_id)
                        .map(|a| a.id)
                        .collect();
                    self.managers.retain(|m| m.id != manager_id);
                    self.agents.retain(|a| a.manager_id != manager_id);
                    self.evaluators.retain(|e| e.manager_id != manager_id);
                    self.conversation_loop_handles.retain(|(aid, flag, _)| {
                        if manager_agent_ids.contains(aid) {
                            *flag.lock().unwrap() = false;
                            false
                        } else {
                            true
                        }
                    });
                }

                // Remove agents and evaluators that were marked for deletion
                for id in agents_to_remove {
                    self.agents.retain(|a| a.id != id);
                }
                for id in evaluators_to_remove {
                    self.evaluators.retain(|e| e.id != id);
                }
                    });
                });
            });
        });
    }
}

