use eframe::egui;
use tokio::runtime::Handle;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct Agent {
    id: usize,
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

pub struct MyApp {
    rt_handle: Handle,
    agents: Vec<Agent>,
    next_agent_id: usize,
    http_endpoint: String,
    conversation_loop_handles: Vec<(usize, Arc<Mutex<bool>>, tokio::task::JoinHandle<()>)>, // (agent_id, active_flag, handle)
}

impl MyApp {
    pub fn new(rt_handle: Handle) -> Self {
        Self { 
            rt_handle,
            agents: Vec::new(),
            next_agent_id: 1,
            http_endpoint: std::env::var("CONVERSATION_HTTP_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:3000/".to_string()),
            conversation_loop_handles: Vec::new(),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui| {
                // HTTP Endpoint configuration
                ui.horizontal(|ui| {
                    ui.label("HTTP Endpoint:");
                    ui.add(egui::TextEdit::singleline(&mut self.http_endpoint)
                        .desired_width(300.0));
                });
                ui.add_space(5.0);
                
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

                    ui.separator();

                    if ui.button("Create Agent").clicked() {
                        // Find the lowest available ID
                        let used_ids: std::collections::HashSet<usize> = 
                            self.agents.iter().map(|a| a.id).collect();
                        let mut new_id = 1;
                        while used_ids.contains(&new_id) {
                            new_id += 1;
                        }
                        
                        self.agents.push(Agent {
                            id: new_id,
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
                        
                        // Update next_agent_id to be at least one more than the highest used ID
                        if new_id >= self.next_agent_id {
                            self.next_agent_id = new_id + 1;
                        }
                    }
                });
                
                ui.separator();
                
                // Scrollable area for agents with green border - full width
                let available_width = ui.available_width() - 12.0;
                egui::Frame::default()
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0, 255, 0)))
                    .inner_margin(egui::Margin { left: 6.0, right: 6.0, top: 6.0, bottom: 6.0 })
                    .show(ui, |ui| {
                        ui.set_width(available_width);
                        egui::ScrollArea::vertical().show(ui, |ui| {
                        
                // Collect IDs of agents to remove
                let mut agents_to_remove = Vec::new();
                
                // Collect agent info for partner dropdown (before mutable borrow)
                let agent_names: Vec<(usize, String)> = self.agents.iter()
                    .map(|a| (a.id, a.name.clone()))
                    .collect();
                
                // Collect full agent info for conversation setup (before mutable borrow)
                let agents_info: Vec<(usize, String, String)> = self.agents.iter()
                    .map(|a| (a.id, a.name.clone(), a.instruction.clone()))
                    .collect();
                
                // Agent Manager row - always at the top
                let manager_bg_color = egui::Color32::from_rgb(40, 40, 40);
                let manager_frame = egui::Frame::default()
                    .fill(manager_bg_color)
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(100, 150, 255))) // Blue border for manager
                    .rounding(4.0)
                    .inner_margin(egui::Margin::same(5.0))
                    .outer_margin(egui::Margin::same(0.0));
                
                manager_frame.show(ui, |ui| {
                    ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing = egui::Vec2::new(5.0, 2.0);
                        
                        // Agent Manager label
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Agent Manager").strong().size(14.0));
                        });
                        
                        // Sub-rows for each agent with buttons
                        for (agent_id, agent_name) in &agent_names {
                            ui.horizontal(|ui| {
                                // Status and Erase Agent buttons on the left
                                if ui.button("Status").clicked() {
                                    if let Some(agent) = self.agents.iter().find(|a| a.id == *agent_id) {
                                        println!("=== Agent {} Status ===", agent.id);
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
                                
                                if ui.button("Erase Agent").clicked() {
                                    agents_to_remove.push(*agent_id);
                                }
                                
                                // Agent name label
                                ui.label(agent_name);
                            });
                        }
                    });
                });
                
                // Add spacing between manager and agent rows
                ui.add_space(6.0);
                
                // Display agents in rows
                for agent in &mut self.agents {
                    let agent_id = agent.id;
                    
                    // Agent row - split into left (70%) and right (30%) sections
                    let bg_color = if agent.selected {
                        egui::Color32::from_rgb(50, 50, 50)
                    } else {
                        egui::Color32::from_rgb(45, 45, 45)
                    };
                    
                    let frame = egui::Frame::default()
                        .fill(bg_color)
                        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(255, 192, 203))) // Pink border
                        .rounding(4.0) // 4px border radius
                        .inner_margin(egui::Margin::same(5.0))
                        .outer_margin(egui::Margin::same(0.0));
                    
                    let _frame_response = frame.show(ui, |ui| {
                        ui.horizontal(|ui| {
                            // Left section (50% width) - existing widgets
                            ui.vertical(|ui| {
                                ui.set_width(ui.available_width() * 0.5);
                                ui.spacing_mut().item_spacing = egui::Vec2::new(5.0, 2.0);
                                
                                // Agent Name row
                                ui.horizontal(|ui| {
                                    ui.label("Agent Name:");
                                    ui.add(egui::TextEdit::singleline(&mut agent.name)
                                        .desired_width(100.0));
                                });
                                
                                // Instruction row (system prompt)
                                ui.horizontal(|ui| {
                                    ui.label("Instruction:");
                                    ui.add(egui::TextEdit::singleline(&mut agent.instruction)
                                        .desired_width(200.0));
                                });
                                
                                // Input row with Send button
                                ui.horizontal(|ui| {
                                    ui.label("Input:");
                                    ui.add(egui::TextEdit::singleline(&mut agent.input)
                                        .desired_width(200.0));
                                    
                                    if ui.button("Send").clicked() {
                                        let agent_clone = agent.clone();
                                        let endpoint = self.http_endpoint.clone();
                                        let ctx = ctx.clone();
                                        let handle = self.rt_handle.clone();
                                        handle.spawn(async move {
                                            match crate::adk_integration::send_to_ollama(
                                                &agent_clone.instruction,
                                                &agent_clone.input,
                                                agent_clone.limit_token,
                                                &agent_clone.num_predict,
                                            ).await {
                                                Ok(response) => {
                                                    // If agent is in conversation, send via HTTP
                                                    if agent_clone.in_conversation && agent_clone.conversation_partner_id.is_some() {
                                                        if let Some(partner_id) = agent_clone.conversation_partner_id {
                                                            // Find partner name
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
                                
                                // Limit token checkbox and num_predict row
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
                                
                            });
                            
                            // Vertical separator
                            ui.separator();
                            
                            // Right section (50% width) - conversation controls
                            ui.vertical(|ui| {
                                ui.set_width(ui.available_width() * 0.5);
                                ui.spacing_mut().item_spacing = egui::Vec2::new(5.0, 5.0);
                                
                                // Start/Stop Conversation button
                                let button_text = if agent.conversation_active {
                                    "Stop Conversation"
                                } else {
                                    "Start Conversation"
                                };
                                // Button styling - red when active, normal when inactive
                                let button = if agent.conversation_active {
                                    egui::Button::new(button_text)
                                        .fill(egui::Color32::from_rgb(200, 50, 50)) // Red when active
                                        .min_size(egui::Vec2::new(ui.available_width(), 30.0))
                                } else {
                                    egui::Button::new(button_text)
                                        .min_size(egui::Vec2::new(ui.available_width(), 30.0))
                                };
                                
                                if ui.add(button).clicked() {
                                    if agent.conversation_active {
                                        // Stop conversation
                                        agent.conversation_active = false;
                                        agent.in_conversation = false;
                                        
                                        // Stop any running loop for this agent
                                        self.conversation_loop_handles.retain(|(aid, flag, _)| {
                                            if *aid == agent_id {
                                                *flag.lock().unwrap() = false;
                                                false
                                            } else {
                                                true
                                            }
                                        });
                                    } else {
                                        // Start conversation - need partner and topic
                                        if let Some(partner_id) = agent.conversation_partner_id {
                                            if !agent.conversation_topic.is_empty() {
                                                // Find partner agent info
                                                if let Some((_, partner_name, partner_instruction)) = agents_info.iter().find(|(id, _, _)| *id == partner_id) {
                                                    agent.conversation_active = true;
                                                    agent.in_conversation = true;
                                                    
                                                    // If loop chat is enabled, start the loop
                                                    if agent.loop_chat {
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
                                                            ).await;
                                                        });
                                                        
                                                        self.conversation_loop_handles.push((agent_id, active_flag, loop_handle));
                                                    }
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
                                
                                // Topic input field
                                ui.horizontal(|ui| {
                                    ui.label("Topic:");
                                    ui.add(egui::TextEdit::singleline(&mut agent.conversation_topic)
                                        .desired_width(200.0));
                                });
                                
                                // Partner selection dropdown
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
                                            for (other_id, other_name) in &agent_names {
                                                if *other_id != agent_id {
                                                    ui.selectable_value(
                                                        &mut agent.conversation_partner_id,
                                                        Some(*other_id),
                                                        other_name,
                                                    );
                                                }
                                            }
                                        });
                                });
                                
                                // Loop Chat checkbox
                                ui.checkbox(&mut agent.loop_chat, "Loop Chat");
                                
                                // Show conversation status
                                if agent.conversation_active {
                                    if let Some(pid) = agent.conversation_partner_id {
                                        ui.label(format!("Chatting with Agent {}", pid));
                                    }
                                }
                            });
                        });
                    });
                    
                    // Add 6.0px spacing between rows
                    ui.add_space(6.0);
                }
                
                // Remove agents that were marked for deletion
                for id in agents_to_remove {
                    self.agents.retain(|a| a.id != id);
                }
                    });
                });
            });
        });
    }
}

