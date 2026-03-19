use super::AMSAgents;
use eframe::egui;
use rand::Rng;
use egui_snarl::ui::{BackgroundPattern, PinInfo, PinPlacement, SnarlStyle, SnarlViewer, SnarlWidget, WireStyle};
use egui_snarl::{InPin, NodeId, OutPin, Snarl};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, PartialEq, Eq)]
enum AgentNodeKind {
    Manager,
    Worker,
    Conversation,
    Evaluator,
    Researcher,
    Topic,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NodeCategory {
    Agent,
    Tool,
    Output,
}

impl AgentNodeKind {
    fn label(&self) -> &'static str {
        match self {
            AgentNodeKind::Manager => "Agent Manager",
            AgentNodeKind::Worker => "Agent Worker",
            AgentNodeKind::Conversation => "Conversation",
            AgentNodeKind::Evaluator => "Agent Evaluator",
            AgentNodeKind::Researcher => "Agent Researcher",
            AgentNodeKind::Topic => "Topic",
        }
    }

    fn category(&self) -> NodeCategory {
        match self {
            AgentNodeKind::Manager
            | AgentNodeKind::Worker
            | AgentNodeKind::Evaluator
            | AgentNodeKind::Researcher => NodeCategory::Agent,
            AgentNodeKind::Topic => NodeCategory::Tool,
            AgentNodeKind::Conversation => NodeCategory::Output,
        }
    }

    fn inputs(&self) -> usize {
        match self {
            AgentNodeKind::Manager => 0,
            AgentNodeKind::Worker => 2,
            AgentNodeKind::Conversation => 2,
            AgentNodeKind::Evaluator => 1,   // worker only
            AgentNodeKind::Researcher => 1,  // worker only
            AgentNodeKind::Topic => 0,
        }
    }

    fn outputs(&self) -> usize {
        match self {
            AgentNodeKind::Manager => 1,
            AgentNodeKind::Worker => 1,
            AgentNodeKind::Conversation => 0, // output node: terminal
            AgentNodeKind::Evaluator => 0,
            AgentNodeKind::Researcher => 0,
            AgentNodeKind::Topic => 1,
        }
    }
}

#[derive(Clone)]
struct NodeManagerData {
    name: String,
    global_id: String,
}

#[derive(Clone)]
struct NodeWorkerData {
    name: String,
    global_id: String,

    instruction_mode: String,
    instruction: String,

    analysis_mode: String,
    conversation_topic: String,
    conversation_topic_source: String,

    // Inferred from graph wires (Manager -> Worker)
    manager_node: Option<NodeId>,

    // Inferred from graph wires (Topic -> Worker)
    topic_node: Option<NodeId>,
}

#[derive(Clone)]
struct NodeEvaluatorData {
    name: String,
    global_id: String,

    analysis_mode: String,
    instruction: String,

    limit_token: bool,
    num_predict: String,

    active: bool,

    // Inferred from graph wires (Worker -> Evaluator -> Manager)
    worker_node: Option<NodeId>,
    manager_node: Option<NodeId>,

}

#[derive(Clone)]
struct NodeResearcherData {
    name: String,
    global_id: String,

    topic_mode: String,
    instruction: String,

    limit_token: bool,
    num_predict: String,

    active: bool,

    // Inferred from graph wires (Worker -> Researcher -> Manager)
    worker_node: Option<NodeId>,
    manager_node: Option<NodeId>,

}

#[derive(Clone)]
struct NodeTopicData {
    name: String,
    global_id: String,

    analysis_mode: String,
    topic: String,
}

#[derive(Clone)]
struct NodeConversationData {
    name: String,
    global_id: String,

    /// "Dialogue" = two workers (back-and-forth), "Monologue" = one worker (single-agent).
    conversation_mode: String,

    /// Inferred: Worker out → this node's input 0 / 1.
    worker_a_node: Option<NodeId>,
    worker_b_node: Option<NodeId>,

    conversation_active: bool,
}

#[derive(Clone)]
enum NodePayload {
    Manager(NodeManagerData),
    Worker(NodeWorkerData),
    Conversation(NodeConversationData),
    Evaluator(NodeEvaluatorData),
    Researcher(NodeResearcherData),
    Topic(NodeTopicData),
}

#[derive(Clone)]
pub(super) struct NodeData {
    kind: AgentNodeKind,
    pub label: String,
    payload: NodePayload,
}

impl NodeData {
    fn new_global_id() -> String {
        const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        const LEN: usize = 10;
        let mut rng = rand::rng();

        // Collisions are extremely unlikely; we don't coordinate across the whole app here.
        (0..LEN)
            .map(|_| {
                let idx = rng.random_range(0..CHARSET.len());
                CHARSET[idx] as char
            })
            .collect()
    }

    fn new_manager() -> Self {
        let global_id = Self::new_global_id();
        Self {
            kind: AgentNodeKind::Manager,
            label: AgentNodeKind::Manager.label().to_string(),
            payload: NodePayload::Manager(NodeManagerData {
                name: AgentNodeKind::Manager.label().to_string(),
                global_id,
            }),
        }
    }

    fn new_worker() -> Self {
        let global_id = Self::new_global_id();
        Self {
            kind: AgentNodeKind::Worker,
            label: AgentNodeKind::Worker.label().to_string(),
            payload: NodePayload::Worker(NodeWorkerData {
                name: AgentNodeKind::Worker.label().to_string(),
                global_id,
                instruction_mode: "Assistant".to_string(),
                instruction: "You are a helpful assistant. Answer clearly, stay concise, and focus on the user request.".to_string(),
                analysis_mode: "European Politics".to_string(),
                conversation_topic: "Discuss European Politics and provide a concise overview of the main issue in one or two sentences.".to_string(),
                conversation_topic_source: "Own".to_string(),
                manager_node: None,
                topic_node: None,
            }),
        }
    }

    fn new_conversation() -> Self {
        let global_id = Self::new_global_id();
        Self {
            kind: AgentNodeKind::Conversation,
            label: AgentNodeKind::Conversation.label().to_string(),
            payload: NodePayload::Conversation(NodeConversationData {
                name: AgentNodeKind::Conversation.label().to_string(),
                global_id,
                conversation_mode: "Monologue".to_string(),
                worker_a_node: None,
                worker_b_node: None,
                conversation_active: false,
            }),
        }
    }

    fn new_evaluator() -> Self {
        let global_id = Self::new_global_id();
        Self {
            kind: AgentNodeKind::Evaluator,
            label: AgentNodeKind::Evaluator.label().to_string(),
            payload: NodePayload::Evaluator(NodeEvaluatorData {
                name: AgentNodeKind::Evaluator.label().to_string(),
                global_id,
                analysis_mode: "Topic Extraction".to_string(),
                instruction: "Topic Extraction: extract the topic in 1 or 2 words. Identify what is the topic of the sentence being analysed.".to_string(),
                limit_token: false,
                num_predict: String::new(),
                active: false,
                worker_node: None,
                manager_node: None,
            }),
        }
    }

    fn new_researcher() -> Self {
        let global_id = Self::new_global_id();
        Self {
            kind: AgentNodeKind::Researcher,
            label: AgentNodeKind::Researcher.label().to_string(),
            payload: NodePayload::Researcher(NodeResearcherData {
                name: AgentNodeKind::Researcher.label().to_string(),
                global_id,
                topic_mode: "Articles".to_string(),
                instruction: "Generate article references connected to the message context. Prefer a mix of classic and recent pieces.".to_string(),
                limit_token: false,
                num_predict: String::new(),
                active: false,
                worker_node: None,
                manager_node: None,
            }),
        }
    }

    fn new_topic() -> Self {
        let global_id = Self::new_global_id();
        Self {
            kind: AgentNodeKind::Topic,
            label: AgentNodeKind::Topic.label().to_string(),
            payload: NodePayload::Topic(NodeTopicData {
                name: AgentNodeKind::Topic.label().to_string(),
                global_id,
                analysis_mode: "European Politics".to_string(),
                topic: "Discuss European Politics and provide a concise overview of the main issue in one or two sentences.".to_string(),
            }),
        }
    }

    fn set_name(&mut self, name: String) {
        match &mut self.payload {
            NodePayload::Manager(m) => m.name = name,
            NodePayload::Worker(w) => w.name = name,
            NodePayload::Conversation(c) => c.name = name,
            NodePayload::Evaluator(e) => e.name = name,
            NodePayload::Researcher(r) => r.name = name,
            NodePayload::Topic(t) => t.name = name,
        }
    }
}

pub(super) struct NodesPanelState {
    snarl: Snarl<NodeData>,
    wire_style: WireStyle,
}

impl Default for NodesPanelState {
    fn default() -> Self {
        let snarl = Snarl::new();
        Self {
            snarl,
            wire_style: WireStyle::Bezier5,
        }
    }
}

#[derive(Default)]
struct BasicNodeViewer;

impl BasicNodeViewer {
    fn numbered_name_for_kind(snarl: &Snarl<NodeData>, kind: AgentNodeKind) -> String {
        let idx = snarl
            .nodes_ids_data()
            .filter(|(_, n)| n.value.kind == kind)
            .count()
            + 1;
        format!("{} {}", kind.label(), idx)
    }
}

impl SnarlViewer<NodeData> for BasicNodeViewer {
    fn title(&mut self, node: &NodeData) -> String {
        node.label.clone()
    }

    fn inputs(&mut self, _node: &NodeData) -> usize {
        _node.kind.inputs()
    }

    fn show_input(
        &mut self,
        _pin: &InPin,
        ui: &mut egui::Ui,
        _snarl: &mut Snarl<NodeData>,
    ) -> impl egui_snarl::ui::SnarlPin + 'static {
        let node_kind = _snarl
            .get_node(_pin.id.node)
            .map(|n| n.kind)
            .unwrap_or(AgentNodeKind::Manager);

        match node_kind {
            AgentNodeKind::Worker => {
                if _pin.id.input == 0 {
                    ui.label("manager");
                } else if _pin.id.input == 1 {
                    ui.label("topic");
                } else {
                    ui.label("in");
                }
            }
            AgentNodeKind::Conversation => {
                if _pin.id.input == 0 {
                    ui.label("A");
                } else if _pin.id.input == 1 {
                    ui.label("B");
                } else {
                    ui.label("in");
                }
            }
            AgentNodeKind::Evaluator => {
                if _pin.id.input == 0 {
                    ui.label("worker");
                } else {
                    ui.label("in");
                }
            }
            AgentNodeKind::Researcher => {
                if _pin.id.input == 0 {
                    ui.label("worker");
                } else {
                    ui.label("in");
                }
            }
            _ => {
                ui.label("in");
            }
        }
        PinInfo::circle()
    }

    fn outputs(&mut self, node: &NodeData) -> usize {
        node.kind.outputs()
    }

    fn show_output(
        &mut self,
        _pin: &OutPin,
        ui: &mut egui::Ui,
        _snarl: &mut Snarl<NodeData>,
    ) -> impl egui_snarl::ui::SnarlPin + 'static {
        ui.label("out");
        PinInfo::circle()
    }

    fn has_body(&mut self, _node: &NodeData) -> bool {
        true
    }

    fn show_body(
        &mut self,
        node: NodeId,
        inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<NodeData>,
    ) {
        let node_kind = snarl.get_node(node).unwrap().kind.clone();

        match node_kind {
            AgentNodeKind::Manager => {
                let (name, global_id) = match &snarl.get_node(node).unwrap().payload {
                    NodePayload::Manager(m) => (m.name.clone(), m.global_id.clone()),
                    _ => unreachable!("kind mismatch"),
                };

                let mut erase = false;
                // Make manager node slightly wider than others.
                ui.vertical(|ui| {
                    ui.small(format!("Node {:?}", node));
                    ui.add_sized(
                        [120.0, 6.0],
                        egui::Separator::default().horizontal(),
                    );
                    ui.label(egui::RichText::new(name).strong().size(12.0));
                    ui.add_sized(
                        [120.0, 6.0],
                        egui::Separator::default().horizontal(),
                    );
                    ui.small(format!("Global ID: {}", global_id));
                    if ui.button("Erase").clicked() {
                        erase = true;
                    }
                });

                if erase {
                    snarl.remove_node(node);
                }
            }
            AgentNodeKind::Worker => {
                let my_manager_node = inputs
                    .get(0)
                    .and_then(|pin| pin.remotes.first())
                    .map(|out_pin_id| out_pin_id.node);

                let my_topic_node = inputs
                    .get(1)
                    .and_then(|pin| pin.remotes.first())
                    .map(|out_pin_id| out_pin_id.node);

                let manager_name = my_manager_node
                    .and_then(|mid| snarl.get_node(mid))
                    .and_then(|nd| match &nd.payload {
                        NodePayload::Manager(m) => Some(m.name.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| "Unassigned".to_string());

                let inferred_topic = my_topic_node.and_then(|tid| snarl.get_node(tid)).and_then(
                    |nd| match &nd.payload {
                        NodePayload::Topic(t) => Some((t.analysis_mode.clone(), t.topic.clone())),
                        _ => None,
                    },
                );

                if let Some(node_mut) = snarl.get_node_mut(node) {
                    if let NodePayload::Worker(w) = &mut node_mut.payload {
                        w.manager_node = my_manager_node;
                        w.topic_node = my_topic_node;
                        if let Some((topic_analysis_mode, topic_text)) = inferred_topic.as_ref() {
                            w.analysis_mode = topic_analysis_mode.clone();
                            w.conversation_topic = topic_text.clone();
                            w.conversation_topic_source = "Own".to_string();
                        }
                    }
                }

                let mut erase = false;
                {
                    let node_mut = snarl.get_node_mut(node).unwrap();
                    let worker_data = match &mut node_mut.payload {
                        NodePayload::Worker(w) => w,
                        _ => unreachable!("kind mismatch"),
                    };

                    ui.vertical(|ui| {
                        ui.small(format!("Node {:?}", node));
                        ui.separator();
                        AMSAgents::render_agent_worker_header(ui, &manager_name);
                        ui.separator();

                        ui.vertical(|ui| {
                            ui.spacing_mut().item_spacing = egui::Vec2::new(5.0, 2.0);

                            ui.horizontal(|ui| {
                                ui.label("Name:");
                                ui.add(egui::TextEdit::singleline(&mut worker_data.name));
                            });

                            ui.horizontal(|ui| {
                                ui.label("Instruction:");
                                egui::ComboBox::from_id_salt(
                                    ui.id().with(node.0).with("instruction_mode"),
                                )
                                .selected_text(if worker_data.instruction_mode.is_empty() {
                                    "Select".to_string()
                                } else {
                                    worker_data.instruction_mode.clone()
                                })
                                .show_ui(ui, |ui| {
                                    if ui
                                        .selectable_label(
                                            worker_data.instruction_mode == "Assistant",
                                            "Assistant",
                                        )
                                        .clicked()
                                    {
                                        worker_data.instruction_mode = "Assistant".to_string();
                                        worker_data.instruction = "You are a helpful assistant. Answer clearly, stay concise, and focus on the user request.".to_string();
                                    }
                                    if ui
                                        .selectable_label(
                                            worker_data.instruction_mode == "Math Teacher",
                                            "Math Teacher",
                                        )
                                        .clicked()
                                    {
                                        worker_data.instruction_mode = "Math Teacher".to_string();
                                    }
                                    if ui
                                        .selectable_label(
                                            worker_data.instruction_mode == "Debate",
                                            "Debate",
                                        )
                                        .clicked()
                                    {
                                        worker_data.instruction_mode = "Debate".to_string();
                                    }
                                });
                            });

                            ui.horizontal(|ui| {
                                ui.label("Instruction:");
                                ui.add(egui::TextEdit::singleline(&mut worker_data.instruction));
                            });
                        });

                        ui.separator();

                        ui.horizontal(|ui| {
                            ui.label("Topic:");
                            if my_topic_node.is_some() {
                                let preview = worker_data.conversation_topic.clone();
                                let short = if preview.chars().count() > 20 {
                                    format!(
                                        "{}…",
                                        preview.chars().take(17).collect::<String>()
                                    )
                                } else {
                                    preview
                                };
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{} — {}",
                                        worker_data.analysis_mode, short
                                    ))
                                    .weak()
                                    .size(11.0),
                                );
                            } else {
                                ui.label(
                                    egui::RichText::new("connect a Topic node (topic pin)")
                                        .weak()
                                        .size(11.0),
                                );
                            }
                        });

                        ui.small(
                            egui::RichText::new("Start/stop: use a Conversation node (wires A/B).")
                                .weak()
                                .size(10.0),
                        );

                        ui.separator();
                        ui.horizontal(|ui| {
                            if ui.button("Status").clicked() {
                                println!("=== Worker Node {:?} Status ===", node);
                                println!("Global ID: {}", worker_data.global_id);
                                println!("Manager node: {:?}", worker_data.manager_node);
                                println!("Name: {}", worker_data.name);
                            }

                            if ui.button("Erase").clicked() {
                                erase = true;
                            }
                        });
                        ui.separator();
                        ui.small(format!("Global ID: {}", worker_data.global_id));
                    });
                }

                if erase {
                    snarl.remove_node(node);
                }
            }
            AgentNodeKind::Conversation => {
                let worker_a = inputs
                    .get(0)
                    .and_then(|pin| pin.remotes.first())
                    .map(|out_pin_id| out_pin_id.node);
                let worker_b = inputs
                    .get(1)
                    .and_then(|pin| pin.remotes.first())
                    .map(|out_pin_id| out_pin_id.node);

                let mut connected_workers: Vec<NodeId> = Vec::new();
                if let Some(w) = worker_a {
                    connected_workers.push(w);
                }
                if let Some(w) = worker_b {
                    if !connected_workers.contains(&w) {
                        connected_workers.push(w);
                    }
                }
                let inferred_mode = if connected_workers.len() >= 2 {
                    "Dialogue"
                } else {
                    "Monologue"
                };
                let worker_a = connected_workers.first().copied();
                let worker_b = connected_workers.get(1).copied();

                let worker_a_name = worker_a
                    .and_then(|wid| snarl.get_node(wid))
                    .and_then(|nd| match &nd.payload {
                        NodePayload::Worker(w) => Some(w.name.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| "—".to_string());
                let worker_b_name = worker_b
                    .and_then(|wid| snarl.get_node(wid))
                    .and_then(|nd| match &nd.payload {
                        NodePayload::Worker(w) => Some(w.name.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| "—".to_string());

                let (a_topic_ok, b_topic_ok) = (
                    worker_a
                        .and_then(|wid| snarl.get_node(wid))
                        .and_then(|nd| match &nd.payload {
                            NodePayload::Worker(w) => {
                                Some(!w.conversation_topic.trim().is_empty())
                            }
                            _ => None,
                        })
                        .unwrap_or(false),
                    worker_b
                        .and_then(|wid| snarl.get_node(wid))
                        .and_then(|nd| match &nd.payload {
                            NodePayload::Worker(w) => {
                                Some(!w.conversation_topic.trim().is_empty())
                            }
                            _ => None,
                        })
                        .unwrap_or(false),
                );

                if let Some(node_mut) = snarl.get_node_mut(node) {
                    if let NodePayload::Conversation(c) = &mut node_mut.payload {
                        c.worker_a_node = worker_a;
                        c.worker_b_node = worker_b;
                        c.conversation_mode = inferred_mode.to_string();
                    }
                }

                let mut erase = false;
                {
                    let node_mut = snarl.get_node_mut(node).unwrap();
                    let conv = match &mut node_mut.payload {
                        NodePayload::Conversation(c) => c,
                        _ => unreachable!("kind mismatch"),
                    };

                    ui.vertical(|ui| {
                        ui.small(format!("Node {:?}", node));
                        ui.separator();
                        ui.label(egui::RichText::new(&conv.name).strong().size(12.0));
                        ui.separator();

                        ui.horizontal(|ui| {
                            ui.label("Agent A:");
                            ui.label(egui::RichText::new(&worker_a_name).weak());
                        });
                        ui.horizontal(|ui| {
                            ui.label("Agent B:");
                            ui.label(egui::RichText::new(&worker_b_name).weak());
                        });

                        ui.horizontal(|ui| {
                            ui.label("Mode:");
                            ui.label(egui::RichText::new(inferred_mode).weak());
                        });

                        ui.small(
                            egui::RichText::new(
                                "Connect Worker nodes to Evaluator/Researcher independently to auto-enable each one.",
                            )
                            .weak()
                            .size(10.0),
                        );

                        let can_start = (inferred_mode == "Monologue"
                            && worker_a.is_some()
                            && a_topic_ok)
                            || (inferred_mode == "Dialogue"
                                && worker_a.is_some()
                                && worker_b.is_some()
                                && a_topic_ok
                                && b_topic_ok);

                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label(if conv.conversation_active {
                                egui::RichText::new("Status: Running").weak()
                            } else {
                                egui::RichText::new("Status: Stopped").weak()
                            });
                            if ui.button("Erase").clicked() {
                                erase = true;
                            }
                        });

                        if !can_start && !conv.conversation_active {
                            ui.small(
                                egui::RichText::new(
                                    "Monologue: wire A + Topic on A. Dialogue: wire A & B + topics.",
                                )
                                .weak()
                                .size(10.0),
                            );
                        }

                        ui.separator();
                        ui.small(format!("Global ID: {}", conv.global_id));
                    });
                }

                if erase {
                    snarl.remove_node(node);
                }
            }
            AgentNodeKind::Evaluator => {
                let my_worker_node = inputs
                    .get(0)
                    .and_then(|pin| pin.remotes.first())
                    .map(|out_pin_id| out_pin_id.node);

                let inferred_manager_node = my_worker_node
                    .and_then(|wn| snarl.get_node(wn))
                    .and_then(|nd| match &nd.payload {
                        NodePayload::Worker(w) => w.manager_node,
                        _ => None,
                    });

                let manager_name = inferred_manager_node
                    .and_then(|mid| snarl.get_node(mid))
                    .and_then(|nd| match &nd.payload {
                        NodePayload::Manager(m) => Some(m.name.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| "Unassigned".to_string());

                // Update inferred wire relationships.
                if let Some(node_mut) = snarl.get_node_mut(node) {
                    if let NodePayload::Evaluator(e) = &mut node_mut.payload {
                        e.worker_node = my_worker_node;
                        e.manager_node = inferred_manager_node;
                        // Connected to a Worker node => behaves like clicking "Evaluate".
                        e.active = my_worker_node.is_some();
                    }
                }

                let mut erase = false;
                {
                    let node_mut = snarl.get_node_mut(node).unwrap();
                    let evaluator_data = match &mut node_mut.payload {
                        NodePayload::Evaluator(e) => e,
                        _ => unreachable!("kind mismatch"),
                    };

                    ui.vertical(|ui| {
                            ui.small(format!("Node {:?}", node));
                            ui.separator();
                            AMSAgents::render_agent_evaluator_header(
                                ui,
                                &manager_name,
                            );
                            ui.separator();

                            ui.horizontal(|ui| {
                                ui.label("Name:");
                                ui.add(egui::TextEdit::singleline(
                                    &mut evaluator_data.name,
                                ));
                            });

                            ui.horizontal(|ui| {
                                ui.label("Analysis:");
                                egui::ComboBox::from_id_salt(
                                    ui.id().with(node.0).with(
                                        "eval_analysis_mode",
                                    ),
                                )
                                .selected_text(if evaluator_data
                                    .analysis_mode
                                    .is_empty()
                                {
                                    "Select".to_string()
                                } else {
                                    evaluator_data.analysis_mode.clone()
                                })
                                .show_ui(ui, |ui| {
                                    if ui.selectable_label(
                                        evaluator_data.analysis_mode
                                            == "Topic Extraction",
                                        "Topic Extraction",
                                    ).clicked() {
                                        evaluator_data.analysis_mode =
                                            "Topic Extraction".to_string();
                                        evaluator_data.instruction = "Topic Extraction: extract the topic in 1 or 2 words. Identify what is the topic of the sentence being analysed.".to_string();
                                    }
                                    if ui.selectable_label(
                                        evaluator_data.analysis_mode
                                            == "Decision Analysis",
                                        "Decision Analysis",
                                    ).clicked() {
                                        evaluator_data.analysis_mode =
                                            "Decision Analysis".to_string();
                                        evaluator_data.instruction = "Decision Analysis: extract a decision in 1 or 2 sentences about the agent in the message being analysed. Focus on the concrete decision and its intent.".to_string();
                                    }
                                    if ui.selectable_label(
                                        evaluator_data.analysis_mode
                                            == "Sentiment Classification",
                                        "Sentiment Classification",
                                    ).clicked() {
                                        evaluator_data.analysis_mode =
                                            "Sentiment Classification".to_string();
                                        evaluator_data.instruction = "Sentiment Classification: extract the sentiment of the message being analysed and return one word that is the sentiment.".to_string();
                                    }
                                });
                            });

                            ui.horizontal(|ui| {
                                ui.label("Instruction:");
                                ui.add(egui::TextEdit::singleline(
                                    &mut evaluator_data.instruction,
                                ));
                            });

                            ui.horizontal(|ui| {
                                if ui
                                    .checkbox(&mut evaluator_data.limit_token, "Limit Token")
                                    .changed()
                                {
                                    if !evaluator_data.limit_token {
                                        evaluator_data.num_predict.clear();
                                    }
                                }
                                if evaluator_data.limit_token {
                                    ui.label("num_predict:");
                                    ui.add(egui::TextEdit::singleline(
                                        &mut evaluator_data.num_predict,
                                    ).desired_width(80.0));
                                }
                            });

                            ui.separator();
                            let is_connected = evaluator_data.worker_node.is_some();
                            let button = if is_connected {
                                egui::Button::new("Evaluating (connected)")
                            } else {
                                egui::Button::new(if evaluator_data.active {
                                    "Stop Evaluating"
                                } else {
                                    "Evaluate"
                                })
                            };

                            ui.horizontal(|ui| {
                                if !is_connected && ui.add(button).clicked() {
                                    evaluator_data.active = !evaluator_data.active;
                                }

                                if ui.button("Status").clicked() {
                                    println!(
                                        "=== Evaluator Node {:?} Status ===",
                                        node
                                    );
                                    println!(
                                        "Global ID: {}",
                                        evaluator_data.global_id
                                    );
                                    println!(
                                        "Name: {}",
                                        evaluator_data.name
                                    );
                                }

                                if ui.button("Erase").clicked() {
                                    erase = true;
                                }
                            });
                            ui.separator();
                            ui.small(format!("Global ID: {}", evaluator_data.global_id));
                    });
                }

                if erase {
                    snarl.remove_node(node);
                }
            }
            AgentNodeKind::Researcher => {
                let my_worker_node = inputs
                    .get(0)
                    .and_then(|pin| pin.remotes.first())
                    .map(|out_pin_id| out_pin_id.node);

                let inferred_manager_node = my_worker_node
                    .and_then(|wn| snarl.get_node(wn))
                    .and_then(|nd| match &nd.payload {
                        NodePayload::Worker(w) => w.manager_node,
                        _ => None,
                    });

                let manager_name = inferred_manager_node
                    .and_then(|mid| snarl.get_node(mid))
                    .and_then(|nd| match &nd.payload {
                        NodePayload::Manager(m) => Some(m.name.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| "Unassigned".to_string());

                if let Some(node_mut) = snarl.get_node_mut(node) {
                    if let NodePayload::Researcher(r) = &mut node_mut.payload {
                        r.worker_node = my_worker_node;
                        r.manager_node = inferred_manager_node;
                        // Connected to a Worker node => behaves like clicking "Research".
                        r.active = my_worker_node.is_some();
                    }
                }

                let mut erase = false;
                {
                    let node_mut = snarl.get_node_mut(node).unwrap();
                    let researcher_data = match &mut node_mut.payload {
                        NodePayload::Researcher(r) => r,
                        _ => unreachable!("kind mismatch"),
                    };

                    ui.vertical(|ui| {
                            ui.small(format!("Node {:?}", node));
                            ui.separator();
                            AMSAgents::render_agent_researcher_header(
                                ui,
                                &manager_name,
                            );
                            ui.separator();

                            ui.horizontal(|ui| {
                                ui.label("Name:");
                                ui.add(egui::TextEdit::singleline(
                                    &mut researcher_data.name,
                                ));
                            });

                            ui.horizontal(|ui| {
                                ui.label("Topics:");
                                egui::ComboBox::from_id_salt(
                                    ui.id().with(node.0).with(
                                        "research_topic_mode",
                                    ),
                                )
                                .selected_text(if researcher_data.topic_mode
                                    .is_empty()
                                {
                                    "Select".to_string()
                                } else {
                                    researcher_data.topic_mode.clone()
                                })
                                .show_ui(ui, |ui| {
                                    if ui.selectable_label(
                                        researcher_data.topic_mode == "Articles",
                                        "Articles",
                                    ).clicked() {
                                        researcher_data.topic_mode =
                                            "Articles".to_string();
                                        researcher_data.instruction = "Generate article references connected to the message context. Prefer a mix of classic and recent pieces.".to_string();
                                    }
                                    if ui.selectable_label(
                                        researcher_data.topic_mode == "Movies",
                                        "Movies",
                                    ).clicked() {
                                        researcher_data.topic_mode =
                                            "Movies".to_string();
                                        researcher_data.instruction = "Generate movie references connected to the message context. Prefer diverse genres and well-known titles.".to_string();
                                    }
                                    if ui.selectable_label(
                                        researcher_data.topic_mode == "Music",
                                        "Music",
                                    ).clicked() {
                                        researcher_data.topic_mode =
                                            "Music".to_string();
                                        researcher_data.instruction = "Generate music references connected to the message context. Include artist and track or album when possible.".to_string();
                                    }
                                });
                            });

                            ui.horizontal(|ui| {
                                ui.label("Instruction:");
                                ui.add(egui::TextEdit::singleline(
                                    &mut researcher_data.instruction,
                                ));
                            });

                            ui.horizontal(|ui| {
                                if ui
                                    .checkbox(&mut researcher_data.limit_token, "Token")
                                    .changed()
                                {
                                    if !researcher_data.limit_token {
                                        researcher_data.num_predict.clear();
                                    }
                                }
                                if researcher_data.limit_token {
                                    ui.label("num_predict:");
                                    ui.add(egui::TextEdit::singleline(
                                        &mut researcher_data.num_predict,
                                    ).desired_width(80.0));
                                }
                            });

                            ui.separator();
                            let is_connected = researcher_data.worker_node.is_some();
                            let button = if is_connected {
                                egui::Button::new("Researching (connected)")
                            } else {
                                egui::Button::new(if researcher_data.active {
                                    "Stop Researching"
                                } else {
                                    "Research"
                                })
                            };

                            ui.horizontal(|ui| {
                                if !is_connected && ui.add(button).clicked() {
                                    researcher_data.active = !researcher_data.active;
                                }

                                if ui.button("Status").clicked() {
                                    println!(
                                        "=== Researcher Node {:?} Status ===",
                                        node
                                    );
                                    println!(
                                        "Global ID: {}",
                                        researcher_data.global_id
                                    );
                                    println!(
                                        "Name: {}",
                                        researcher_data.name
                                    );
                                }

                                if ui.button("Erase").clicked() {
                                    erase = true;
                                }
                            });
                            ui.separator();
                            ui.small(format!("Global ID: {}", researcher_data.global_id));
                    });
                }

                if erase {
                    snarl.remove_node(node);
                }
            }
            AgentNodeKind::Topic => {
                let mut erase = false;
                {
                    let node_mut = snarl.get_node_mut(node).unwrap();
                    let topic_data = match &mut node_mut.payload {
                        NodePayload::Topic(t) => t,
                        _ => unreachable!("kind mismatch"),
                    };

                    ui.vertical(|ui| {
                        ui.small(format!("Node {:?}", node));
                        ui.separator();
                        ui.label(
                            egui::RichText::new(&topic_data.name)
                                .strong()
                                .size(12.0),
                        );
                        ui.separator();

                        // Topic preset + topic text (mirrors Agent Worker widgets).
                        ui.vertical(|ui| {
                            ui.horizontal(|ui| {
                                ui.label("Topic:");
                                egui::ComboBox::from_id_salt(
                                    ui.id().with(node.0).with("topic_analysis_mode"),
                                )
                                .selected_text(if topic_data.analysis_mode.is_empty() {
                                    "Select".to_string()
                                } else {
                                    topic_data.analysis_mode.clone()
                                })
                                .show_ui(ui, |ui| {
                                    if ui
                                        .selectable_label(
                                            topic_data.analysis_mode
                                                == "European Politics",
                                            "European Politics",
                                        )
                                        .clicked()
                                    {
                                        topic_data.analysis_mode =
                                            "European Politics".to_string();
                                        topic_data.topic = "Discuss European Politics and provide a concise overview of the main issue in one or two sentences.".to_string();
                                    }
                                    if ui
                                        .selectable_label(
                                            topic_data.analysis_mode
                                                == "Mental Health",
                                            "Mental Health",
                                        )
                                        .clicked()
                                    {
                                        topic_data.analysis_mode =
                                            "Mental Health".to_string();
                                        topic_data.topic = "Discuss Mental Health and provide one or two practical insights about the topic.".to_string();
                                    }
                                    if ui
                                        .selectable_label(
                                            topic_data.analysis_mode
                                                == "Electronics",
                                            "Electronics",
                                        )
                                        .clicked()
                                    {
                                        topic_data.analysis_mode =
                                            "Electronics".to_string();
                                        topic_data.topic = "Discuss Electronics and summarize one or two important points about the selected subject.".to_string();
                                    }
                                });
                            });

                            ui.horizontal(|ui| {
                                ui.label("Topic:");
                                ui.add(egui::TextEdit::singleline(
                                    &mut topic_data.topic,
                                ));
                            });
                        });

                        ui.separator();
                        if ui.button("Erase").clicked() {
                            erase = true;
                        }
                        ui.separator();
                        ui.small(format!("Global ID: {}", topic_data.global_id));
                    });
                }

                if erase {
                    snarl.remove_node(node);
                }
            }
        }
    }

    fn has_graph_menu(&mut self, _pos: egui::Pos2, _snarl: &mut Snarl<NodeData>) -> bool {
        true
    }

    fn show_graph_menu(
        &mut self,
        pos: egui::Pos2,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<NodeData>,
    ) {
        ui.vertical(|ui| {
            ui.label("Add Node");
            ui.separator();

            ui.small(egui::RichText::new("Agent").strong());
            for kind in [
                AgentNodeKind::Manager,
                AgentNodeKind::Worker,
                AgentNodeKind::Evaluator,
                AgentNodeKind::Researcher,
            ] {
                debug_assert_eq!(kind.category(), NodeCategory::Agent);
                if ui.button(kind.label()).clicked() {
                    let mut node = match kind {
                        AgentNodeKind::Manager => NodeData::new_manager(),
                        AgentNodeKind::Worker => NodeData::new_worker(),
                        AgentNodeKind::Evaluator => NodeData::new_evaluator(),
                        AgentNodeKind::Researcher => NodeData::new_researcher(),
                        _ => unreachable!("category mismatch"),
                    };
                    node.set_name(Self::numbered_name_for_kind(snarl, kind));
                    snarl.insert_node(pos, node);
                }
            }

            ui.separator();
            ui.small(egui::RichText::new("Tool").strong());
            debug_assert_eq!(AgentNodeKind::Topic.category(), NodeCategory::Tool);
            if ui.button(AgentNodeKind::Topic.label()).clicked() {
                let mut node = NodeData::new_topic();
                node.set_name(Self::numbered_name_for_kind(snarl, AgentNodeKind::Topic));
                snarl.insert_node(pos, node);
            }

            ui.separator();
            ui.small(egui::RichText::new("Output").strong());
            debug_assert_eq!(
                AgentNodeKind::Conversation.category(),
                NodeCategory::Output
            );
            if ui.button(AgentNodeKind::Conversation.label()).clicked() {
                let mut node = NodeData::new_conversation();
                node.set_name(Self::numbered_name_for_kind(snarl, AgentNodeKind::Conversation));
                snarl.insert_node(pos, node);
            }
        });
    }
}

impl AMSAgents {
    fn print_nodes_graph_snapshot(&self) {
        let mut nodes: Vec<(usize, String, &'static str)> = self
            .nodes_panel
            .snarl
            .nodes_ids_data()
            .map(|(id, node)| {
                let (name, kind) = match &node.value.payload {
                    NodePayload::Manager(m) => (m.name.clone(), "Manager"),
                    NodePayload::Worker(w) => (w.name.clone(), "Worker"),
                    NodePayload::Conversation(c) => (c.name.clone(), "Conversation"),
                    NodePayload::Evaluator(e) => (e.name.clone(), "Evaluator"),
                    NodePayload::Researcher(r) => (r.name.clone(), "Researcher"),
                    NodePayload::Topic(t) => (t.name.clone(), "Topic"),
                };
                (id.0, name, kind)
            })
            .collect();
        nodes.sort_by_key(|(id, _, _)| *id);

        let label_by_id: HashMap<usize, String> = nodes
            .iter()
            .map(|(id, name, kind)| (*id, format!("{} [{}]", name, kind)))
            .collect();

        let mut edges: Vec<String> = Vec::new();
        for (id, node) in self.nodes_panel.snarl.nodes_ids_data() {
            match &node.value.payload {
                NodePayload::Worker(w) => {
                    if let Some(mid) = w.manager_node {
                        edges.push(format!(
                            "{} -> {}  (manager_to_worker)",
                            label_by_id.get(&mid.0).cloned().unwrap_or_else(|| format!("Node {}", mid.0)),
                            label_by_id.get(&id.0).cloned().unwrap_or_else(|| format!("Node {}", id.0)),
                        ));
                    }
                    if let Some(tid) = w.topic_node {
                        edges.push(format!(
                            "{} -> {}  (topic_to_worker)",
                            label_by_id.get(&tid.0).cloned().unwrap_or_else(|| format!("Node {}", tid.0)),
                            label_by_id.get(&id.0).cloned().unwrap_or_else(|| format!("Node {}", id.0)),
                        ));
                    }
                }
                NodePayload::Conversation(c) => {
                    if let Some(wid) = c.worker_a_node {
                        edges.push(format!(
                            "{} -> {}  (worker_to_conversation_A)",
                            label_by_id.get(&wid.0).cloned().unwrap_or_else(|| format!("Node {}", wid.0)),
                            label_by_id.get(&id.0).cloned().unwrap_or_else(|| format!("Node {}", id.0)),
                        ));
                    }
                    if let Some(wid) = c.worker_b_node {
                        edges.push(format!(
                            "{} -> {}  (worker_to_conversation_B)",
                            label_by_id.get(&wid.0).cloned().unwrap_or_else(|| format!("Node {}", wid.0)),
                            label_by_id.get(&id.0).cloned().unwrap_or_else(|| format!("Node {}", id.0)),
                        ));
                    }
                }
                NodePayload::Evaluator(e) => {
                    if let Some(wid) = e.worker_node {
                        edges.push(format!(
                            "{} -> {}  (worker_to_evaluator)",
                            label_by_id.get(&wid.0).cloned().unwrap_or_else(|| format!("Node {}", wid.0)),
                            label_by_id.get(&id.0).cloned().unwrap_or_else(|| format!("Node {}", id.0)),
                        ));
                    }
                    if let Some(mid) = e.manager_node {
                        edges.push(format!(
                            "{} -> {}  (evaluator_to_manager)",
                            label_by_id.get(&id.0).cloned().unwrap_or_else(|| format!("Node {}", id.0)),
                            label_by_id.get(&mid.0).cloned().unwrap_or_else(|| format!("Node {}", mid.0)),
                        ));
                    }
                }
                NodePayload::Researcher(r) => {
                    if let Some(wid) = r.worker_node {
                        edges.push(format!(
                            "{} -> {}  (worker_to_researcher)",
                            label_by_id.get(&wid.0).cloned().unwrap_or_else(|| format!("Node {}", wid.0)),
                            label_by_id.get(&id.0).cloned().unwrap_or_else(|| format!("Node {}", id.0)),
                        ));
                    }
                    if let Some(mid) = r.manager_node {
                        edges.push(format!(
                            "{} -> {}  (researcher_to_manager)",
                            label_by_id.get(&id.0).cloned().unwrap_or_else(|| format!("Node {}", id.0)),
                            label_by_id.get(&mid.0).cloned().unwrap_or_else(|| format!("Node {}", mid.0)),
                        ));
                    }
                }
                NodePayload::Manager(_) | NodePayload::Topic(_) => {}
            }
        }
        edges.sort();

        println!("=== Run Graph ===");
        println!("Nodes ({}):", nodes.len());
        for (id, name, kind) in &nodes {
            println!("  [{}] {} ({})", id, name, kind);
        }
        println!("Edges ({}):", edges.len());
        for e in &edges {
            println!("  {}", e);
        }
    }

    fn stop_graph(&mut self) {
        // Stop all currently running conversation loops keyed by Conversation node id.
        for (_, flag, _) in &self.conversation_loop_handles {
            *flag.lock().unwrap() = false;
        }
        self.conversation_loop_handles.clear();

        // Reflect stopped state in Conversation nodes.
        for (_, node) in self.nodes_panel.snarl.nodes_ids_data_mut() {
            if let NodePayload::Conversation(c) = &mut node.value.payload {
                c.conversation_active = false;
            }
        }
    }

    fn run_graph(&mut self) {
        // Bulletproof behavior: re-run means stop existing graph processes first.
        self.stop_graph();

        let mut to_start: Vec<(
            NodeId,
            NodeId,
            String,
            String,
            String,
            String,
            usize,
            String,
            String,
            String,
            String,
        )> = Vec::new();

        for (conv_id, node) in self.nodes_panel.snarl.nodes_ids_data() {
            let NodePayload::Conversation(c) = &node.value.payload else {
                continue;
            };

            let Some(worker_a_id) = c.worker_a_node else {
                continue;
            };
            let Some(w_a) = self.nodes_panel.snarl.get_node(worker_a_id) else {
                continue;
            };
            let NodePayload::Worker(wa) = &w_a.payload else {
                continue;
            };

            let a_topic_ok = !wa.conversation_topic.trim().is_empty();
            let (
                agent_b_id,
                agent_b_name,
                agent_b_instruction,
                agent_b_topic,
                agent_b_topic_source,
            ) = if c.conversation_mode == "Monologue" {
                if !a_topic_ok {
                    continue;
                }
                (
                    worker_a_id.0,
                    wa.name.clone(),
                    wa.instruction.clone(),
                    wa.conversation_topic.clone(),
                    wa.conversation_topic_source.clone(),
                )
            } else {
                let Some(worker_b_id) = c.worker_b_node else {
                    continue;
                };
                let Some(w_b) = self.nodes_panel.snarl.get_node(worker_b_id) else {
                    continue;
                };
                let NodePayload::Worker(wb) = &w_b.payload else {
                    continue;
                };
                let b_topic_ok = !wb.conversation_topic.trim().is_empty();
                if !a_topic_ok || !b_topic_ok {
                    continue;
                }
                (
                    worker_b_id.0,
                    wb.name.clone(),
                    wb.instruction.clone(),
                    wb.conversation_topic.clone(),
                    wb.conversation_topic_source.clone(),
                )
            };

            to_start.push((
                conv_id,
                worker_a_id,
                wa.name.clone(),
                wa.instruction.clone(),
                wa.conversation_topic.clone(),
                wa.conversation_topic_source.clone(),
                agent_b_id,
                agent_b_name,
                agent_b_instruction,
                agent_b_topic,
                agent_b_topic_source,
            ));
        }

        let started: HashSet<NodeId> = to_start.iter().map(|(id, ..)| *id).collect();
        for (
            conv_id,
            worker_a_id,
            a_name,
            a_instr,
            a_topic,
            a_src,
            b_id,
            b_name,
            b_instr,
            b_topic,
            b_src,
        ) in to_start
        {
            self.start_conversation_from_node_worker_resolved(
                conv_id,
                worker_a_id,
                a_name,
                a_instr,
                a_topic,
                a_src,
                b_id,
                b_name,
                b_instr,
                b_topic,
                b_src,
            );
        }

        for (id, node) in self.nodes_panel.snarl.nodes_ids_data_mut() {
            if let NodePayload::Conversation(c) = &mut node.value.payload {
                c.conversation_active = started.contains(&id);
            }
        }
    }

    /// Starts the conversation loop for a **Conversation** node using the same ADK/HTTP flow as the old Workspace.
    /// `agent_a_node_id` is the primary worker (input A); the async handle is keyed by `conversation_node_id`.
    fn start_conversation_from_node_worker_resolved(
        &mut self,
        conversation_node_id: NodeId,
        agent_a_node_id: NodeId,
        agent_a_name: String,
        agent_a_instruction: String,
        agent_a_topic: String,
        agent_a_topic_source: String,
        agent_b_id: usize,
        agent_b_name: String,
        agent_b_instruction: String,
        agent_b_topic: String,
        agent_b_topic_source: String,
    ) {
        let agent_a_id = agent_a_node_id.0;
        let active_flag = Arc::new(Mutex::new(true));
        let flag_clone = active_flag.clone();
        let endpoint = self.http_endpoint.clone();
        let last_msg = self.last_message_in_chat.clone();
        let message_events = self.conversation_message_events.clone();
        let selected_model = if self.selected_ollama_model.trim().is_empty() {
            None
        } else {
            Some(self.selected_ollama_model.clone())
        };
        let history_size = self.conversation_history_size;
        let turn_delay_secs = self.conversation_turn_delay_secs;
        let handle = self.rt_handle.clone();

        let loop_handle = handle.spawn(async move {
            crate::agent_conversation_loop::start_conversation_loop(
                agent_a_id,
                agent_a_name,
                agent_a_instruction,
                agent_a_topic,
                agent_a_topic_source,
                agent_b_id,
                agent_b_name,
                agent_b_instruction,
                agent_b_topic,
                agent_b_topic_source,
                endpoint,
                flag_clone,
                last_msg,
                message_events,
                selected_model,
                history_size,
                turn_delay_secs,
            )
            .await;
        });

        self.conversation_loop_handles
            .push((conversation_node_id.0, active_flag, loop_handle));
    }

    pub(super) fn render_nodes_panel(&mut self, ui: &mut egui::Ui) {
        let panel_border_color = ui.visuals().widgets.noninteractive.bg_stroke.color;
        let nodes_panel = egui::Frame::default()
            .fill(egui::Color32::from_rgb(40, 40, 40))
            .stroke(egui::Stroke::new(1.0, panel_border_color))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::same(6));

        // Nodes panel extends to bottom of window (no fixed Outgoing HTTP panel).
        let panel_height = ui.available_height().max(120.0);
        let mut viewer = BasicNodeViewer;

        // Customize Snarl appearance:
        // - start zoom at ~1.0 (clamp initial scaling)
        // - place pins on the node edge (the "ball" on border)
        // - thin grey grid lines
        let mut style = SnarlStyle::new();
        style.min_scale = Some(0.25);
        style.max_scale = Some(2.0);
        style.pin_placement = Some(PinPlacement::Edge);
        style.wire_style = Some(self.nodes_panel.wire_style);
        style.wire_width = Some(3.0);
        style.wire_smoothness = Some(0.0);
        style.pin_stroke = Some(egui::Stroke::new(1.5, egui::Color32::from_gray(200)));
        // Vertical + horizontal grid (no rotation).
        style.bg_pattern = Some(BackgroundPattern::grid(egui::vec2(50.0, 50.0), 0.0));
        style.bg_pattern_stroke = Some(egui::Stroke::new(
            1.0,
            egui::Color32::from_gray(110).gamma_multiply(0.5),
        ));

        ui.allocate_ui_with_layout(
            egui::vec2(ui.available_width(), panel_height),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                nodes_panel.show(ui, |ui| {
                    let mut run_graph_requested = false;
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Nodes").strong().size(12.0));
                        ui.add_space(8.0);
                        let label = match self.nodes_panel.wire_style {
                            WireStyle::Bezier5 => "Wires: Bezier5",
                            WireStyle::Line => "Wires: Line",
                            _ => "Wires: (custom)",
                        };
                        if ui.button(label).clicked() {
                            self.nodes_panel.wire_style = match self.nodes_panel.wire_style {
                                WireStyle::Bezier5 => WireStyle::Line,
                                _ => WireStyle::Bezier5,
                            };
                        }
                        if ui.button("Run Graph").clicked() {
                            run_graph_requested = true;
                        }
                        if ui.button("Stop Graph").clicked() {
                            self.stop_graph();
                        }
                    });
                    ui.add_space(4.0);
                    SnarlWidget::new()
                        .id_salt("ams_nodes_panel_v2")
                        .style(style)
                        .show(&mut self.nodes_panel.snarl, &mut viewer, ui);
                    if run_graph_requested {
                        self.print_nodes_graph_snapshot();
                        self.run_graph();
                    }
                });

                let ctx = ui.ctx().clone();
                let mut pending_events = {
                    let mut q = self.conversation_message_events.lock().unwrap();
                    std::mem::take(&mut *q)
                };
                // Backward-compatible fallback if queue is empty but latest still has a message.
                if pending_events.is_empty() {
                    if let Some(last_msg) = self.last_message_in_chat.lock().unwrap().clone() {
                        pending_events.push(last_msg);
                    }
                }
                // Output node category is currently represented by Conversation nodes.
                let has_output_nodes = self.nodes_panel.snarl.nodes_ids_data().any(|(_, node)| {
                    matches!(node.value.payload, NodePayload::Conversation(_))
                });

                // Queue new events for active Evaluator nodes.
                for (id, node) in self.nodes_panel.snarl.nodes_ids_data() {
                    if let NodePayload::Evaluator(e) = &node.value.payload {
                        if !e.active {
                            continue;
                        }
                        for raw in &pending_events {
                            let (event_key, message) = if let Some((key, body)) =
                                raw.split_once("::MSG::")
                            {
                                (key.to_string(), body.to_string())
                            } else {
                                ("LEGACY".to_string(), raw.clone())
                            };
                            let last_eval = self
                                .last_evaluated_message_by_evaluator
                                .lock()
                                .unwrap()
                                .get(&id.0)
                                .cloned();
                            if event_key.is_empty() || last_eval.as_ref() == Some(&event_key) {
                                continue;
                            }
                            self.last_evaluated_message_by_evaluator
                                .lock()
                                .unwrap()
                                .insert(id.0, event_key.clone());
                            self.evaluator_event_queues
                                .lock()
                                .unwrap()
                                .entry(id.0)
                                .or_default()
                                .push_back((event_key, message));
                        }
                    }
                }

                // Queue new events for active Researcher nodes.
                for (id, node) in self.nodes_panel.snarl.nodes_ids_data() {
                    if let NodePayload::Researcher(r) = &node.value.payload {
                        if !r.active {
                            continue;
                        }
                        for raw in &pending_events {
                            let (event_key, message) = if let Some((key, body)) =
                                raw.split_once("::MSG::")
                            {
                                (key.to_string(), body.to_string())
                            } else {
                                ("LEGACY".to_string(), raw.clone())
                            };
                            let last_research = self
                                .last_researched_message_by_researcher
                                .lock()
                                .unwrap()
                                .get(&id.0)
                                .cloned();
                            if event_key.is_empty()
                                || last_research.as_ref() == Some(&event_key)
                            {
                                continue;
                            }
                            self.last_researched_message_by_researcher
                                .lock()
                                .unwrap()
                                .insert(id.0, event_key.clone());
                            self.researcher_event_queues
                                .lock()
                                .unwrap()
                                .entry(id.0)
                                .or_default()
                                .push_back((event_key, message));
                        }
                    }
                }

                // Start at most one Evaluator inference per node (strict sequential order).
                for (id, node) in self.nodes_panel.snarl.nodes_ids_data() {
                    let NodePayload::Evaluator(e) = &node.value.payload else {
                        continue;
                    };
                    if !e.active {
                        continue;
                    }
                    if self.evaluator_inflight_nodes.lock().unwrap().contains(&id.0) {
                        continue;
                    }
                    let next_msg = self
                        .evaluator_event_queues
                        .lock()
                        .unwrap()
                        .entry(id.0)
                        .or_default()
                        .pop_front();
                    let Some((_, message)) = next_msg else {
                        continue;
                    };

                    self.evaluator_inflight_nodes.lock().unwrap().insert(id.0);
                    let inflight = self.evaluator_inflight_nodes.clone();
                    let node_key = id.0;
                    let instruction = e.instruction.clone();
                    let analysis_mode = e.analysis_mode.clone();
                    let limit_token = e.limit_token;
                    let num_predict = e.num_predict.clone();
                    let endpoint = self.http_endpoint.clone();
                    let has_output_nodes = has_output_nodes;
                    let ctx = ctx.clone();
                    let handle = self.rt_handle.clone();
                    let selected_model = if self.selected_ollama_model.trim().is_empty() {
                        None
                    } else {
                        Some(self.selected_ollama_model.clone())
                    };
                    handle.spawn(async move {
                        match crate::adk_integration::send_to_ollama(
                            &instruction,
                            &message,
                            limit_token,
                            &num_predict,
                            selected_model.as_deref(),
                        )
                        .await
                        {
                            Ok(response) => {
                                let response_lower = response.to_lowercase();
                                let sentiment = match analysis_mode.as_str() {
                                    "Topic Extraction" => "topic",
                                    "Decision Analysis" => "decision",
                                    "Sentiment Classification" => {
                                        if response_lower.contains("positive")
                                            || response_lower.contains("happy")
                                        {
                                            "sentiment"
                                        } else if response_lower.contains("negative")
                                            || response_lower.contains("sad")
                                            || response_lower.contains("angry")
                                            || response_lower.contains("frustrated")
                                        {
                                            "sentiment"
                                        } else if response_lower.contains("neutral") {
                                            "sentiment"
                                        } else {
                                            "unknown"
                                        }
                                    }
                                    _ => {
                                        if response_lower.contains("happy") {
                                            "happy"
                                        } else if response_lower.contains("sad") {
                                            "sad"
                                        } else {
                                            "analysis"
                                        }
                                    }
                                };
                                if has_output_nodes {
                                    if let Err(e) = crate::http_client::send_evaluator_result(
                                        &endpoint,
                                        "Agent Evaluator",
                                        sentiment,
                                        &response,
                                    )
                                    .await
                                    {
                                        eprintln!("[Evaluator] Failed to send to ams-chat: {}", e);
                                    }
                                }
                            }
                            Err(e) => eprintln!("[Evaluator] Ollama error: {}", e),
                        }
                        inflight.lock().unwrap().remove(&node_key);
                        ctx.request_repaint();
                    });
                }

                // Start at most one Researcher inference per node (strict sequential order).
                for (id, node) in self.nodes_panel.snarl.nodes_ids_data() {
                    let NodePayload::Researcher(r) = &node.value.payload else {
                        continue;
                    };
                    if !r.active {
                        continue;
                    }
                    if self.researcher_inflight_nodes.lock().unwrap().contains(&id.0) {
                        continue;
                    }
                    let next_msg = self
                        .researcher_event_queues
                        .lock()
                        .unwrap()
                        .entry(id.0)
                        .or_default()
                        .pop_front();
                    let Some((_, message)) = next_msg else {
                        continue;
                    };

                    self.researcher_inflight_nodes.lock().unwrap().insert(id.0);
                    let inflight = self.researcher_inflight_nodes.clone();
                    let node_key = id.0;
                    let topic = if r.topic_mode.trim().is_empty() {
                        "Articles".to_string()
                    } else {
                        r.topic_mode.clone()
                    };
                    let instruction = format!(
                        "{}\n\nUsing the latest chat message, suggest 3 {} references related to what was said. Keep it concise with bullet points: title and one-line why it matches.",
                        r.instruction,
                        topic.to_lowercase()
                    );
                    let limit_token = r.limit_token;
                    let num_predict = r.num_predict.clone();
                    let endpoint = self.http_endpoint.clone();
                    let has_output_nodes = has_output_nodes;
                    let ctx = ctx.clone();
                    let handle = self.rt_handle.clone();
                    let selected_model = if self.selected_ollama_model.trim().is_empty() {
                        None
                    } else {
                        Some(self.selected_ollama_model.clone())
                    };
                    handle.spawn(async move {
                        match crate::adk_integration::send_to_ollama(
                            &instruction,
                            &message,
                            limit_token,
                            &num_predict,
                            selected_model.as_deref(),
                        )
                        .await
                        {
                            Ok(response) => {
                                if has_output_nodes {
                                    if let Err(e) = crate::http_client::send_researcher_result(
                                        &endpoint,
                                        "Agent Researcher",
                                        &topic,
                                        &response,
                                    )
                                    .await
                                    {
                                        eprintln!("[Researcher] Failed to send to ams-chat: {}", e);
                                    }
                                }
                            }
                            Err(e) => eprintln!("[Researcher] Ollama error: {}", e),
                        }
                        inflight.lock().unwrap().remove(&node_key);
                        ctx.request_repaint();
                    });
                }
            },
        );
    }
}
