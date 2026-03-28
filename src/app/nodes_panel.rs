use super::AMSAgents;
use crate::event_ledger::EventLedger;
use crate::reproducibility::{
    APP_NAME, GraphSnapshot, MANIFEST_VERSION, ManifestEdge, ManifestNode, RunContext, RunManifest,
    RunRuntimeSettings, canonical_graph_signature, derive_experiment_id, export_manifest_to,
    new_run_id, now_rfc3339_utc, read_manifest, runs_root, write_manifest,
};
use eframe::egui;
use egui_phosphor::regular;
use rand::Rng;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Preset topic labels and short conversation prompts (ethical, discussion-oriented).
const TOPIC_PRESETS: &[(&str, &str)] = &[
    (
        "European Politics",
        "Discuss European Politics and provide a concise overview of the main issue in one or two sentences.",
    ),
    (
        "Mental Health",
        "Discuss Mental Health and provide one or two practical insights about the topic.",
    ),
    (
        "Electronics",
        "Discuss Electronics and summarize one or two important points about the selected subject.",
    ),
    (
        "Climate & Environmental Justice",
        "Discuss how climate policy can balance urgency with fairness across communities; give one or two measured points.",
    ),
    (
        "Digital Privacy & Consent",
        "Discuss informed consent and respect for users in how personal data is collected and used; stay concise and neutral.",
    ),
    (
        "AI Ethics & Accountability",
        "Discuss one ethical risk of automated decision-making (e.g. bias, transparency) and what accountability could mean in practice.",
    ),
    (
        "Research Integrity",
        "Discuss why reproducibility and honest reporting matter in science; one or two sentences, in a constructive tone.",
    ),
    (
        "Healthcare Access & Equity",
        "Discuss trade-offs in fair allocation of healthcare resources; remain respectful and avoid stigmatizing any group.",
    ),
    (
        "Education & Inclusion",
        "Discuss barriers some learners face and one principle for more inclusive education; keep it practical and brief.",
    ),
    (
        "Civil Discourse & Democracy",
        "Discuss how public disagreement can stay productive without demeaning others; one or two grounded suggestions.",
    ),
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum AgentNodeKind {
    Manager,
    Worker,
    Evaluator,
    Researcher,
    Topic,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PanelTab {
    Overview,
    Agents,
    Ollama,
    Settings,
}

#[derive(Clone, Copy)]
enum EvaluatorAgentsPick {
    Unassigned,
    AllWorkers,
    Worker(usize),
}

impl AgentNodeKind {
    fn label(&self) -> &'static str {
        match self {
            AgentNodeKind::Manager => "Agent Manager",
            AgentNodeKind::Worker => "Agent Worker",
            AgentNodeKind::Evaluator => "Agent Evaluator",
            AgentNodeKind::Researcher => "Agent Researcher",
            AgentNodeKind::Topic => "Topic",
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

    /// Selected manager agent id (row model; no graph wires).
    manager_node: Option<usize>,

    /// Optional topic tool agent id.
    topic_node: Option<usize>,
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

    /// When true, evaluate on traffic from all workers (no specific pin); pin 1 stays empty.
    evaluate_all_workers: bool,

    worker_node: Option<usize>,
    manager_node: Option<usize>,
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

    worker_node: Option<usize>,
    manager_node: Option<usize>,
}

#[derive(Clone)]
struct NodeTopicData {
    name: String,
    global_id: String,

    analysis_mode: String,
    topic: String,
}

#[derive(Clone)]
enum NodePayload {
    Manager(NodeManagerData),
    Worker(NodeWorkerData),
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
                analysis_mode: TOPIC_PRESETS[0].0.to_string(),
                conversation_topic: TOPIC_PRESETS[0].1.to_string(),
                conversation_topic_source: "Own".to_string(),
                manager_node: None,
                topic_node: None,
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
                evaluate_all_workers: false,
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
                analysis_mode: TOPIC_PRESETS[0].0.to_string(),
                topic: TOPIC_PRESETS[0].1.to_string(),
            }),
        }
    }

    fn set_name(&mut self, name: String) {
        match &mut self.payload {
            NodePayload::Manager(m) => m.name = name,
            NodePayload::Worker(w) => w.name = name,
            NodePayload::Evaluator(e) => e.name = name,
            NodePayload::Researcher(r) => r.name = name,
            NodePayload::Topic(t) => t.name = name,
        }
    }
}

/// One row in the Agents list (stable `id` for manifests and conversation loops).
#[derive(Clone)]
pub(super) struct AgentRecord {
    pub id: usize,
    pub position: egui::Pos2,
    pub open: bool,
    pub data: NodeData,
}

pub(super) struct NodesPanelState {
    pub(super) next_agent_id: usize,
    pub(super) agents: Vec<AgentRecord>,
    selected_add_kind: AgentNodeKind,
    active_tab: PanelTab,
}

impl Default for NodesPanelState {
    fn default() -> Self {
        Self {
            next_agent_id: 0,
            agents: Vec::new(),
            selected_add_kind: AgentNodeKind::Worker,
            active_tab: PanelTab::Overview,
        }
    }
}

impl NodesPanelState {
    fn push_agent(&mut self, pos: egui::Pos2, data: NodeData) -> usize {
        let id = self.next_agent_id;
        self.next_agent_id += 1;
        self.agents.push(AgentRecord {
            id,
            position: pos,
            open: true,
            data,
        });
        id
    }

    fn insert_agent_with_id(&mut self, id: usize, pos: egui::Pos2, open: bool, data: NodeData) {
        self.agents.push(AgentRecord {
            id,
            position: pos,
            open,
            data,
        });
        if id + 1 > self.next_agent_id {
            self.next_agent_id = id + 1;
        }
    }

    fn remove_agent(&mut self, id: usize) {
        self.agents.retain(|a| a.id != id);
    }

    fn agent_by_id_mut(&mut self, id: usize) -> Option<&mut AgentRecord> {
        self.agents.iter_mut().find(|a| a.id == id)
    }
}

#[derive(Default)]
struct BasicNodeViewer;

impl BasicNodeViewer {
    fn numbered_name_for_kind(agents: &[AgentRecord], kind: AgentNodeKind) -> String {
        let idx = agents.iter().filter(|a| a.data.kind == kind).count() + 1;
        format!("{} {}", kind.label(), idx)
    }
}

impl BasicNodeViewer {
    fn show_body(&mut self, id: usize, ui: &mut egui::Ui, agents: &mut [AgentRecord]) {
        let Some(idx) = agents.iter().position(|a| a.id == id) else {
            return;
        };
        match agents[idx].data.kind {
            AgentNodeKind::Manager => {
                // Make manager node slightly wider than others.
                {
                    let manager_data = match &mut agents[idx].data.payload {
                        NodePayload::Manager(m) => m,
                        _ => unreachable!("kind mismatch"),
                    };
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.label("Name:");
                            ui.add(egui::TextEdit::singleline(&mut manager_data.name));
                        });

                        ui.separator();
                    });
                }
            }
            AgentNodeKind::Worker => {
                let mut managers: Vec<(usize, String)> = agents
                    .iter()
                    .filter_map(|a| {
                        if let NodePayload::Manager(m) = &a.data.payload {
                            Some((a.id, m.name.clone()))
                        } else {
                            None
                        }
                    })
                    .collect();
                managers.sort_by(|a, b| a.1.cmp(&b.1));

                let my_manager_node = match &agents[idx].data.payload {
                    NodePayload::Worker(w) => w.manager_node,
                    _ => None,
                };

                let mut pending_manager_pick: Option<Option<usize>> = None;
                {
                    let worker_data = match &mut agents[idx].data.payload {
                        NodePayload::Worker(w) => w,
                        _ => unreachable!("kind mismatch"),
                    };

                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.label("Manager:");
                            let selected_text = my_manager_node
                                .and_then(|mid| {
                                    managers
                                        .iter()
                                        .find(|(id, _)| *id == mid)
                                        .map(|(_, n)| n.clone())
                                })
                                .unwrap_or_else(|| "Unassigned".to_string());
                            egui::ComboBox::from_id_salt(ui.id().with(id).with("manager_pick"))
                                .selected_text(selected_text)
                                .show_ui(ui, |ui| {
                                    if ui
                                        .selectable_label(my_manager_node.is_none(), "Unassigned")
                                        .clicked()
                                    {
                                        pending_manager_pick = Some(None);
                                    }
                                    for &(mgr_id, ref mgr_name) in &managers {
                                        if ui
                                            .selectable_label(
                                                my_manager_node == Some(mgr_id),
                                                mgr_name.as_str(),
                                            )
                                            .clicked()
                                        {
                                            pending_manager_pick = Some(Some(mgr_id));
                                        }
                                    }
                                });
                        });

                        ui.vertical(|ui| {
                            ui.spacing_mut().item_spacing = egui::Vec2::new(5.0, 2.0);

                            ui.horizontal(|ui| {
                                ui.label("Name:");
                                ui.add(egui::TextEdit::singleline(&mut worker_data.name));
                            });

                            ui.horizontal(|ui| {
                                ui.label("Instruction:");
                                egui::ComboBox::from_id_salt(
                                    ui.id().with(id).with("instruction_mode"),
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
                            egui::ComboBox::from_id_salt(
                                ui.id().with(id).with("worker_topic_analysis_mode"),
                            )
                            .selected_text(if worker_data.analysis_mode.is_empty() {
                                "Select".to_string()
                            } else {
                                worker_data.analysis_mode.clone()
                            })
                            .show_ui(ui, |ui| {
                                for &(label, sentence) in TOPIC_PRESETS {
                                    if ui
                                        .selectable_label(worker_data.analysis_mode == label, label)
                                        .clicked()
                                    {
                                        worker_data.analysis_mode = label.to_string();
                                        worker_data.conversation_topic = sentence.to_string();
                                    }
                                }
                            });
                        });
                        ui.horizontal(|ui| {
                            ui.label("Topic:");
                            ui.add(egui::TextEdit::singleline(
                                &mut worker_data.conversation_topic,
                            ));
                        });

                        ui.separator();
                    });
                }

                if let Some(pick) = pending_manager_pick {
                    if let NodePayload::Worker(w) = &mut agents[idx].data.payload {
                        w.manager_node = pick;
                    }
                }
            }
            AgentNodeKind::Evaluator => {
                let mut managers: Vec<(usize, String)> = agents
                    .iter()
                    .filter_map(|a| {
                        if let NodePayload::Manager(m) = &a.data.payload {
                            Some((a.id, m.name.clone()))
                        } else {
                            None
                        }
                    })
                    .collect();
                managers.sort_by(|a, b| a.1.cmp(&b.1));

                let mut workers: Vec<(usize, String)> = agents
                    .iter()
                    .filter_map(|a| {
                        if let NodePayload::Worker(w) = &a.data.payload {
                            Some((a.id, w.name.clone()))
                        } else {
                            None
                        }
                    })
                    .collect();
                workers.sort_by(|a, b| a.1.cmp(&b.1));

                let (my_manager_node, my_eval_worker_pin) = match &agents[idx].data.payload {
                    NodePayload::Evaluator(e) => (e.manager_node, e.worker_node),
                    _ => (None, None),
                };

                let mut pending_manager_pick: Option<Option<usize>> = None;
                let mut pending_agents_pick: Option<EvaluatorAgentsPick> = None;

                {
                    let evaluator_data = match &mut agents[idx].data.payload {
                        NodePayload::Evaluator(e) => e,
                        _ => unreachable!("kind mismatch"),
                    };

                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.label("Manager:");
                            let selected_text = my_manager_node
                                .and_then(|mid| {
                                    managers
                                        .iter()
                                        .find(|(id, _)| *id == mid)
                                        .map(|(_, n)| n.clone())
                                })
                                .unwrap_or_else(|| "Unassigned".to_string());
                            egui::ComboBox::from_id_salt(
                                ui.id().with(id).with("eval_manager_pick"),
                            )
                            .selected_text(selected_text)
                            .show_ui(ui, |ui| {
                                if ui
                                    .selectable_label(my_manager_node.is_none(), "Unassigned")
                                    .clicked()
                                {
                                    pending_manager_pick = Some(None);
                                }
                                for &(mgr_id, ref mgr_name) in &managers {
                                    if ui
                                        .selectable_label(
                                            my_manager_node == Some(mgr_id),
                                            mgr_name.as_str(),
                                        )
                                        .clicked()
                                    {
                                        pending_manager_pick = Some(Some(mgr_id));
                                    }
                                }
                            });
                        });

                        ui.horizontal(|ui| {
                            ui.label("Agents to Evaluate:");
                            let selected_text = if evaluator_data.evaluate_all_workers {
                                "All Workers".to_string()
                            } else {
                                my_eval_worker_pin
                                    .and_then(|wid| {
                                        workers
                                            .iter()
                                            .find(|(id, _)| *id == wid)
                                            .map(|(_, n)| n.clone())
                                    })
                                    .unwrap_or_else(|| "Unassigned".to_string())
                            };
                            egui::ComboBox::from_id_salt(
                                ui.id().with(id).with("eval_agents_pick"),
                            )
                            .selected_text(selected_text)
                            .show_ui(ui, |ui| {
                                let unassigned_sel = !evaluator_data.evaluate_all_workers
                                    && my_eval_worker_pin.is_none();
                                if ui.selectable_label(unassigned_sel, "Unassigned").clicked() {
                                    pending_agents_pick = Some(EvaluatorAgentsPick::Unassigned);
                                }
                                let all_sel = evaluator_data.evaluate_all_workers;
                                if ui.selectable_label(all_sel, "All Workers").clicked() {
                                    pending_agents_pick = Some(EvaluatorAgentsPick::AllWorkers);
                                }
                                for &(worker_id, ref worker_name) in &workers {
                                    let sel = !evaluator_data.evaluate_all_workers
                                        && my_eval_worker_pin == Some(worker_id);
                                    if ui
                                        .selectable_label(sel, worker_name.as_str())
                                        .clicked()
                                    {
                                        pending_agents_pick =
                                            Some(EvaluatorAgentsPick::Worker(worker_id));
                                    }
                                }
                            });
                        });

                        ui.separator();

                        ui.horizontal(|ui| {
                            ui.label("Name:");
                            ui.add(egui::TextEdit::singleline(&mut evaluator_data.name));
                        });

                        ui.horizontal(|ui| {
                            ui.label("Analysis:");
                            egui::ComboBox::from_id_salt(
                                ui.id().with(id).with("eval_analysis_mode"),
                            )
                            .selected_text(
                                if evaluator_data.analysis_mode.is_empty() {
                                    "Select".to_string()
                                } else {
                                    evaluator_data.analysis_mode.clone()
                                },
                            )
                            .show_ui(ui, |ui| {
                                if ui
                                    .selectable_label(
                                        evaluator_data.analysis_mode == "Topic Extraction",
                                        "Topic Extraction",
                                    )
                                    .clicked()
                                {
                                    evaluator_data.analysis_mode =
                                        "Topic Extraction".to_string();
                                    evaluator_data.instruction = "Topic Extraction: extract the topic in 1 or 2 words. Identify what is the topic of the sentence being analysed.".to_string();
                                }
                                if ui
                                    .selectable_label(
                                        evaluator_data.analysis_mode == "Decision Analysis",
                                        "Decision Analysis",
                                    )
                                    .clicked()
                                {
                                    evaluator_data.analysis_mode =
                                        "Decision Analysis".to_string();
                                    evaluator_data.instruction = "Decision Analysis: extract a decision in 1 or 2 sentences about the agent in the message being analysed. Focus on the concrete decision and its intent.".to_string();
                                }
                                if ui
                                    .selectable_label(
                                        evaluator_data.analysis_mode
                                            == "Sentiment Classification",
                                        "Sentiment Classification",
                                    )
                                    .clicked()
                                {
                                    evaluator_data.analysis_mode =
                                        "Sentiment Classification".to_string();
                                    evaluator_data.instruction = "Sentiment Classification: extract the sentiment of the message being analysed and return one word that is the sentiment.".to_string();
                                }
                            });
                        });

                        ui.horizontal(|ui| {
                            ui.label("Instruction:");
                            ui.add(egui::TextEdit::singleline(&mut evaluator_data.instruction));
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
                                ui.add(
                                    egui::TextEdit::singleline(&mut evaluator_data.num_predict)
                                        .desired_width(80.0),
                                );
                            }
                        });

                        ui.separator();
                    });
                }

                if let Some(pick) = pending_manager_pick {
                    if let NodePayload::Evaluator(e) = &mut agents[idx].data.payload {
                        e.manager_node = pick;
                    }
                }
                if let Some(pick) = pending_agents_pick {
                    if let NodePayload::Evaluator(e) = &mut agents[idx].data.payload {
                        match pick {
                            EvaluatorAgentsPick::Unassigned => {
                                e.evaluate_all_workers = false;
                                e.worker_node = None;
                                e.active = false;
                            }
                            EvaluatorAgentsPick::AllWorkers => {
                                e.evaluate_all_workers = true;
                                e.worker_node = None;
                                e.active = true;
                            }
                            EvaluatorAgentsPick::Worker(wid) => {
                                e.evaluate_all_workers = false;
                                e.worker_node = Some(wid);
                                e.active = true;
                            }
                        }
                    }
                }
            }
            AgentNodeKind::Researcher => {
                let mut managers: Vec<(usize, String)> = agents
                    .iter()
                    .filter_map(|a| {
                        if let NodePayload::Manager(m) = &a.data.payload {
                            Some((a.id, m.name.clone()))
                        } else {
                            None
                        }
                    })
                    .collect();
                managers.sort_by(|a, b| a.1.cmp(&b.1));

                let mut workers: Vec<(usize, String)> = agents
                    .iter()
                    .filter_map(|a| {
                        if let NodePayload::Worker(w) = &a.data.payload {
                            Some((a.id, w.name.clone()))
                        } else {
                            None
                        }
                    })
                    .collect();
                workers.sort_by(|a, b| a.1.cmp(&b.1));

                let (my_manager_node, my_injection_worker) = match &agents[idx].data.payload {
                    NodePayload::Researcher(r) => (r.manager_node, r.worker_node),
                    _ => (None, None),
                };

                let mut pending_manager_pick: Option<Option<usize>> = None;
                let mut pending_injection_pick: Option<Option<usize>> = None;
                {
                    let researcher_data = match &mut agents[idx].data.payload {
                        NodePayload::Researcher(r) => r,
                        _ => unreachable!("kind mismatch"),
                    };

                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.label("Manager:");
                            let selected_text = my_manager_node
                                .and_then(|mid| {
                                    managers
                                        .iter()
                                        .find(|(id, _)| *id == mid)
                                        .map(|(_, n)| n.clone())
                                })
                                .unwrap_or_else(|| "Unassigned".to_string());
                            egui::ComboBox::from_id_salt(
                                ui.id().with(id).with("researcher_manager_pick"),
                            )
                            .selected_text(selected_text)
                            .show_ui(ui, |ui| {
                                if ui
                                    .selectable_label(my_manager_node.is_none(), "Unassigned")
                                    .clicked()
                                {
                                    pending_manager_pick = Some(None);
                                }
                                for &(mgr_id, ref mgr_name) in &managers {
                                    if ui
                                        .selectable_label(
                                            my_manager_node == Some(mgr_id),
                                            mgr_name.as_str(),
                                        )
                                        .clicked()
                                    {
                                        pending_manager_pick = Some(Some(mgr_id));
                                    }
                                }
                            });
                        });

                        ui.horizontal(|ui| {
                            ui.label("Injection:");
                            let selected_text = my_injection_worker
                                .and_then(|wid| {
                                    workers
                                        .iter()
                                        .find(|(id, _)| *id == wid)
                                        .map(|(_, n)| n.clone())
                                })
                                .unwrap_or_else(|| "Unassigned".to_string());
                            egui::ComboBox::from_id_salt(
                                ui.id().with(id).with("researcher_injection_pick"),
                            )
                            .selected_text(selected_text)
                            .show_ui(ui, |ui| {
                                if ui
                                    .selectable_label(
                                        my_injection_worker.is_none(),
                                        "Unassigned",
                                    )
                                    .clicked()
                                {
                                    pending_injection_pick = Some(None);
                                }
                                for &(worker_id, ref worker_name) in &workers {
                                    if ui
                                        .selectable_label(
                                            my_injection_worker == Some(worker_id),
                                            worker_name.as_str(),
                                        )
                                        .clicked()
                                    {
                                        pending_injection_pick = Some(Some(worker_id));
                                    }
                                }
                            });
                        });

                        ui.separator();

                        ui.horizontal(|ui| {
                            ui.label("Name:");
                            ui.add(egui::TextEdit::singleline(&mut researcher_data.name));
                        });

                        ui.horizontal(|ui| {
                            ui.label("Topics:");
                            egui::ComboBox::from_id_salt(
                                ui.id().with(id).with("research_topic_mode"),
                            )
                            .selected_text(
                                if researcher_data.topic_mode.is_empty() {
                                    "Select".to_string()
                                } else {
                                    researcher_data.topic_mode.clone()
                                },
                            )
                            .show_ui(ui, |ui| {
                                if ui
                                    .selectable_label(
                                        researcher_data.topic_mode == "Articles",
                                        "Articles",
                                    )
                                    .clicked()
                                {
                                    researcher_data.topic_mode = "Articles".to_string();
                                    researcher_data.instruction = "Generate article references connected to the message context. Prefer a mix of classic and recent pieces.".to_string();
                                }
                                if ui
                                    .selectable_label(
                                        researcher_data.topic_mode == "Movies",
                                        "Movies",
                                    )
                                    .clicked()
                                {
                                    researcher_data.topic_mode = "Movies".to_string();
                                    researcher_data.instruction = "Generate movie references connected to the message context. Prefer diverse genres and well-known titles.".to_string();
                                }
                                if ui
                                    .selectable_label(
                                        researcher_data.topic_mode == "Music",
                                        "Music",
                                    )
                                    .clicked()
                                {
                                    researcher_data.topic_mode = "Music".to_string();
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
                                .checkbox(&mut researcher_data.limit_token, "Limit Tokens:")
                                .changed()
                            {
                                if !researcher_data.limit_token {
                                    researcher_data.num_predict.clear();
                                }
                            }
                            if researcher_data.limit_token {
                                ui.label("num_predict:");
                                ui.add(
                                    egui::TextEdit::singleline(&mut researcher_data.num_predict)
                                        .desired_width(80.0),
                                );
                            }
                        });

                        ui.separator();
                    });
                }

                if let Some(pick) = pending_manager_pick {
                    if let NodePayload::Researcher(r) = &mut agents[idx].data.payload {
                        r.manager_node = pick;
                    }
                }
                if let Some(pick) = pending_injection_pick {
                    if let NodePayload::Researcher(r) = &mut agents[idx].data.payload {
                        r.worker_node = pick;
                        r.active = r.worker_node.is_some();
                    }
                }
            }
            AgentNodeKind::Topic => {
                {
                    let topic_data = match &mut agents[idx].data.payload {
                        NodePayload::Topic(t) => t,
                        _ => unreachable!("kind mismatch"),
                    };

                    ui.vertical(|ui| {
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("Name:");
                            ui.add(egui::TextEdit::singleline(&mut topic_data.name));
                        });
                        ui.separator();

                        // Topic preset + topic text (mirrors Agent Worker widgets).
                        ui.vertical(|ui| {
                            ui.horizontal(|ui| {
                                ui.label("Topic:");
                                egui::ComboBox::from_id_salt(
                                    ui.id().with(id).with("topic_analysis_mode"),
                                )
                                .selected_text(if topic_data.analysis_mode.is_empty() {
                                    "Select".to_string()
                                } else {
                                    topic_data.analysis_mode.clone()
                                })
                                .show_ui(ui, |ui| {
                                    for &(label, sentence) in TOPIC_PRESETS {
                                        if ui
                                            .selectable_label(
                                                topic_data.analysis_mode == label,
                                                label,
                                            )
                                            .clicked()
                                        {
                                            topic_data.analysis_mode = label.to_string();
                                            topic_data.topic = sentence.to_string();
                                        }
                                    }
                                });
                            });

                            ui.horizontal(|ui| {
                                ui.label("Topic:");
                                ui.add(egui::TextEdit::singleline(&mut topic_data.topic));
                            });
                        });

                        ui.separator();
                    });
                }
            }
        }
    }
}
/// Refresh evaluator/researcher `active` from row selections (combiners may skip the row body when collapsed).
fn sync_evaluator_researcher_activity(agents: &mut [AgentRecord]) {
    for r in agents.iter_mut() {
        match &mut r.data.payload {
            NodePayload::Evaluator(e) => {
                e.active = e.evaluate_all_workers || e.worker_node.is_some();
            }
            NodePayload::Researcher(res) => {
                res.active = res.worker_node.is_some();
            }
            _ => {}
        }
    }
}

fn manifest_edges_from_agents(agents: &[AgentRecord]) -> Vec<ManifestEdge> {
    let mut edges = Vec::new();
    for r in agents {
        let nid = r.id;
        match &r.data.payload {
            NodePayload::Worker(w) => {
                if let Some(m) = w.manager_node {
                    edges.push(ManifestEdge {
                        from_node_id: m,
                        from_output_pin: 0,
                        to_node_id: nid,
                        to_input_pin: 0,
                    });
                }
                if let Some(t) = w.topic_node {
                    edges.push(ManifestEdge {
                        from_node_id: t,
                        from_output_pin: 0,
                        to_node_id: nid,
                        to_input_pin: 1,
                    });
                }
            }
            NodePayload::Evaluator(e) => {
                if let Some(m) = e.manager_node {
                    edges.push(ManifestEdge {
                        from_node_id: m,
                        from_output_pin: 0,
                        to_node_id: nid,
                        to_input_pin: 0,
                    });
                }
                if let Some(wid) = e.worker_node {
                    if !e.evaluate_all_workers {
                        edges.push(ManifestEdge {
                            from_node_id: wid,
                            from_output_pin: 0,
                            to_node_id: nid,
                            to_input_pin: 1,
                        });
                    }
                }
            }
            NodePayload::Researcher(res) => {
                if let Some(m) = res.manager_node {
                    edges.push(ManifestEdge {
                        from_node_id: m,
                        from_output_pin: 0,
                        to_node_id: nid,
                        to_input_pin: 0,
                    });
                }
                if let Some(wid) = res.worker_node {
                    edges.push(ManifestEdge {
                        from_node_id: wid,
                        from_output_pin: 0,
                        to_node_id: nid,
                        to_input_pin: 1,
                    });
                }
            }
            _ => {}
        }
    }
    edges.sort_by_key(|e| {
        (
            e.from_node_id,
            e.from_output_pin,
            e.to_node_id,
            e.to_input_pin,
        )
    });
    edges
}

#[derive(serde::Serialize)]
struct RunPlayPlanJson {
    conversations: Vec<PlayConversationPairJson>,
    managers: Vec<PlayManagerJson>,
    evaluators: Vec<PlayEvaluatorInPlayJson>,
    researchers: Vec<PlayResearcherInPlayJson>,
}

#[derive(serde::Serialize)]
struct PlayConversationPairJson {
    loop_key_node_id: usize,
    agent_a: PlayWorkerInPlayJson,
    agent_b: PlayWorkerInPlayJson,
    /// True when only one eligible worker exists (paired with itself for the loop).
    solo: bool,
}

#[derive(serde::Serialize)]
struct PlayWorkerInPlayJson {
    node_id: usize,
    name: String,
    global_id: String,
    conversation_topic: String,
    conversation_topic_source: String,
}

#[derive(serde::Serialize)]
struct PlayManagerJson {
    node_id: usize,
    name: String,
    global_id: String,
}

#[derive(serde::Serialize)]
struct PlayEvaluatorInPlayJson {
    node_id: usize,
    name: String,
    global_id: String,
    analysis_mode: String,
    evaluate_all_workers: bool,
}

#[derive(serde::Serialize)]
struct PlayResearcherInPlayJson {
    node_id: usize,
    name: String,
    global_id: String,
    topic_mode: String,
}

fn build_conversation_sidecar_from_agents(
    agents: &[AgentRecord],
) -> crate::agent_conversation_loop::ConversationSidecarConfig {
    use crate::agent_conversation_loop::{
        ConversationSidecarConfig, SidecarEvaluator, SidecarResearcher,
    };
    let mut evaluators = Vec::new();
    let mut researchers = Vec::new();
    for r in agents {
        match &r.data.payload {
            NodePayload::Evaluator(e) if e.active => {
                evaluators.push(SidecarEvaluator {
                    global_id: e.global_id.clone(),
                    instruction: e.instruction.clone(),
                    analysis_mode: e.analysis_mode.clone(),
                    limit_token: e.limit_token,
                    num_predict: e.num_predict.clone(),
                });
            }
            NodePayload::Researcher(res) if res.active => {
                researchers.push(SidecarResearcher {
                    global_id: res.global_id.clone(),
                    topic_mode: res.topic_mode.clone(),
                    instruction: res.instruction.clone(),
                    limit_token: res.limit_token,
                    num_predict: res.num_predict.clone(),
                });
            }
            _ => {}
        }
    }
    ConversationSidecarConfig {
        evaluators,
        researchers,
    }
}

fn collect_run_play_plan_from_agents(
    agents: &[AgentRecord],
    conversations: Vec<PlayConversationPairJson>,
) -> RunPlayPlanJson {
    let mut managers = Vec::new();
    let mut evaluators = Vec::new();
    let mut researchers = Vec::new();
    for r in agents {
        let id = r.id;
        match &r.data.payload {
            NodePayload::Manager(m) => managers.push(PlayManagerJson {
                node_id: id,
                name: m.name.clone(),
                global_id: m.global_id.clone(),
            }),
            NodePayload::Evaluator(e) if e.active => {
                evaluators.push(PlayEvaluatorInPlayJson {
                    node_id: id,
                    name: e.name.clone(),
                    global_id: e.global_id.clone(),
                    analysis_mode: e.analysis_mode.clone(),
                    evaluate_all_workers: e.evaluate_all_workers,
                });
            }
            NodePayload::Researcher(res) if res.active => {
                researchers.push(PlayResearcherInPlayJson {
                    node_id: id,
                    name: res.name.clone(),
                    global_id: res.global_id.clone(),
                    topic_mode: res.topic_mode.clone(),
                });
            }
            _ => {}
        }
    }
    managers.sort_by_key(|m| m.node_id);
    evaluators.sort_by_key(|e| e.node_id);
    researchers.sort_by_key(|r| r.node_id);
    RunPlayPlanJson {
        conversations,
        managers,
        evaluators,
        researchers,
    }
}

impl AMSAgents {
    fn selected_model_option(&self) -> Option<String> {
        if self.selected_ollama_model.trim().is_empty() {
            None
        } else {
            Some(self.selected_ollama_model.clone())
        }
    }

    fn capture_runtime_settings(&self) -> RunRuntimeSettings {
        RunRuntimeSettings {
            selected_model: self.selected_model_option(),
            http_endpoint: self.http_endpoint.clone(),
            ollama_host: self.ollama_host.clone(),
            turn_delay_secs: self.conversation_turn_delay_secs,
            history_size: self.conversation_history_size,
            read_only_replay: self.read_only_replay_mode,
        }
    }

    fn capture_graph_snapshot(&self) -> GraphSnapshot {
        let mut nodes: Vec<ManifestNode> = self
            .nodes_panel
            .agents
            .iter()
            .map(|rec| {
                let node = &rec.data;
                let (kind, config) = match &node.payload {
                    NodePayload::Manager(m) => (
                        "manager".to_string(),
                        serde_json::json!({
                            "name": m.name,
                            "global_id": m.global_id,
                        }),
                    ),
                    NodePayload::Worker(w) => (
                        "worker".to_string(),
                        serde_json::json!({
                            "name": w.name,
                            "global_id": w.global_id,
                            "instruction_mode": w.instruction_mode,
                            "instruction": w.instruction,
                            "analysis_mode": w.analysis_mode,
                            "conversation_topic": w.conversation_topic,
                            "conversation_topic_source": w.conversation_topic_source,
                        }),
                    ),
                    NodePayload::Evaluator(e) => (
                        "evaluator".to_string(),
                        serde_json::json!({
                            "name": e.name,
                            "global_id": e.global_id,
                            "analysis_mode": e.analysis_mode,
                            "instruction": e.instruction,
                            "limit_token": e.limit_token,
                            "num_predict": e.num_predict,
                            "active": e.active,
                            "evaluate_all_workers": e.evaluate_all_workers,
                        }),
                    ),
                    NodePayload::Researcher(r) => (
                        "researcher".to_string(),
                        serde_json::json!({
                            "name": r.name,
                            "global_id": r.global_id,
                            "topic_mode": r.topic_mode,
                            "instruction": r.instruction,
                            "limit_token": r.limit_token,
                            "num_predict": r.num_predict,
                            "active": r.active,
                        }),
                    ),
                    NodePayload::Topic(t) => (
                        "topic".to_string(),
                        serde_json::json!({
                            "name": t.name,
                            "global_id": t.global_id,
                            "analysis_mode": t.analysis_mode,
                            "topic": t.topic,
                        }),
                    ),
                };
                ManifestNode {
                    node_id: rec.id,
                    kind,
                    label: node.label.clone(),
                    pos_x: rec.position.x,
                    pos_y: rec.position.y,
                    open: rec.open,
                    config,
                }
            })
            .collect();
        nodes.sort_by_key(|n| n.node_id);

        let edges = manifest_edges_from_agents(&self.nodes_panel.agents);
        GraphSnapshot { nodes, edges }
    }

    fn build_run_manifest(
        &self,
        experiment_id_override: Option<String>,
        read_only_replay: bool,
    ) -> anyhow::Result<RunManifest> {
        let mut runtime = self.capture_runtime_settings();
        runtime.read_only_replay = read_only_replay;
        let graph = self.capture_graph_snapshot();
        let graph_signature = canonical_graph_signature(&runtime, &graph)?;
        let experiment_id =
            experiment_id_override.unwrap_or_else(|| derive_experiment_id(&graph_signature));
        let run_id = new_run_id();

        Ok(RunManifest {
            manifest_version: MANIFEST_VERSION.to_string(),
            app_name: APP_NAME.to_string(),
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            created_at: now_rfc3339_utc(),
            experiment_id,
            run_id,
            graph_signature,
            runtime,
            graph,
        })
    }

    fn persist_active_manifest(&mut self, manifest: RunManifest) -> anyhow::Result<PathBuf> {
        let path = write_manifest(&runs_root(), &manifest)?;
        self.current_run_context = Some(RunContext {
            manifest_version: manifest.manifest_version.clone(),
            experiment_id: manifest.experiment_id.clone(),
            run_id: manifest.run_id.clone(),
        });
        self.current_manifest = Some(manifest);
        self.manifest_status_message = format!("Manifest saved: {}", path.display());
        Ok(path)
    }

    pub(super) fn export_manifest_to_path(&mut self, path: PathBuf) -> anyhow::Result<()> {
        let manifest = if let Some(existing) = &self.current_manifest {
            existing.clone()
        } else {
            self.build_run_manifest(None, self.read_only_replay_mode)?
        };
        export_manifest_to(&manifest, &path)?;
        self.current_manifest = Some(manifest);
        self.manifest_status_message = format!("Manifest exported: {}", path.display());
        Ok(())
    }

    fn clear_graph(&mut self) {
        self.nodes_panel.agents.clear();
        self.nodes_panel.next_agent_id = 0;
    }

    /// Rebuilds graph and runtime fields from a manifest (stops graph, clears agents).
    fn apply_manifest_graph_and_runtime(&mut self, manifest: &RunManifest) -> anyhow::Result<()> {
        self.stop_graph();
        self.clear_graph();

        let mut nodes_sorted = manifest.graph.nodes.clone();
        nodes_sorted.sort_by_key(|n| n.node_id);

        for node in nodes_sorted {
            let kind = node.kind.as_str();
            let pos = egui::pos2(node.pos_x, node.pos_y);
            let mut node_data = match kind {
                "manager" => NodeData::new_manager(),
                "worker" => NodeData::new_worker(),
                "evaluator" => NodeData::new_evaluator(),
                "researcher" => NodeData::new_researcher(),
                "topic" => NodeData::new_topic(),
                _ => continue,
            };
            node_data.label = node.label.clone();
            match (&mut node_data.payload, kind) {
                (NodePayload::Manager(m), "manager") => {
                    m.name = node.config["name"].as_str().unwrap_or(&m.name).to_string();
                    m.global_id = node.config["global_id"]
                        .as_str()
                        .unwrap_or(&m.global_id)
                        .to_string();
                }
                (NodePayload::Worker(w), "worker") => {
                    w.name = node.config["name"].as_str().unwrap_or(&w.name).to_string();
                    w.global_id = node.config["global_id"]
                        .as_str()
                        .unwrap_or(&w.global_id)
                        .to_string();
                    w.instruction_mode = node.config["instruction_mode"]
                        .as_str()
                        .unwrap_or(&w.instruction_mode)
                        .to_string();
                    w.instruction = node.config["instruction"]
                        .as_str()
                        .unwrap_or(&w.instruction)
                        .to_string();
                    w.analysis_mode = node.config["analysis_mode"]
                        .as_str()
                        .unwrap_or(&w.analysis_mode)
                        .to_string();
                    w.conversation_topic = node.config["conversation_topic"]
                        .as_str()
                        .unwrap_or(&w.conversation_topic)
                        .to_string();
                    w.conversation_topic_source = node.config["conversation_topic_source"]
                        .as_str()
                        .unwrap_or(&w.conversation_topic_source)
                        .to_string();
                }
                (NodePayload::Evaluator(e), "evaluator") => {
                    e.name = node.config["name"].as_str().unwrap_or(&e.name).to_string();
                    e.global_id = node.config["global_id"]
                        .as_str()
                        .unwrap_or(&e.global_id)
                        .to_string();
                    e.analysis_mode = node.config["analysis_mode"]
                        .as_str()
                        .unwrap_or(&e.analysis_mode)
                        .to_string();
                    e.instruction = node.config["instruction"]
                        .as_str()
                        .unwrap_or(&e.instruction)
                        .to_string();
                    e.limit_token = node.config["limit_token"]
                        .as_bool()
                        .unwrap_or(e.limit_token);
                    e.num_predict = node.config["num_predict"]
                        .as_str()
                        .unwrap_or(&e.num_predict)
                        .to_string();
                    e.active = node.config["active"].as_bool().unwrap_or(e.active);
                    e.evaluate_all_workers = node.config["evaluate_all_workers"]
                        .as_bool()
                        .unwrap_or(e.evaluate_all_workers);
                }
                (NodePayload::Researcher(r), "researcher") => {
                    r.name = node.config["name"].as_str().unwrap_or(&r.name).to_string();
                    r.global_id = node.config["global_id"]
                        .as_str()
                        .unwrap_or(&r.global_id)
                        .to_string();
                    r.topic_mode = node.config["topic_mode"]
                        .as_str()
                        .unwrap_or(&r.topic_mode)
                        .to_string();
                    r.instruction = node.config["instruction"]
                        .as_str()
                        .unwrap_or(&r.instruction)
                        .to_string();
                    r.limit_token = node.config["limit_token"]
                        .as_bool()
                        .unwrap_or(r.limit_token);
                    r.num_predict = node.config["num_predict"]
                        .as_str()
                        .unwrap_or(&r.num_predict)
                        .to_string();
                    r.active = node.config["active"].as_bool().unwrap_or(r.active);
                }
                (NodePayload::Topic(t), "topic") => {
                    t.name = node.config["name"].as_str().unwrap_or(&t.name).to_string();
                    t.global_id = node.config["global_id"]
                        .as_str()
                        .unwrap_or(&t.global_id)
                        .to_string();
                    t.analysis_mode = node.config["analysis_mode"]
                        .as_str()
                        .unwrap_or(&t.analysis_mode)
                        .to_string();
                    t.topic = node.config["topic"]
                        .as_str()
                        .unwrap_or(&t.topic)
                        .to_string();
                }
                _ => {}
            }

            self.nodes_panel
                .insert_agent_with_id(node.node_id, pos, node.open, node_data);
        }

        for edge in &manifest.graph.edges {
            let from_id = edge.from_node_id;
            let to_id = edge.to_node_id;
            let mut to_input = edge.to_input_pin;
            if let Some(to_n) = manifest
                .graph
                .nodes
                .iter()
                .find(|n| n.node_id == edge.to_node_id)
            {
                let kind = to_n.kind.as_str();
                if (kind == "researcher" || kind == "evaluator") && to_input == 0 {
                    if let Some(from_n) = manifest
                        .graph
                        .nodes
                        .iter()
                        .find(|n| n.node_id == edge.from_node_id)
                    {
                        if from_n.kind.as_str() == "worker" {
                            to_input = 1;
                        }
                    }
                }
            }
            let Some(rec) = self.nodes_panel.agent_by_id_mut(to_id) else {
                continue;
            };
            match &mut rec.data.payload {
                NodePayload::Worker(w) => {
                    if to_input == 0 {
                        w.manager_node = Some(from_id);
                    } else if to_input == 1 {
                        w.topic_node = Some(from_id);
                    }
                }
                NodePayload::Evaluator(e) => {
                    if to_input == 0 {
                        e.manager_node = Some(from_id);
                    } else if to_input == 1 {
                        e.worker_node = Some(from_id);
                    }
                }
                NodePayload::Researcher(r) => {
                    if to_input == 0 {
                        r.manager_node = Some(from_id);
                    } else if to_input == 1 {
                        r.worker_node = Some(from_id);
                    }
                }
                _ => {}
            }
        }

        for r in self.nodes_panel.agents.iter_mut() {
            if let NodePayload::Evaluator(e) = &mut r.data.payload {
                if e.evaluate_all_workers {
                    e.worker_node = None;
                }
            }
        }
        sync_evaluator_researcher_activity(&mut self.nodes_panel.agents);

        self.selected_ollama_model = manifest.runtime.selected_model.clone().unwrap_or_default();
        self.http_endpoint = manifest.runtime.http_endpoint.clone();
        self.ollama_host = manifest.runtime.ollama_host.clone();
        self.conversation_turn_delay_secs = manifest.runtime.turn_delay_secs;
        self.conversation_history_size = manifest.runtime.history_size;

        Ok(())
    }

    pub(super) fn load_manifest_from_path(&mut self, path: PathBuf) -> anyhow::Result<()> {
        let manifest = read_manifest(&path)?;
        self.apply_manifest_graph_and_runtime(&manifest)?;
        self.read_only_replay_mode = true;
        self.current_run_context = Some(RunContext {
            manifest_version: manifest.manifest_version.clone(),
            experiment_id: manifest.experiment_id.clone(),
            run_id: manifest.run_id.clone(),
        });
        self.current_manifest = Some(manifest);
        self.manifest_status_message = format!("Loaded replay manifest: {}", path.display());
        Ok(())
    }

    pub(super) fn save_agents_workspace_to_path(&mut self, path: PathBuf) -> anyhow::Result<()> {
        let manifest = self.build_run_manifest(None, false)?;
        export_manifest_to(&manifest, &path)?;
        self.current_manifest = Some(manifest);
        self.manifest_status_message = format!("Saved workspace: {}", path.display());
        Ok(())
    }

    pub(super) fn load_agents_workspace_from_path(&mut self, path: PathBuf) -> anyhow::Result<()> {
        let manifest = read_manifest(&path)?;
        self.apply_manifest_graph_and_runtime(&manifest)?;
        self.read_only_replay_mode = false;
        self.current_run_context = None;
        self.current_manifest = Some(manifest);
        self.manifest_status_message = format!("Loaded workspace: {}", path.display());
        Ok(())
    }

    pub(super) fn run_from_manifest_path(&mut self, path: PathBuf) -> anyhow::Result<()> {
        self.load_manifest_from_path(path)?;
        let _ = self.run_graph();
        Ok(())
    }

    /// Writes `manifest.json`, `events.jsonl`, and `summary.json` from the current run into a zip.
    pub(super) fn download_run_bundle_to_path(&mut self, zip_path: PathBuf) -> anyhow::Result<()> {
        let ctx = self.current_run_context.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "no active run context; start the graph or load a manifest with a run id"
            )
        })?;
        let run_dir =
            crate::reproducibility::run_dir(&runs_root(), &ctx.experiment_id, &ctx.run_id);
        if !run_dir.is_dir() {
            anyhow::bail!("run directory not found: {}", run_dir.display());
        }
        crate::event_ledger::write_run_bundle_zip(&run_dir, &zip_path)?;
        self.manifest_status_message = format!("Run bundle written: {}", zip_path.display());
        Ok(())
    }

    fn stop_graph(&mut self) {
        self.ollama_run_epoch.fetch_add(1, Ordering::SeqCst);
        for (_, flag, _) in &self.conversation_loop_handles {
            *flag.lock().unwrap() = false;
        }
        self.conversation_loop_handles.clear();
        if let Some(ledger) = self.event_ledger.take() {
            let _ = ledger.try_finalize_run_stopped("graph_stopped");
        }
        self.conversation_graph_running
            .store(false, Ordering::Release);
        *self.last_message_in_chat.lock().unwrap() = None;
        self.conversation_message_events.lock().unwrap().clear();
        self.evaluator_event_queues.lock().unwrap().clear();
        self.researcher_event_queues.lock().unwrap().clear();
        self.last_evaluated_message_by_evaluator
            .lock()
            .unwrap()
            .clear();
        self.last_researched_message_by_researcher
            .lock()
            .unwrap()
            .clear();
    }

    /// Returns `true` if the manifest was saved and conversation loops were (re)scheduled.
    fn run_graph(&mut self) -> bool {
        // Bulletproof behavior: re-run means stop existing graph processes first.
        self.stop_graph();
        self.last_evaluated_message_by_evaluator
            .lock()
            .unwrap()
            .clear();
        self.last_researched_message_by_researcher
            .lock()
            .unwrap()
            .clear();
        self.evaluator_event_queues.lock().unwrap().clear();
        self.researcher_event_queues.lock().unwrap().clear();
        self.evaluator_inflight_nodes.lock().unwrap().clear();
        self.researcher_inflight_nodes.lock().unwrap().clear();

        let experiment_id_override = if self.read_only_replay_mode {
            self.current_manifest
                .as_ref()
                .map(|m| m.experiment_id.clone())
        } else {
            None
        };
        let manifest =
            match self.build_run_manifest(experiment_id_override, self.read_only_replay_mode) {
                Ok(m) => m,
                Err(e) => {
                    self.manifest_status_message = format!("Manifest build failed: {e}");
                    eprintln!("[Run Graph] Manifest build failed: {e}");
                    return false;
                }
            };
        let manifest_path = match self.persist_active_manifest(manifest) {
            Ok(p) => p,
            Err(e) => {
                self.manifest_status_message = format!("Manifest save failed: {e}");
                eprintln!("[Run Graph] Manifest save failed: {e}");
                return false;
            }
        };

        if let Some(ctx) = self.current_run_context.as_ref() {
            let run_dir = manifest_path
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| runs_root().join(&ctx.experiment_id).join(&ctx.run_id));
            match EventLedger::open(run_dir, ctx.experiment_id.clone(), ctx.run_id.clone()) {
                Ok(ledger) => {
                    let arc = Arc::new(ledger);
                    if let Err(e) = arc.append_system_run_started(&manifest_path) {
                        self.manifest_status_message = format!("Ledger start failed: {e}");
                        eprintln!("[Run Graph] Ledger start failed: {e}");
                    } else {
                        self.event_ledger = Some(arc);
                    }
                }
                Err(e) => {
                    self.manifest_status_message = format!("Ledger open failed: {e}");
                    eprintln!("[Run Graph] Ledger open failed: {e}");
                }
            }
        }

        sync_evaluator_researcher_activity(&mut self.nodes_panel.agents);
        let sidecars = std::sync::Arc::new(build_conversation_sidecar_from_agents(
            &self.nodes_panel.agents,
        ));

        // Workers with a non-empty topic, stable order for pairing (row order by id).
        struct EligibleWorker {
            id: usize,
            name: String,
            instruction: String,
            topic: String,
            topic_source: String,
        }

        let mut eligible: Vec<EligibleWorker> = self
            .nodes_panel
            .agents
            .iter()
            .filter_map(|r| {
                if let NodePayload::Worker(w) = &r.data.payload {
                    if !w.conversation_topic.trim().is_empty() {
                        return Some(EligibleWorker {
                            id: r.id,
                            name: w.name.clone(),
                            instruction: w.instruction.clone(),
                            topic: w.conversation_topic.clone(),
                            topic_source: w.conversation_topic_source.clone(),
                        });
                    }
                }
                None
            })
            .collect();
        eligible.sort_by_key(|w| w.id);

        if eligible.is_empty() {
            let play_plan = collect_run_play_plan_from_agents(&self.nodes_panel.agents, vec![]);
            match serde_json::to_string_pretty(&play_plan) {
                Ok(json) => println!("[Run Graph] play plan:\n{}", json),
                Err(e) => eprintln!("[Run Graph] failed to serialize play plan: {e}"),
            }
            if let Some(ref l) = self.event_ledger {
                let _ = l.try_finalize_run_stopped("no_eligible_conversation_workers");
            }
            return true;
        }

        let n_conversation_loops = (eligible.len() + 1) / 2;
        self.conversation_run_generation
            .fetch_add(1, Ordering::SeqCst);
        let run_generation = self.conversation_run_generation.load(Ordering::SeqCst);
        let loops_remaining = Arc::new(AtomicUsize::new(n_conversation_loops));
        let gen_counter = self.conversation_run_generation.clone();
        let graph_running_flag = self.conversation_graph_running.clone();

        let mut conversations_plan = Vec::new();
        let mut i = 0;
        while i < eligible.len() {
            if i + 1 < eligible.len() {
                let a = &eligible[i];
                let b = &eligible[i + 1];
                let gid_a = self
                    .nodes_panel
                    .agents
                    .iter()
                    .find(|r| r.id == a.id)
                    .and_then(|r| {
                        if let NodePayload::Worker(w) = &r.data.payload {
                            Some(w.global_id.clone())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_default();
                let gid_b = self
                    .nodes_panel
                    .agents
                    .iter()
                    .find(|r| r.id == b.id)
                    .and_then(|r| {
                        if let NodePayload::Worker(w) = &r.data.payload {
                            Some(w.global_id.clone())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_default();
                conversations_plan.push(PlayConversationPairJson {
                    loop_key_node_id: a.id,
                    agent_a: PlayWorkerInPlayJson {
                        node_id: a.id,
                        name: a.name.clone(),
                        global_id: gid_a,
                        conversation_topic: a.topic.clone(),
                        conversation_topic_source: a.topic_source.clone(),
                    },
                    agent_b: PlayWorkerInPlayJson {
                        node_id: b.id,
                        name: b.name.clone(),
                        global_id: gid_b,
                        conversation_topic: b.topic.clone(),
                        conversation_topic_source: b.topic_source.clone(),
                    },
                    solo: false,
                });
                self.start_conversation_from_node_worker_resolved(
                    sidecars.clone(),
                    run_generation,
                    gen_counter.clone(),
                    loops_remaining.clone(),
                    graph_running_flag.clone(),
                    a.id,
                    a.id,
                    a.name.clone(),
                    a.instruction.clone(),
                    a.topic.clone(),
                    a.topic_source.clone(),
                    b.id,
                    b.name.clone(),
                    b.instruction.clone(),
                    b.topic.clone(),
                    b.topic_source.clone(),
                );
                i += 2;
            } else {
                let a = &eligible[i];
                let gid = self
                    .nodes_panel
                    .agents
                    .iter()
                    .find(|r| r.id == a.id)
                    .and_then(|r| {
                        if let NodePayload::Worker(w) = &r.data.payload {
                            Some(w.global_id.clone())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_default();
                conversations_plan.push(PlayConversationPairJson {
                    loop_key_node_id: a.id,
                    agent_a: PlayWorkerInPlayJson {
                        node_id: a.id,
                        name: a.name.clone(),
                        global_id: gid.clone(),
                        conversation_topic: a.topic.clone(),
                        conversation_topic_source: a.topic_source.clone(),
                    },
                    agent_b: PlayWorkerInPlayJson {
                        node_id: a.id,
                        name: a.name.clone(),
                        global_id: gid,
                        conversation_topic: a.topic.clone(),
                        conversation_topic_source: a.topic_source.clone(),
                    },
                    solo: true,
                });
                self.start_conversation_from_node_worker_resolved(
                    sidecars.clone(),
                    run_generation,
                    gen_counter.clone(),
                    loops_remaining.clone(),
                    graph_running_flag.clone(),
                    a.id,
                    a.id,
                    a.name.clone(),
                    a.instruction.clone(),
                    a.topic.clone(),
                    a.topic_source.clone(),
                    a.id,
                    a.name.clone(),
                    a.instruction.clone(),
                    a.topic.clone(),
                    a.topic_source.clone(),
                );
                i += 1;
            }
        }

        let play_plan =
            collect_run_play_plan_from_agents(&self.nodes_panel.agents, conversations_plan);
        match serde_json::to_string_pretty(&play_plan) {
            Ok(json) => println!("[Run Graph] play plan:\n{}", json),
            Err(e) => eprintln!("[Run Graph] failed to serialize play plan: {e}"),
        }

        self.conversation_graph_running
            .store(true, Ordering::Release);
        true
    }

    /// Keys the async loop by `loop_key_node_id` (first worker in each pair). Conversation output nodes were removed; pairing is automatic from eligible workers.
    fn start_conversation_from_node_worker_resolved(
        &mut self,
        sidecars: Arc<crate::agent_conversation_loop::ConversationSidecarConfig>,
        run_generation: u64,
        run_generation_counter: Arc<AtomicU64>,
        loops_remaining_in_run: Arc<AtomicUsize>,
        conversation_graph_running_flag: Arc<AtomicBool>,
        loop_key_node_id: usize,
        agent_a_node_id: usize,
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
        let agent_a_id = agent_a_node_id;
        let active_flag = Arc::new(Mutex::new(true));
        let flag_clone = active_flag.clone();
        let endpoint = self.http_endpoint.clone();
        let ollama_host = self.ollama_host.clone();
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
        let run_context = self.current_run_context.clone();
        let message_event_source_id = loop_key_node_id;
        let ollama_epoch = self.ollama_run_epoch.clone();
        let ollama_caught = self.ollama_run_epoch.load(Ordering::SeqCst);
        let ollama_stop_epoch = Some((ollama_epoch, ollama_caught));

        let agent_a_global_id = self
            .nodes_panel
            .agents
            .iter()
            .find(|r| r.id == agent_a_id)
            .and_then(|r| {
                if let NodePayload::Worker(w) = &r.data.payload {
                    Some(w.global_id.clone())
                } else {
                    None
                }
            })
            .unwrap_or_default();
        let agent_b_global_id = self
            .nodes_panel
            .agents
            .iter()
            .find(|r| r.id == agent_b_id)
            .and_then(|r| {
                if let NodePayload::Worker(w) = &r.data.payload {
                    Some(w.global_id.clone())
                } else {
                    None
                }
            })
            .unwrap_or_default();
        let ledger = self.event_ledger.clone();

        let loop_handle = handle.spawn(async move {
            crate::agent_conversation_loop::start_conversation_loop(
                message_event_source_id,
                ollama_stop_epoch,
                sidecars,
                agent_a_id,
                agent_a_name,
                agent_a_instruction,
                agent_a_topic,
                agent_a_topic_source,
                agent_a_global_id,
                agent_b_id,
                agent_b_name,
                agent_b_instruction,
                agent_b_topic,
                agent_b_topic_source,
                agent_b_global_id,
                ollama_host,
                endpoint,
                flag_clone,
                last_msg,
                message_events,
                selected_model,
                history_size,
                turn_delay_secs,
                run_context,
                run_generation,
                run_generation_counter,
                loops_remaining_in_run,
                conversation_graph_running_flag,
                ledger,
            )
            .await;
        });

        self.conversation_loop_handles
            .push((loop_key_node_id, active_flag, loop_handle));
    }

    pub(super) fn render_nodes_panel(&mut self, ui: &mut egui::Ui) {
        let panel_border_color = ui.visuals().widgets.noninteractive.bg_stroke.color;
        let nodes_panel = egui::Frame::default()
            .fill(ui.visuals().panel_fill)
            .stroke(egui::Stroke::new(1.0, panel_border_color))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::same(6));

        let panel_height = ui.available_height().max(120.0);
        let mut viewer = BasicNodeViewer;

        ui.allocate_ui_with_layout(
            egui::vec2(ui.available_width(), panel_height),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                nodes_panel.show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if ui
                            .selectable_label(
                                self.nodes_panel.active_tab == PanelTab::Overview,
                                "Overview",
                            )
                            .clicked()
                        {
                            self.nodes_panel.active_tab = PanelTab::Overview;
                        }
                        if ui
                            .selectable_label(
                                self.nodes_panel.active_tab == PanelTab::Agents,
                                "Agents",
                            )
                            .clicked()
                        {
                            self.nodes_panel.active_tab = PanelTab::Agents;
                        }
                        if ui
                            .selectable_label(
                                self.nodes_panel.active_tab == PanelTab::Ollama,
                                "Ollama",
                            )
                            .clicked()
                        {
                            self.nodes_panel.active_tab = PanelTab::Ollama;
                        }
                        if ui
                            .selectable_label(
                                self.nodes_panel.active_tab == PanelTab::Settings,
                                "Settings",
                            )
                            .clicked()
                        {
                            self.nodes_panel.active_tab = PanelTab::Settings;
                        }
                    });
                    ui.separator();

                    if self.nodes_panel.active_tab == PanelTab::Overview {
                        ui.label("Lorem ipsum");
                        return;
                    }
                    if self.nodes_panel.active_tab == PanelTab::Ollama {
                        let ctx = ui.ctx().clone();
                        self.render_ollama_settings_widgets(ui, &ctx);
                        return;
                    }
                    if self.nodes_panel.active_tab == PanelTab::Settings {
                        self.render_reproducibility_settings_widgets(ui);
                        return;
                    }

                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Agents").strong().size(12.0));
                        ui.add_space(8.0);
                        egui::ComboBox::from_id_salt("add_agent_kind")
                            .selected_text(self.nodes_panel.selected_add_kind.label())
                            .show_ui(ui, |ui| {
                                for kind in [
                                    AgentNodeKind::Manager,
                                    AgentNodeKind::Worker,
                                    AgentNodeKind::Evaluator,
                                    AgentNodeKind::Researcher,
                                ] {
                                    ui.selectable_value(
                                        &mut self.nodes_panel.selected_add_kind,
                                        kind,
                                        kind.label(),
                                    );
                                }
                            });
                        if ui
                            .add_enabled(!self.read_only_replay_mode, egui::Button::new("Add"))
                            .clicked()
                        {
                            let mut node = match self.nodes_panel.selected_add_kind {
                                AgentNodeKind::Manager => NodeData::new_manager(),
                                AgentNodeKind::Worker => NodeData::new_worker(),
                                AgentNodeKind::Evaluator => NodeData::new_evaluator(),
                                AgentNodeKind::Researcher => NodeData::new_researcher(),
                                AgentNodeKind::Topic => NodeData::new_topic(),
                            };
                            node.set_name(BasicNodeViewer::numbered_name_for_kind(
                                &self.nodes_panel.agents,
                                self.nodes_panel.selected_add_kind,
                            ));
                            self.nodes_panel
                                .push_agent(egui::pos2(0.0, 0.0), node);
                        }
                        if self.read_only_replay_mode {
                            ui.add_space(8.0);
                            ui.label(egui::RichText::new("Replay mode (read-only)").weak());
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("File:");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.agents_workspace_path)
                                .desired_width(280.0),
                        );
                        if ui.button("Load").clicked() {
                            let path = PathBuf::from(self.agents_workspace_path.trim());
                            if let Err(e) = self.load_agents_workspace_from_path(path) {
                                self.manifest_status_message =
                                    format!("Load workspace failed: {e}");
                            }
                        }
                        if ui.button("Save").clicked() {
                            let path = PathBuf::from(self.agents_workspace_path.trim());
                            if let Err(e) = self.save_agents_workspace_to_path(path) {
                                self.manifest_status_message =
                                    format!("Save workspace failed: {e}");
                            }
                        }
                        let (start_stop_label, start_stop_hover) =
                            if self
                                .conversation_graph_running
                                .load(Ordering::Acquire)
                            {
                                (
                                    "Stop",
                                    "Stop conversation loops and agent streaming to the chat endpoint.",
                                )
                            } else {
                                (
                                    "Start",
                                    "Save run manifest and start conversation loops for workers with a topic (paired in node order; optional HTTP streaming).",
                                )
                            };
                        if ui
                            .add_enabled(
                                !self.read_only_replay_mode,
                                egui::Button::new(start_stop_label),
                            )
                            .on_hover_text(start_stop_hover)
                            .clicked()
                        {
                            if self
                                .conversation_graph_running
                                .load(Ordering::Acquire)
                            {
                                self.stop_graph();
                            } else {
                                let _ = self.run_graph();
                            }
                        }
                    });
                    if !self.manifest_status_message.trim().is_empty() {
                        ui.label(
                            egui::RichText::new(self.manifest_status_message.clone()).small(),
                        );
                    }
                    ui.separator();

                    let mut node_ids: Vec<usize> =
                        self.nodes_panel.agents.iter().map(|a| a.id).collect();
                    node_ids.sort_unstable();

                    egui::ScrollArea::vertical().show(ui, |ui| {
                        let mut row_remove: Vec<usize> = Vec::new();
                        for node_id in node_ids {
                            let Some(rec) = self
                                .nodes_panel
                                .agents
                                .iter()
                                .find(|a| a.id == node_id)
                            else {
                                continue;
                            };
                            let node = &rec.data;
                            let global_id = match &node.payload {
                                NodePayload::Manager(m) => m.global_id.as_str(),
                                NodePayload::Worker(w) => w.global_id.as_str(),
                                NodePayload::Evaluator(e) => e.global_id.as_str(),
                                NodePayload::Researcher(r) => r.global_id.as_str(),
                                NodePayload::Topic(t) => t.global_id.as_str(),
                            };
                            let row_label = node.label.clone();

                            ui.set_width(ui.available_width());
                            let row_state_id = ui.make_persistent_id(("agent_row", node_id));
                            let header_close = egui::collapsing_header::CollapsingState::load_with_default_open(
                                ui.ctx(),
                                row_state_id,
                                true,
                            )
                            .show_header(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = 6.0;
                                    ui.label(&row_label);
                                    ui.label("•");
                                    ui.label(global_id);
                                    ui.label("•");
                                    if !self.read_only_replay_mode {
                                        let x_btn = egui::Button::new(
                                            egui::RichText::new(regular::X)
                                                .line_height(Some(ui.text_style_height(&egui::TextStyle::Body))),
                                        )
                                        .frame(false)
                                        .min_size(egui::Vec2::ZERO)
                                        .small();
                                        if ui
                                            .add(x_btn)
                                            .on_hover_text("Remove")
                                            .clicked()
                                        {
                                            row_remove.push(node_id);
                                        }
                                    }
                                });
                            });
                            let _ = header_close.body(|ui| {
                                ui.add_enabled_ui(!self.read_only_replay_mode, |ui| {
                                    viewer.show_body(node_id, ui, &mut self.nodes_panel.agents);
                                });
                            });
                            ui.add_space(4.0);
                        }
                        for id in row_remove {
                            self.nodes_panel.remove_agent(id);
                        }
                    });
                });

                let ctx = ui.ctx().clone();
                sync_evaluator_researcher_activity(&mut self.nodes_panel.agents);
                let mut pending_events = {
                    let mut q = self.conversation_message_events.lock().unwrap();
                    std::mem::take(&mut *q)
                };
                if !self
                    .conversation_graph_running
                    .load(Ordering::Acquire)
                {
                    // Backward-compatible fallback if queue is empty but latest still has a message.
                    if pending_events.is_empty() {
                        if let Some(last_msg) = self.last_message_in_chat.lock().unwrap().clone()
                        {
                            pending_events.push(last_msg);
                        }
                    }
                    // HTTP POST for evaluator/researcher results (no separate output node type).
                    let has_output_nodes = true;

                    // Queue new events for active Evaluator nodes.
                    for rec in &self.nodes_panel.agents {
                        let id = rec.id;
                        if let NodePayload::Evaluator(e) = &rec.data.payload {
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
                                    .get(&id)
                                    .cloned();
                                if event_key.is_empty() || last_eval.as_ref() == Some(&event_key) {
                                    continue;
                                }
                                self.last_evaluated_message_by_evaluator
                                    .lock()
                                    .unwrap()
                                    .insert(id, event_key.clone());
                                self.evaluator_event_queues
                                    .lock()
                                    .unwrap()
                                    .entry(id)
                                    .or_default()
                                    .push_back((event_key, message));
                            }
                        }
                    }

                    // Queue new events for active Researcher nodes.
                    for rec in &self.nodes_panel.agents {
                        let id = rec.id;
                        if let NodePayload::Researcher(r) = &rec.data.payload {
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
                                    .get(&id)
                                    .cloned();
                                if event_key.is_empty()
                                    || last_research.as_ref() == Some(&event_key)
                                {
                                    continue;
                                }
                                self.last_researched_message_by_researcher
                                    .lock()
                                    .unwrap()
                                    .insert(id, event_key.clone());
                                self.researcher_event_queues
                                    .lock()
                                    .unwrap()
                                    .entry(id)
                                    .or_default()
                                    .push_back((event_key, message));
                            }
                        }
                    }

                    // Start at most one Evaluator inference per node (strict sequential order).
                    for rec in &self.nodes_panel.agents {
                        let id = rec.id;
                        let NodePayload::Evaluator(e) = &rec.data.payload else {
                            continue;
                        };
                        if !e.active {
                            continue;
                        }
                        if self.evaluator_inflight_nodes.lock().unwrap().contains(&id) {
                            continue;
                        }
                        let next_msg = self
                            .evaluator_event_queues
                            .lock()
                            .unwrap()
                            .entry(id)
                            .or_default()
                            .pop_front();
                        let Some((_, message)) = next_msg else {
                            continue;
                        };

                        self.evaluator_inflight_nodes.lock().unwrap().insert(id);
                        let inflight = self.evaluator_inflight_nodes.clone();
                        let node_key = id;
                        let instruction = e.instruction.clone();
                        let analysis_mode = e.analysis_mode.clone();
                        let limit_token = e.limit_token;
                        let num_predict = e.num_predict.clone();
                        let endpoint = self.http_endpoint.clone();
                        let ollama_host = self.ollama_host.clone();
                        let has_output_nodes = has_output_nodes;
                        let run_context = self.current_run_context.clone();
                        let ctx = ctx.clone();
                        let handle = self.rt_handle.clone();
                        let selected_model = if self.selected_ollama_model.trim().is_empty() {
                            None
                        } else {
                            Some(self.selected_ollama_model.clone())
                        };
                        let epoch_arc = self.ollama_run_epoch.clone();
                        let epoch_caught = self.ollama_run_epoch.load(Ordering::SeqCst);
                        let ledger = self.event_ledger.clone();
                        let eval_global_id = e.global_id.clone();
                        handle.spawn(async move {
                            let ollama_in = format!("{}\n{}", instruction, message);
                            match crate::adk_integration::send_to_ollama(
                                ollama_host.as_str(),
                                &instruction,
                                &message,
                                limit_token,
                                &num_predict,
                                selected_model.as_deref(),
                                Some((epoch_arc, epoch_caught)),
                            )
                            .await
                            {
                                Ok(response) => {
                                    if let Some(ref l) = ledger {
                                        let _ = l.append_with_hashes(
                                            "sidecar.evaluator",
                                            Some(eval_global_id.clone()),
                                            selected_model.clone(),
                                            &ollama_in,
                                            &response,
                                            serde_json::json!({ "analysis_mode": analysis_mode }),
                                        );
                                    }
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
                                        if let Err(e) =
                                            crate::http_client::send_evaluator_result(
                                                &endpoint,
                                                "Agent Evaluator",
                                                sentiment,
                                                &response,
                                                run_context.as_ref(),
                                                ledger.as_ref(),
                                            )
                                            .await
                                        {
                                            eprintln!(
                                                "[Evaluator] Failed to send to ams-chat: {}",
                                                e
                                            );
                                        }
                                    }
                                }
                                Err(e) => {
                                    if e.to_string() != crate::adk_integration::OLLAMA_STOPPED_MSG
                                    {
                                        if let Some(ref l) = ledger {
                                            let _ = l.append_with_hashes(
                                                "sidecar.evaluator",
                                                Some(eval_global_id.clone()),
                                                selected_model.clone(),
                                                &ollama_in,
                                                "",
                                                serde_json::json!({
                                                    "analysis_mode": analysis_mode,
                                                    "stage": "ollama",
                                                    "error": e.to_string(),
                                                }),
                                            );
                                        }
                                        eprintln!("[Evaluator] Ollama error: {}", e);
                                    }
                                }
                            }
                            inflight.lock().unwrap().remove(&node_key);
                            ctx.request_repaint();
                        });
                    }

                    // Start at most one Researcher inference per node (strict sequential order).
                    for rec in &self.nodes_panel.agents {
                        let id = rec.id;
                        let NodePayload::Researcher(r) = &rec.data.payload else {
                            continue;
                        };
                        if !r.active {
                            continue;
                        }
                        if self.researcher_inflight_nodes.lock().unwrap().contains(&id) {
                            continue;
                        }
                        let next_msg = self
                            .researcher_event_queues
                            .lock()
                            .unwrap()
                            .entry(id)
                            .or_default()
                            .pop_front();
                        let Some((_, message)) = next_msg else {
                            continue;
                        };

                        self.researcher_inflight_nodes.lock().unwrap().insert(id);
                        let inflight = self.researcher_inflight_nodes.clone();
                        let node_key = id;
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
                        let ollama_host = self.ollama_host.clone();
                        let has_output_nodes = has_output_nodes;
                        let run_context = self.current_run_context.clone();
                        let ctx = ctx.clone();
                        let handle = self.rt_handle.clone();
                        let selected_model = if self.selected_ollama_model.trim().is_empty() {
                            None
                        } else {
                            Some(self.selected_ollama_model.clone())
                        };
                        let epoch_arc = self.ollama_run_epoch.clone();
                        let epoch_caught = self.ollama_run_epoch.load(Ordering::SeqCst);
                        let ledger = self.event_ledger.clone();
                        let res_global_id = r.global_id.clone();
                        handle.spawn(async move {
                            let ollama_in = format!("{}\n{}", instruction, message);
                            match crate::adk_integration::send_to_ollama(
                                ollama_host.as_str(),
                                &instruction,
                                &message,
                                limit_token,
                                &num_predict,
                                selected_model.as_deref(),
                                Some((epoch_arc, epoch_caught)),
                            )
                            .await
                            {
                                Ok(response) => {
                                    if let Some(ref l) = ledger {
                                        let _ = l.append_with_hashes(
                                            "sidecar.researcher",
                                            Some(res_global_id.clone()),
                                            selected_model.clone(),
                                            &ollama_in,
                                            &response,
                                            serde_json::json!({ "topic": topic }),
                                        );
                                    }
                                    if has_output_nodes {
                                        if let Err(e) =
                                            crate::http_client::send_researcher_result(
                                                &endpoint,
                                                "Agent Researcher",
                                                &topic,
                                                &response,
                                                run_context.as_ref(),
                                                ledger.as_ref(),
                                            )
                                            .await
                                        {
                                            eprintln!(
                                                "[Researcher] Failed to send to ams-chat: {}",
                                                e
                                            );
                                        }
                                    }
                                }
                                Err(e) => {
                                    if e.to_string()
                                        != crate::adk_integration::OLLAMA_STOPPED_MSG
                                    {
                                        if let Some(ref l) = ledger {
                                            let _ = l.append_with_hashes(
                                                "sidecar.researcher",
                                                Some(res_global_id.clone()),
                                                selected_model.clone(),
                                                &ollama_in,
                                                "",
                                                serde_json::json!({
                                                    "topic": topic,
                                                    "stage": "ollama",
                                                    "error": e.to_string(),
                                                }),
                                            );
                                        }
                                        eprintln!("[Researcher] Ollama error: {}", e);
                                    }
                                }
                            }
                            inflight.lock().unwrap().remove(&node_key);
                            ctx.request_repaint();
                        });
                    }
                }
            },
        );
    }
}
