//! Panel tab selection and the Agents list (`NodesPanelState`).

use eframe::egui;

use super::model::NodeData;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PanelTab {
    Overview,
    Agents,
    Ollama,
    Python,
    Settings,
}

/// One row in the Agents list (stable `id` for manifests and conversation loops).
#[derive(Clone)]
pub(crate) struct AgentRecord {
    pub(crate) id: usize,
    pub(crate) position: egui::Pos2,
    pub(crate) open: bool,
    pub(crate) data: super::model::NodeData,
}

pub struct NodesPanelState {
    pub(crate) next_agent_id: usize,
    pub(crate) agents: Vec<AgentRecord>,
    pub(crate) selected_add_kind: super::model::AgentNodeKind,
    pub(crate) active_tab: PanelTab,
}

impl Default for NodesPanelState {
    fn default() -> Self {
        Self {
            next_agent_id: 0,
            agents: Vec::new(),
            selected_add_kind: super::model::AgentNodeKind::Worker,
            active_tab: PanelTab::Overview,
        }
    }
}

impl NodesPanelState {
    // pub(crate) fn push_agent(&mut self, pos: egui::Pos2, data: NodeData) -> usize {
    //     let id = self.next_agent_id;
    //     self.next_agent_id += 1;
    //     self.agents.push(AgentRecord {
    //         id,
    //         position: pos,
    //         open: true,
    //         data,
    //     });
    //     id
    // }

    pub(crate) fn insert_agent_with_id(
        &mut self,
        id: usize,
        pos: egui::Pos2,
        open: bool,
        data: NodeData,
    ) {
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

    pub(crate) fn remove_agent(&mut self, id: usize) {
        self.agents.retain(|a| a.id != id);
    }
}
