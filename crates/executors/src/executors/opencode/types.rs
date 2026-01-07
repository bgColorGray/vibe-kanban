use serde::{Deserialize, Serialize};
use workspace_utils::approvals::ApprovalStatus;

/// JSON log events emitted by the OpenCode SDK executor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpencodeExecutorEvent {
    SessionStart {
        session_id: String,
    },
    SdkEvent {
        event: serde_json::Value,
    },
    ApprovalResponse {
        tool_call_id: String,
        status: ApprovalStatus,
    },
    Error {
        message: String,
    },
    Done,
}
