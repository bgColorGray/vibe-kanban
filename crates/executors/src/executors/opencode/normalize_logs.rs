use std::{collections::HashMap, path::Path, sync::Arc};

use futures::StreamExt;
use serde::Deserialize;
use serde_json::Value;
use workspace_utils::{approvals::ApprovalStatus, msg_store::MsgStore, path::make_path_relative};

use super::types::OpencodeExecutorEvent;
use crate::{
    approvals::ToolCallMetadata,
    logs::{
        ActionType, CommandExitStatus, CommandRunResult, FileChange, NormalizedEntry,
        NormalizedEntryError, NormalizedEntryType, TodoItem, ToolResult, ToolStatus,
        stderr_processor::normalize_stderr_logs,
        utils::{
            EntryIndexProvider,
            patch::{add_normalized_entry, replace_normalized_entry, upsert_normalized_entry},
        },
    },
};

trait ToNormalizedEntry {
    fn to_normalized_entry(&self, worktree_path: &Path) -> NormalizedEntry;
}

pub fn normalize_logs(msg_store: Arc<MsgStore>, worktree_path: &Path) {
    let entry_index = EntryIndexProvider::start_from(&msg_store);
    normalize_stderr_logs(msg_store.clone(), entry_index.clone());

    let worktree_path = worktree_path.to_path_buf();
    tokio::spawn(async move {
        let mut stored_session_id = false;
        let mut state = LogState::new(entry_index.clone());

        let mut stdout_lines = msg_store.stdout_lines_stream();
        while let Some(Ok(line)) = stdout_lines.next().await {
            let Some(event) = parse_event(&line) else {
                continue;
            };

            match event {
                OpencodeExecutorEvent::SessionStart { session_id } => {
                    if !stored_session_id {
                        msg_store.push_session_id(session_id);
                        stored_session_id = true;
                    }
                }
                OpencodeExecutorEvent::SdkEvent { event } => {
                    state.handle_sdk_event(&event, &worktree_path, &msg_store);
                }
                OpencodeExecutorEvent::ApprovalResponse {
                    tool_call_id,
                    status,
                } => {
                    state.handle_approval_response(
                        &tool_call_id,
                        status,
                        &worktree_path,
                        &msg_store,
                    );
                }
                OpencodeExecutorEvent::Error { message } => {
                    let idx = entry_index.next();
                    msg_store.push_patch(
                        crate::logs::utils::ConversationPatch::add_normalized_entry(
                            idx,
                            NormalizedEntry {
                                timestamp: None,
                                entry_type: NormalizedEntryType::ErrorMessage {
                                    error_type: NormalizedEntryError::Other,
                                },
                                content: message,
                                metadata: None,
                            },
                        ),
                    );
                }
                OpencodeExecutorEvent::Done => {
                    // no-op; stream will end naturally
                }
            }
        }
    });
}

fn parse_event(line: &str) -> Option<OpencodeExecutorEvent> {
    serde_json::from_str::<OpencodeExecutorEvent>(line.trim()).ok()
}

#[derive(Debug, Clone)]
struct StreamingText {
    index: usize,
    content: String,
}

#[derive(Debug, Clone)]
enum UpdateMode {
    Append,
    Set,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum MessageRole {
    User,
    Assistant,
}

#[derive(Default)]
struct LogState {
    entry_index: EntryIndexProvider,
    message_roles: HashMap<String, MessageRole>,
    assistant_text: HashMap<String, StreamingText>,
    thinking_text: HashMap<String, StreamingText>,
    tool_states: HashMap<String, ToolCallState>,
    approvals: HashMap<String, ApprovalStatus>,
    model_system_message_emitted: bool,
}

impl LogState {
    fn new(entry_index: EntryIndexProvider) -> Self {
        Self {
            entry_index,
            message_roles: HashMap::new(),
            assistant_text: HashMap::new(),
            thinking_text: HashMap::new(),
            tool_states: HashMap::new(),
            approvals: HashMap::new(),
            model_system_message_emitted: false,
        }
    }

    fn handle_sdk_event(&mut self, raw: &Value, worktree_path: &Path, msg_store: &Arc<MsgStore>) {
        let Some(event) = SdkEvent::parse(raw) else {
            return;
        };

        match event {
            SdkEvent::MessageUpdated(MessageUpdatedEvent { info }) => {
                self.maybe_emit_model_system_message(&info, msg_store);
                self.message_roles.insert(info.id, info.role);
            }
            SdkEvent::MessagePartUpdated(MessagePartUpdatedEvent { part, delta }) => {
                self.handle_part_update(part, delta.as_deref(), worktree_path, msg_store);
            }
            SdkEvent::SessionError(SessionErrorEvent { error }) => {
                let (error_type, message) = match error {
                    Some(err) if err.kind() == "ProviderAuthError" => (
                        NormalizedEntryError::SetupRequired,
                        err.message()
                            .unwrap_or_else(|| format!("OpenCode session error: {}", err.raw)),
                    ),
                    Some(err) => (
                        NormalizedEntryError::Other,
                        format!("OpenCode session error: {}", err.raw),
                    ),
                    None => (
                        NormalizedEntryError::Other,
                        "OpenCode session error".to_string(),
                    ),
                };

                let idx = self.entry_index.next();
                msg_store.push_patch(crate::logs::utils::ConversationPatch::add_normalized_entry(
                    idx,
                    NormalizedEntry {
                        timestamp: None,
                        entry_type: NormalizedEntryType::ErrorMessage { error_type },
                        content: message,
                        metadata: None,
                    },
                ));
            }
        }
    }

    fn maybe_emit_model_system_message(&mut self, info: &MessageInfo, msg_store: &Arc<MsgStore>) {
        if self.model_system_message_emitted {
            return;
        }

        let Some(model_id) = info.model_id() else {
            return;
        };
        let Some(provider_id) = info.provider_id() else {
            return;
        };

        add_normalized_entry(
            msg_store,
            &self.entry_index,
            NormalizedEntry {
                timestamp: None,
                entry_type: NormalizedEntryType::SystemMessage,
                content: format!("model: {model_id}  provider: {provider_id}"),
                metadata: None,
            },
        );
        self.model_system_message_emitted = true;
    }

    fn handle_part_update(
        &mut self,
        part: Part,
        delta: Option<&str>,
        worktree_path: &Path,
        msg_store: &Arc<MsgStore>,
    ) {
        match part {
            Part::Text(part) => {
                if self.message_roles.get(&part.message_id) != Some(&MessageRole::Assistant) {
                    return;
                }

                let (text, mode) = if let Some(delta) = delta {
                    (delta, UpdateMode::Append)
                } else {
                    (part.text.as_str(), UpdateMode::Set)
                };

                let entry_index = self.entry_index.clone();
                update_streaming_text(
                    &entry_index,
                    text,
                    NormalizedEntryType::AssistantMessage,
                    &part.message_id,
                    &mut self.assistant_text,
                    msg_store,
                    mode,
                );
            }
            Part::Reasoning(part) => {
                if self.message_roles.get(&part.message_id) != Some(&MessageRole::Assistant) {
                    return;
                }

                let (text, mode) = if let Some(delta) = delta {
                    (delta, UpdateMode::Append)
                } else {
                    (part.text.as_str(), UpdateMode::Set)
                };

                let entry_index = self.entry_index.clone();
                update_streaming_text(
                    &entry_index,
                    text,
                    NormalizedEntryType::Thinking,
                    &part.message_id,
                    &mut self.thinking_text,
                    msg_store,
                    mode,
                );
            }
            Part::Tool(part) => {
                if self.message_roles.get(&part.message_id) != Some(&MessageRole::Assistant) {
                    return;
                }
                if part.call_id.trim().is_empty() {
                    return;
                }

                let tool_state = self
                    .tool_states
                    .entry(part.call_id.clone())
                    .or_insert_with(|| ToolCallState::new(part.call_id.clone()));

                tool_state.set_approval_if_missing(self.approvals.get(&part.call_id).cloned());

                tool_state.update_from_part(part);
                let entry = tool_state.to_normalized_entry(worktree_path);
                if let Some(index) = tool_state.index() {
                    replace_normalized_entry(msg_store, index, entry);
                } else {
                    let index = add_normalized_entry(msg_store, &self.entry_index, entry);
                    tool_state.set_index(index);
                }
            }
            Part::Other => {}
        }
    }

    fn handle_approval_response(
        &mut self,
        tool_call_id: &str,
        status: ApprovalStatus,
        worktree_path: &Path,
        msg_store: &Arc<MsgStore>,
    ) {
        self.approvals
            .insert(tool_call_id.to_string(), status.clone());

        if let ApprovalStatus::Denied { reason } = &status {
            let tool_name = self
                .tool_states
                .get(tool_call_id)
                .map(|t| t.tool_name().to_string())
                .unwrap_or_else(|| "tool".to_string());

            let idx = self.entry_index.next();
            msg_store.push_patch(crate::logs::utils::ConversationPatch::add_normalized_entry(
                idx,
                NormalizedEntry {
                    timestamp: None,
                    entry_type: NormalizedEntryType::UserFeedback {
                        denied_tool: tool_name,
                    },
                    content: reason
                        .clone()
                        .unwrap_or_else(|| "User denied this tool use request".to_string())
                        .trim()
                        .to_string(),
                    metadata: None,
                },
            ));
        }

        let Some(tool_state) = self.tool_states.get_mut(tool_call_id) else {
            return;
        };

        tool_state.set_approval(status);

        let Some(index) = tool_state.index() else {
            return;
        };

        replace_normalized_entry(
            msg_store,
            index,
            tool_state.to_normalized_entry(worktree_path),
        );
    }
}

fn update_streaming_text(
    entry_index: &EntryIndexProvider,
    text: &str,
    entry_type: NormalizedEntryType,
    message_id: &str,
    map: &mut HashMap<String, StreamingText>,
    msg_store: &Arc<MsgStore>,
    mode: UpdateMode,
) {
    if text.is_empty() {
        return;
    }

    let is_new = !map.contains_key(message_id);
    let state = map
        .entry(message_id.to_string())
        .or_insert_with(|| StreamingText {
            index: entry_index.next(),
            content: String::new(),
        });

    match mode {
        UpdateMode::Append => state.content.push_str(text),
        UpdateMode::Set => state.content = text.to_string(),
    }

    let entry = NormalizedEntry {
        timestamp: None,
        entry_type,
        content: state.content.clone(),
        metadata: None,
    };
    upsert_normalized_entry(msg_store, state.index, entry, is_new);
}

#[derive(Debug, Clone)]
struct ToolCallCommon {
    index: Option<usize>,
    call_id: String,
    tool_name: String,
    state: ToolStateStatus,
    title: Option<String>,
    approval: Option<ApprovalStatus>,
}

impl ToolCallCommon {
    fn new(call_id: String) -> Self {
        Self {
            index: None,
            call_id,
            tool_name: "tool".to_string(),
            state: ToolStateStatus::Unknown,
            title: None,
            approval: None,
        }
    }

    fn set_title_if_present(&mut self, title: Option<String>) {
        if let Some(title) = title
            && !title.trim().is_empty()
        {
            self.title = Some(title);
        }
    }

    fn set_tool_name_if_present(&mut self, tool: String) {
        if tool.trim().is_empty() {
            return;
        }
        self.tool_name = tool;
    }

    fn tool_status(&self) -> ToolStatus {
        if let Some(ApprovalStatus::Denied { reason }) = &self.approval {
            return ToolStatus::Denied {
                reason: reason.clone(),
            };
        }
        if matches!(self.approval, Some(ApprovalStatus::TimedOut)) {
            return ToolStatus::TimedOut;
        }

        match self.state {
            ToolStateStatus::Completed => ToolStatus::Success,
            ToolStateStatus::Error => ToolStatus::Failed,
            ToolStateStatus::Pending | ToolStateStatus::Running | ToolStateStatus::Unknown => {
                ToolStatus::Created
            }
        }
    }
}

#[derive(Debug, Clone)]
enum ToolCallState {
    Bash(BashToolState),
    Read(ReadToolState),
    FileEdit(FileEditToolState),
    WebFetch(WebFetchToolState),
    Search(SearchToolState),
    Todo(TodoToolState),
    Task(TaskToolState),
    Other(OtherToolState),
}

impl ToolCallState {
    fn new(call_id: String) -> Self {
        Self::Other(OtherToolState::new(call_id))
    }

    fn common(&self) -> &ToolCallCommon {
        match self {
            ToolCallState::Bash(s) => &s.common,
            ToolCallState::Read(s) => &s.common,
            ToolCallState::FileEdit(s) => &s.common,
            ToolCallState::WebFetch(s) => &s.common,
            ToolCallState::Search(s) => &s.common,
            ToolCallState::Todo(s) => &s.common,
            ToolCallState::Task(s) => &s.common,
            ToolCallState::Other(s) => &s.common,
        }
    }

    fn common_mut(&mut self) -> &mut ToolCallCommon {
        match self {
            ToolCallState::Bash(s) => &mut s.common,
            ToolCallState::Read(s) => &mut s.common,
            ToolCallState::FileEdit(s) => &mut s.common,
            ToolCallState::WebFetch(s) => &mut s.common,
            ToolCallState::Search(s) => &mut s.common,
            ToolCallState::Todo(s) => &mut s.common,
            ToolCallState::Task(s) => &mut s.common,
            ToolCallState::Other(s) => &mut s.common,
        }
    }

    fn index(&self) -> Option<usize> {
        self.common().index
    }

    fn set_index(&mut self, index: usize) {
        self.common_mut().index = Some(index);
    }

    fn tool_name(&self) -> &str {
        self.common().tool_name.as_str()
    }

    fn set_approval_if_missing(&mut self, approval: Option<ApprovalStatus>) {
        if self.common().approval.is_none() {
            self.common_mut().approval = approval;
        }
    }

    fn set_approval(&mut self, approval: ApprovalStatus) {
        self.common_mut().approval = Some(approval);
    }

    fn update_from_part(&mut self, part: ToolPart) {
        match self {
            ToolCallState::Bash(s) => s.update_from_part(part),
            ToolCallState::Read(s) => s.update_from_part(part),
            ToolCallState::FileEdit(s) => s.update_from_part(part),
            ToolCallState::WebFetch(s) => s.update_from_part(part),
            ToolCallState::Search(s) => s.update_from_part(part),
            ToolCallState::Todo(s) => s.update_from_part(part),
            ToolCallState::Task(s) => s.update_from_part(part),
            ToolCallState::Other(s) => s.update_from_part(part),
        }

        self.promote_if_known_tool();
    }

    fn promote_if_known_tool(&mut self) {
        let (tool_name, is_other) = match self {
            ToolCallState::Other(s) => (s.common.tool_name.clone(), true),
            _ => return,
        };

        let promote = matches!(
            tool_name.as_str(),
            "bash"
                | "read"
                | "edit"
                | "write"
                | "multiedit"
                | "webfetch"
                | "websearch"
                | "codesearch"
                | "grep"
                | "glob"
                | "todoread"
                | "todowrite"
                | "task"
        );

        if !is_other || !promote {
            return;
        }

        let call_id = self.common().call_id.clone();
        let old = std::mem::replace(self, ToolCallState::Other(OtherToolState::new(call_id)));
        let ToolCallState::Other(other) = old else {
            return;
        };

        *self = ToolCallState::from_other(other);
    }

    fn from_other(other: OtherToolState) -> Self {
        match other.common.tool_name.as_str() {
            "bash" => ToolCallState::Bash(BashToolState::from_other(other)),
            "read" => ToolCallState::Read(ReadToolState::from_other(other)),
            "edit" | "write" | "multiedit" => {
                ToolCallState::FileEdit(FileEditToolState::from_other(other))
            }
            "webfetch" => ToolCallState::WebFetch(WebFetchToolState::from_other(other)),
            "websearch" | "codesearch" | "grep" | "glob" => {
                ToolCallState::Search(SearchToolState::from_other(other))
            }
            "todoread" | "todowrite" => ToolCallState::Todo(TodoToolState::from_other(other)),
            "task" => ToolCallState::Task(TaskToolState::from_other(other)),
            _ => ToolCallState::Other(other),
        }
    }

    fn to_normalized_entry(&self, worktree_path: &Path) -> NormalizedEntry {
        match self {
            ToolCallState::Bash(s) => s.to_normalized_entry(worktree_path),
            ToolCallState::Read(s) => s.to_normalized_entry(worktree_path),
            ToolCallState::FileEdit(s) => s.to_normalized_entry(worktree_path),
            ToolCallState::WebFetch(s) => s.to_normalized_entry(worktree_path),
            ToolCallState::Search(s) => s.to_normalized_entry(worktree_path),
            ToolCallState::Todo(s) => s.to_normalized_entry(worktree_path),
            ToolCallState::Task(s) => s.to_normalized_entry(worktree_path),
            ToolCallState::Other(s) => s.to_normalized_entry(worktree_path),
        }
    }
}

#[derive(Debug, Clone)]
struct BashToolState {
    common: ToolCallCommon,
    command: Option<String>,
    output: Option<String>,
    error: Option<String>,
    exit_code: Option<i32>,
}

impl BashToolState {
    fn from_other(other: OtherToolState) -> Self {
        let mut state = Self {
            common: other.common,
            command: None,
            output: other.output,
            error: other.error,
            exit_code: None,
        };
        state.update_from_raw(other.input, other.metadata);
        state
    }

    fn update_from_part(&mut self, part: ToolPart) {
        self.common.set_tool_name_if_present(part.tool);
        match part.state {
            ToolStateUpdate::Pending { input } => {
                self.common.state = ToolStateStatus::Pending;
                self.update_from_raw(input, None);
            }
            ToolStateUpdate::Running {
                input,
                title,
                metadata,
            } => {
                self.common.state = ToolStateStatus::Running;
                self.common.set_title_if_present(title);
                self.update_from_raw(input, metadata);
            }
            ToolStateUpdate::Completed {
                input,
                output,
                title,
                metadata,
            } => {
                self.common.state = ToolStateStatus::Completed;
                self.common.set_title_if_present(title);
                if let Some(output) = output {
                    self.output = Some(output);
                }
                self.error = None;
                self.update_from_raw(input, metadata);
            }
            ToolStateUpdate::Error {
                input,
                error,
                metadata,
            } => {
                self.common.state = ToolStateStatus::Error;
                if let Some(error) = error
                    && !error.trim().is_empty()
                {
                    self.error = Some(error);
                }
                self.update_from_raw(input, metadata);
            }
            ToolStateUpdate::Unknown => {}
        }
    }

    fn update_from_raw(&mut self, input: Option<Value>, metadata: Option<Value>) {
        if let Some(input) = input {
            if let Ok(parsed) = serde_json::from_value::<BashInput>(input) {
                self.command = Some(parsed.command);
            }
        }

        if let Some(metadata) = metadata {
            self.exit_code = metadata
                .get("exit")
                .and_then(Value::as_i64)
                .map(|c| c as i32);
        }
    }

    fn output(&self) -> Option<&str> {
        self.output.as_deref().or(self.error.as_deref())
    }
}

impl ToNormalizedEntry for BashToolState {
    fn to_normalized_entry(&self, _worktree_path: &Path) -> NormalizedEntry {
        let command = self.command.clone().unwrap_or_default();
        let action_type = ActionType::CommandRun {
            command: command.clone(),
            result: Some(CommandRunResult {
                exit_status: self
                    .exit_code
                    .map(|code| CommandExitStatus::ExitCode { code }),
                output: self.output().map(|s| s.to_string()),
            }),
        };
        let content = tool_content(
            self.common.title.as_deref(),
            &self.common.tool_name,
            &action_type,
        );

        NormalizedEntry {
            timestamp: None,
            entry_type: NormalizedEntryType::ToolUse {
                tool_name: self.common.tool_name.clone(),
                action_type,
                status: self.common.tool_status(),
            },
            content,
            metadata: serde_json::to_value(ToolCallMetadata {
                tool_call_id: self.common.call_id.clone(),
            })
            .ok(),
        }
    }
}

#[derive(Debug, Clone)]
struct ReadToolState {
    common: ToolCallCommon,
    file_path: Option<String>,
}

impl ReadToolState {
    fn from_other(other: OtherToolState) -> Self {
        let mut state = Self {
            common: other.common,
            file_path: None,
        };
        state.update_from_raw(other.input);
        state
    }

    fn update_from_part(&mut self, part: ToolPart) {
        self.common.set_tool_name_if_present(part.tool);
        match part.state {
            ToolStateUpdate::Pending { input } => {
                self.common.state = ToolStateStatus::Pending;
                self.update_from_raw(input);
            }
            ToolStateUpdate::Running {
                input,
                title,
                metadata: _,
            } => {
                self.common.state = ToolStateStatus::Running;
                self.common.set_title_if_present(title);
                self.update_from_raw(input);
            }
            ToolStateUpdate::Completed {
                input,
                title,
                metadata: _,
                output: _,
            } => {
                self.common.state = ToolStateStatus::Completed;
                self.common.set_title_if_present(title);
                self.update_from_raw(input);
            }
            ToolStateUpdate::Error {
                input,
                error: _,
                metadata: _,
            } => {
                self.common.state = ToolStateStatus::Error;
                self.update_from_raw(input);
            }
            ToolStateUpdate::Unknown => {}
        }
    }

    fn update_from_raw(&mut self, input: Option<Value>) {
        if let Some(input) = input {
            if let Ok(parsed) = serde_json::from_value::<FilePathInput>(input) {
                self.file_path = Some(parsed.file_path);
            }
        }
    }
}

impl ToNormalizedEntry for ReadToolState {
    fn to_normalized_entry(&self, worktree_path: &Path) -> NormalizedEntry {
        let path = self
            .file_path
            .as_deref()
            .map(|p| make_relative_path(p, worktree_path))
            .unwrap_or_default();
        let action_type = ActionType::FileRead { path };
        let content = tool_content(
            self.common.title.as_deref(),
            &self.common.tool_name,
            &action_type,
        );

        NormalizedEntry {
            timestamp: None,
            entry_type: NormalizedEntryType::ToolUse {
                tool_name: self.common.tool_name.clone(),
                action_type,
                status: self.common.tool_status(),
            },
            content,
            metadata: serde_json::to_value(ToolCallMetadata {
                tool_call_id: self.common.call_id.clone(),
            })
            .ok(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileEditKind {
    Edit,
    Write,
    MultiEdit,
}

#[derive(Debug, Clone)]
struct FileEditToolState {
    common: ToolCallCommon,
    kind: FileEditKind,
    file_path: Option<String>,
    write_content: Option<String>,
    unified_diff: Option<String>,
}

impl FileEditToolState {
    fn from_other(other: OtherToolState) -> Self {
        let kind = match other.common.tool_name.as_str() {
            "write" => FileEditKind::Write,
            "multiedit" => FileEditKind::MultiEdit,
            _ => FileEditKind::Edit,
        };

        let mut state = Self {
            common: other.common,
            kind,
            file_path: None,
            write_content: None,
            unified_diff: None,
        };
        state.update_from_raw(other.input, other.metadata);
        state
    }

    fn update_from_part(&mut self, part: ToolPart) {
        self.common.set_tool_name_if_present(part.tool);
        self.kind = match self.common.tool_name.as_str() {
            "write" => FileEditKind::Write,
            "multiedit" => FileEditKind::MultiEdit,
            _ => FileEditKind::Edit,
        };

        match part.state {
            ToolStateUpdate::Pending { input } => {
                self.common.state = ToolStateStatus::Pending;
                self.update_from_raw(input, None);
            }
            ToolStateUpdate::Running {
                input,
                title,
                metadata,
            } => {
                self.common.state = ToolStateStatus::Running;
                self.common.set_title_if_present(title);
                self.update_from_raw(input, metadata);
            }
            ToolStateUpdate::Completed {
                input,
                title,
                metadata,
                output: _,
            } => {
                self.common.state = ToolStateStatus::Completed;
                self.common.set_title_if_present(title);
                self.update_from_raw(input, metadata);
            }
            ToolStateUpdate::Error {
                input,
                error: _,
                metadata,
            } => {
                self.common.state = ToolStateStatus::Error;
                self.update_from_raw(input, metadata);
            }
            ToolStateUpdate::Unknown => {}
        }
    }

    fn update_from_raw(&mut self, input: Option<Value>, metadata: Option<Value>) {
        if let Some(input) = input {
            match self.kind {
                FileEditKind::Write => {
                    if let Ok(parsed) = serde_json::from_value::<WriteInput>(input) {
                        self.file_path = Some(parsed.file_path);
                        self.write_content = Some(parsed.content);
                    }
                }
                FileEditKind::Edit | FileEditKind::MultiEdit => {
                    if let Ok(parsed) = serde_json::from_value::<FilePathInput>(input) {
                        self.file_path = Some(parsed.file_path);
                    }
                }
            }
        }

        if matches!(self.kind, FileEditKind::Edit | FileEditKind::MultiEdit)
            && let Some(metadata) = metadata
        {
            self.unified_diff = extract_diff_from_metadata(&metadata).map(|s| s.to_string());
        }
    }
}

impl ToNormalizedEntry for FileEditToolState {
    fn to_normalized_entry(&self, worktree_path: &Path) -> NormalizedEntry {
        let path = self
            .file_path
            .as_deref()
            .map(|p| make_relative_path(p, worktree_path))
            .unwrap_or_default();

        let changes = match self.kind {
            FileEditKind::Write => self
                .write_content
                .as_ref()
                .filter(|s| !s.is_empty())
                .map(|content| {
                    vec![FileChange::Write {
                        content: content.clone(),
                    }]
                })
                .unwrap_or_default(),
            FileEditKind::Edit | FileEditKind::MultiEdit => self
                .unified_diff
                .as_ref()
                .map(|diff| {
                    vec![FileChange::Edit {
                        unified_diff: workspace_utils::diff::normalize_unified_diff(&path, diff),
                        has_line_numbers: true,
                    }]
                })
                .unwrap_or_default(),
        };

        let action_type = ActionType::FileEdit { path, changes };
        let content = tool_content(
            self.common.title.as_deref(),
            &self.common.tool_name,
            &action_type,
        );

        NormalizedEntry {
            timestamp: None,
            entry_type: NormalizedEntryType::ToolUse {
                tool_name: self.common.tool_name.clone(),
                action_type,
                status: self.common.tool_status(),
            },
            content,
            metadata: serde_json::to_value(ToolCallMetadata {
                tool_call_id: self.common.call_id.clone(),
            })
            .ok(),
        }
    }
}

#[derive(Debug, Clone)]
struct WebFetchToolState {
    common: ToolCallCommon,
    url: Option<String>,
}

impl WebFetchToolState {
    fn from_other(other: OtherToolState) -> Self {
        let mut state = Self {
            common: other.common,
            url: None,
        };
        state.update_from_raw(other.input);
        state
    }

    fn update_from_part(&mut self, part: ToolPart) {
        self.common.set_tool_name_if_present(part.tool);
        match part.state {
            ToolStateUpdate::Pending { input } => {
                self.common.state = ToolStateStatus::Pending;
                self.update_from_raw(input);
            }
            ToolStateUpdate::Running {
                input,
                title,
                metadata: _,
            } => {
                self.common.state = ToolStateStatus::Running;
                self.common.set_title_if_present(title);
                self.update_from_raw(input);
            }
            ToolStateUpdate::Completed {
                input,
                title,
                metadata: _,
                output: _,
            } => {
                self.common.state = ToolStateStatus::Completed;
                self.common.set_title_if_present(title);
                self.update_from_raw(input);
            }
            ToolStateUpdate::Error {
                input,
                error: _,
                metadata: _,
            } => {
                self.common.state = ToolStateStatus::Error;
                self.update_from_raw(input);
            }
            ToolStateUpdate::Unknown => {}
        }
    }

    fn update_from_raw(&mut self, input: Option<Value>) {
        if let Some(input) = input {
            if let Ok(parsed) = serde_json::from_value::<WebFetchInput>(input) {
                self.url = Some(parsed.url);
            }
        }
    }
}

impl ToNormalizedEntry for WebFetchToolState {
    fn to_normalized_entry(&self, _worktree_path: &Path) -> NormalizedEntry {
        let url = self.url.clone().unwrap_or_default();
        let action_type = ActionType::WebFetch { url };
        let content = tool_content(
            self.common.title.as_deref(),
            &self.common.tool_name,
            &action_type,
        );

        NormalizedEntry {
            timestamp: None,
            entry_type: NormalizedEntryType::ToolUse {
                tool_name: self.common.tool_name.clone(),
                action_type,
                status: self.common.tool_status(),
            },
            content,
            metadata: serde_json::to_value(ToolCallMetadata {
                tool_call_id: self.common.call_id.clone(),
            })
            .ok(),
        }
    }
}

#[derive(Debug, Clone)]
struct SearchToolState {
    common: ToolCallCommon,
    query: Option<String>,
}

impl SearchToolState {
    fn from_other(other: OtherToolState) -> Self {
        let mut state = Self {
            common: other.common,
            query: None,
        };
        state.update_from_raw(other.input);
        state
    }

    fn update_from_part(&mut self, part: ToolPart) {
        self.common.set_tool_name_if_present(part.tool);
        match part.state {
            ToolStateUpdate::Pending { input } => {
                self.common.state = ToolStateStatus::Pending;
                self.update_from_raw(input);
            }
            ToolStateUpdate::Running {
                input,
                title,
                metadata: _,
            } => {
                self.common.state = ToolStateStatus::Running;
                self.common.set_title_if_present(title);
                self.update_from_raw(input);
            }
            ToolStateUpdate::Completed {
                input,
                title,
                metadata: _,
                output: _,
            } => {
                self.common.state = ToolStateStatus::Completed;
                self.common.set_title_if_present(title);
                self.update_from_raw(input);
            }
            ToolStateUpdate::Error {
                input,
                error: _,
                metadata: _,
            } => {
                self.common.state = ToolStateStatus::Error;
                self.update_from_raw(input);
            }
            ToolStateUpdate::Unknown => {}
        }
    }

    fn update_from_raw(&mut self, input: Option<Value>) {
        if let Some(input) = input {
            if let Ok(parsed) = serde_json::from_value::<SearchInput>(input) {
                self.query = parsed.query.or(parsed.pattern);
            }
        }
    }
}

impl ToNormalizedEntry for SearchToolState {
    fn to_normalized_entry(&self, _worktree_path: &Path) -> NormalizedEntry {
        let query = self.query.clone().unwrap_or_default();
        let action_type = ActionType::Search { query };
        let content = tool_content(
            self.common.title.as_deref(),
            &self.common.tool_name,
            &action_type,
        );

        NormalizedEntry {
            timestamp: None,
            entry_type: NormalizedEntryType::ToolUse {
                tool_name: self.common.tool_name.clone(),
                action_type,
                status: self.common.tool_status(),
            },
            content,
            metadata: serde_json::to_value(ToolCallMetadata {
                tool_call_id: self.common.call_id.clone(),
            })
            .ok(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum TodoOperation {
    Read,
    Write,
}

#[derive(Debug, Clone)]
struct TodoToolState {
    common: ToolCallCommon,
    operation: TodoOperation,
    todos: Vec<TodoItem>,
}

impl TodoToolState {
    fn from_other(other: OtherToolState) -> Self {
        let operation = match other.common.tool_name.as_str() {
            "todoread" => TodoOperation::Read,
            _ => TodoOperation::Write,
        };

        let mut state = Self {
            common: other.common,
            operation,
            todos: vec![],
        };
        state.update_from_raw(other.input, other.metadata);
        state
    }

    fn update_from_part(&mut self, part: ToolPart) {
        self.common.set_tool_name_if_present(part.tool);
        self.operation = match self.common.tool_name.as_str() {
            "todoread" => TodoOperation::Read,
            _ => TodoOperation::Write,
        };

        match part.state {
            ToolStateUpdate::Pending { input } => {
                self.common.state = ToolStateStatus::Pending;
                self.update_from_raw(input, None);
            }
            ToolStateUpdate::Running {
                input,
                title,
                metadata,
            } => {
                self.common.state = ToolStateStatus::Running;
                self.common.set_title_if_present(title);
                self.update_from_raw(input, metadata);
            }
            ToolStateUpdate::Completed {
                input,
                title,
                metadata,
                output: _,
            } => {
                self.common.state = ToolStateStatus::Completed;
                self.common.set_title_if_present(title);
                self.update_from_raw(input, metadata);
            }
            ToolStateUpdate::Error {
                input,
                error: _,
                metadata,
            } => {
                self.common.state = ToolStateStatus::Error;
                self.update_from_raw(input, metadata);
            }
            ToolStateUpdate::Unknown => {}
        }
    }

    fn update_from_raw(&mut self, input: Option<Value>, metadata: Option<Value>) {
        match self.operation {
            TodoOperation::Write => {
                if let Some(input) = input {
                    if let Ok(parsed) = serde_json::from_value::<TodoWriteInput>(input) {
                        self.todos = parsed.todos;
                    }
                }
            }
            TodoOperation::Read => {
                if let Some(metadata) = metadata {
                    if let Ok(parsed) = serde_json::from_value::<TodoReadMetadata>(metadata) {
                        self.todos = parsed.todos;
                    }
                }
            }
        }
    }
}

impl ToNormalizedEntry for TodoToolState {
    fn to_normalized_entry(&self, _worktree_path: &Path) -> NormalizedEntry {
        let operation = match self.operation {
            TodoOperation::Read => "read",
            TodoOperation::Write => "write",
        }
        .to_string();
        let action_type = ActionType::TodoManagement {
            todos: self.todos.clone(),
            operation,
        };
        let content = tool_content(
            self.common.title.as_deref(),
            &self.common.tool_name,
            &action_type,
        );

        NormalizedEntry {
            timestamp: None,
            entry_type: NormalizedEntryType::ToolUse {
                tool_name: self.common.tool_name.clone(),
                action_type,
                status: self.common.tool_status(),
            },
            content,
            metadata: serde_json::to_value(ToolCallMetadata {
                tool_call_id: self.common.call_id.clone(),
            })
            .ok(),
        }
    }
}

#[derive(Debug, Clone)]
struct TaskToolState {
    common: ToolCallCommon,
    description: Option<String>,
}

impl TaskToolState {
    fn from_other(other: OtherToolState) -> Self {
        let mut state = Self {
            common: other.common,
            description: None,
        };
        state.update_from_raw(other.input);
        state
    }

    fn update_from_part(&mut self, part: ToolPart) {
        self.common.set_tool_name_if_present(part.tool);
        match part.state {
            ToolStateUpdate::Pending { input } => {
                self.common.state = ToolStateStatus::Pending;
                self.update_from_raw(input);
            }
            ToolStateUpdate::Running {
                input,
                title,
                metadata: _,
            } => {
                self.common.state = ToolStateStatus::Running;
                self.common.set_title_if_present(title);
                self.update_from_raw(input);
            }
            ToolStateUpdate::Completed {
                input,
                title,
                metadata: _,
                output: _,
            } => {
                self.common.state = ToolStateStatus::Completed;
                self.common.set_title_if_present(title);
                self.update_from_raw(input);
            }
            ToolStateUpdate::Error {
                input,
                error: _,
                metadata: _,
            } => {
                self.common.state = ToolStateStatus::Error;
                self.update_from_raw(input);
            }
            ToolStateUpdate::Unknown => {}
        }
    }

    fn update_from_raw(&mut self, input: Option<Value>) {
        if let Some(input) = input {
            if let Ok(parsed) = serde_json::from_value::<TaskInput>(input) {
                self.description = Some(parsed.description);
            }
        }
    }
}

impl ToNormalizedEntry for TaskToolState {
    fn to_normalized_entry(&self, _worktree_path: &Path) -> NormalizedEntry {
        let action_type = ActionType::TaskCreate {
            description: self.description.clone().unwrap_or_default(),
        };
        let content = tool_content(
            self.common.title.as_deref(),
            &self.common.tool_name,
            &action_type,
        );

        NormalizedEntry {
            timestamp: None,
            entry_type: NormalizedEntryType::ToolUse {
                tool_name: self.common.tool_name.clone(),
                action_type,
                status: self.common.tool_status(),
            },
            content,
            metadata: serde_json::to_value(ToolCallMetadata {
                tool_call_id: self.common.call_id.clone(),
            })
            .ok(),
        }
    }
}

#[derive(Debug, Clone)]
struct OtherToolState {
    common: ToolCallCommon,
    input: Option<Value>,
    metadata: Option<Value>,
    output: Option<String>,
    error: Option<String>,
}

impl OtherToolState {
    fn new(call_id: String) -> Self {
        Self {
            common: ToolCallCommon::new(call_id),
            input: None,
            metadata: None,
            output: None,
            error: None,
        }
    }

    fn update_from_part(&mut self, part: ToolPart) {
        self.common.set_tool_name_if_present(part.tool);

        match part.state {
            ToolStateUpdate::Pending { input } => {
                self.common.state = ToolStateStatus::Pending;
                if let Some(input) = input {
                    self.input = Some(input);
                }
            }
            ToolStateUpdate::Running {
                input,
                title,
                metadata,
            } => {
                self.common.state = ToolStateStatus::Running;
                self.common.set_title_if_present(title);
                if let Some(input) = input {
                    self.input = Some(input);
                }
                if let Some(metadata) = metadata {
                    self.metadata = Some(metadata);
                }
            }
            ToolStateUpdate::Completed {
                input,
                output,
                title,
                metadata,
            } => {
                self.common.state = ToolStateStatus::Completed;
                self.common.set_title_if_present(title);
                if let Some(input) = input {
                    self.input = Some(input);
                }
                if let Some(output) = output {
                    self.output = Some(output);
                }
                if let Some(metadata) = metadata {
                    self.metadata = Some(metadata);
                }
                self.error = None;
            }
            ToolStateUpdate::Error {
                input,
                error,
                metadata,
            } => {
                self.common.state = ToolStateStatus::Error;
                if let Some(input) = input {
                    self.input = Some(input);
                }
                if let Some(error) = error
                    && !error.trim().is_empty()
                {
                    self.error = Some(error);
                }
                if let Some(metadata) = metadata {
                    self.metadata = Some(metadata);
                }
            }
            ToolStateUpdate::Unknown => {}
        }
    }

    fn output(&self) -> Option<&str> {
        self.output.as_deref().or(self.error.as_deref())
    }

    fn arguments(&self) -> Option<Value> {
        self.input
            .as_ref()
            .and_then(|v| v.as_object().map(|_| v.clone()))
    }
}

impl ToNormalizedEntry for OtherToolState {
    fn to_normalized_entry(&self, _worktree_path: &Path) -> NormalizedEntry {
        let result = self.output().map(|o| ToolResult::markdown(o.to_string()));
        let action_type = ActionType::Tool {
            tool_name: self.common.tool_name.clone(),
            arguments: self.arguments(),
            result,
        };
        let content = tool_content(
            self.common.title.as_deref(),
            &self.common.tool_name,
            &action_type,
        );

        NormalizedEntry {
            timestamp: None,
            entry_type: NormalizedEntryType::ToolUse {
                tool_name: self.common.tool_name.clone(),
                action_type,
                status: self.common.tool_status(),
            },
            content,
            metadata: serde_json::to_value(ToolCallMetadata {
                tool_call_id: self.common.call_id.clone(),
            })
            .ok(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct BashInput {
    command: String,
}

#[derive(Debug, Deserialize)]
struct FilePathInput {
    #[serde(rename = "filePath")]
    file_path: String,
}

#[derive(Debug, Deserialize)]
struct WriteInput {
    #[serde(rename = "filePath")]
    file_path: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct WebFetchInput {
    url: String,
}

#[derive(Debug, Deserialize)]
struct SearchInput {
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    pattern: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TodoWriteInput {
    #[serde(default)]
    todos: Vec<TodoItem>,
}

#[derive(Debug, Deserialize)]
struct TodoReadMetadata {
    #[serde(default)]
    todos: Vec<TodoItem>,
}

#[derive(Debug, Deserialize)]
struct TaskInput {
    description: String,
}

#[derive(Debug, Deserialize)]
struct SdkEventEnvelope {
    #[serde(rename = "type")]
    type_: String,
    #[serde(default)]
    properties: Value,
}

#[derive(Debug)]
enum SdkEvent {
    MessageUpdated(MessageUpdatedEvent),
    MessagePartUpdated(MessagePartUpdatedEvent),
    SessionError(SessionErrorEvent),
}

impl SdkEvent {
    fn parse(value: &Value) -> Option<Self> {
        let envelope = serde_json::from_value::<SdkEventEnvelope>(value.clone()).ok()?;
        match envelope.type_.as_str() {
            "message.updated" => serde_json::from_value(envelope.properties)
                .ok()
                .map(SdkEvent::MessageUpdated),
            "message.part.updated" => serde_json::from_value(envelope.properties)
                .ok()
                .map(SdkEvent::MessagePartUpdated),
            "session.error" => serde_json::from_value(envelope.properties)
                .ok()
                .map(SdkEvent::SessionError),
            _ => None,
        }
    }
}

#[derive(Debug, Deserialize)]
struct MessageUpdatedEvent {
    info: MessageInfo,
}

#[derive(Debug, Deserialize)]
struct MessageInfo {
    id: String,
    role: MessageRole,
    #[serde(default)]
    model: Option<MessageModelInfo>,
    #[serde(rename = "providerID", default)]
    provider_id: Option<String>,
    #[serde(rename = "modelID", default)]
    model_id: Option<String>,
}

impl MessageInfo {
    fn provider_id(&self) -> Option<&str> {
        self.model
            .as_ref()
            .map(|m| m.provider_id.as_str())
            .or_else(|| self.provider_id.as_deref())
    }

    fn model_id(&self) -> Option<&str> {
        self.model
            .as_ref()
            .map(|m| m.model_id.as_str())
            .or_else(|| self.model_id.as_deref())
    }
}

#[derive(Debug, Deserialize)]
struct MessageModelInfo {
    #[serde(rename = "providerID", alias = "providerId")]
    provider_id: String,
    #[serde(rename = "modelID", alias = "modelId")]
    model_id: String,
}

#[derive(Debug, Deserialize)]
struct MessagePartUpdatedEvent {
    part: Part,
    #[serde(default)]
    delta: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum Part {
    #[serde(rename = "text")]
    Text(TextPart),
    #[serde(rename = "reasoning")]
    Reasoning(ReasoningPart),
    #[serde(rename = "tool")]
    Tool(ToolPart),
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct TextPart {
    #[serde(rename = "messageID")]
    message_id: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct ReasoningPart {
    #[serde(rename = "messageID")]
    message_id: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct ToolPart {
    #[serde(rename = "messageID")]
    message_id: String,
    #[serde(rename = "callID")]
    call_id: String,
    #[serde(default)]
    tool: String,
    state: ToolStateUpdate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolStateStatus {
    Pending,
    Running,
    Completed,
    Error,
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "status", rename_all = "lowercase")]
enum ToolStateUpdate {
    Pending {
        #[serde(default)]
        input: Option<Value>,
    },
    Running {
        #[serde(default)]
        input: Option<Value>,
        #[serde(default)]
        title: Option<String>,
        #[serde(default)]
        metadata: Option<Value>,
    },
    Completed {
        #[serde(default)]
        input: Option<Value>,
        #[serde(default)]
        output: Option<String>,
        #[serde(default)]
        title: Option<String>,
        #[serde(default)]
        metadata: Option<Value>,
    },
    Error {
        #[serde(default)]
        input: Option<Value>,
        #[serde(default)]
        error: Option<String>,
        #[serde(default)]
        metadata: Option<Value>,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
struct SessionErrorEvent {
    #[serde(default)]
    error: Option<SdkError>,
}

#[derive(Debug)]
struct SdkError {
    raw: Value,
}

impl SdkError {
    fn kind(&self) -> &str {
        self.raw
            .get("name")
            .or_else(|| self.raw.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("unknown")
    }

    fn message(&self) -> Option<String> {
        self.raw
            .pointer("/data/message")
            .or_else(|| self.raw.get("message"))
            .and_then(Value::as_str)
            .map(|s| s.to_string())
    }
}

impl<'de> Deserialize<'de> for SdkError {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = Value::deserialize(deserializer)?;
        Ok(Self { raw })
    }
}

fn make_relative_path(path: &str, worktree_path: &Path) -> String {
    make_path_relative(path, &worktree_path.to_string_lossy())
}

fn tool_content(title: Option<&str>, tool_name: &str, action_type: &ActionType) -> String {
    let content = match action_type {
        ActionType::CommandRun { command, .. } => command.clone(),
        ActionType::FileRead { path } => path.clone(),
        ActionType::FileEdit { path, .. } => path.clone(),
        ActionType::Search { query } => query.clone(),
        ActionType::WebFetch { url } => url.clone(),
        _ => "".to_string(),
    }
    .trim()
    .to_string();

    if !content.is_empty() {
        content
    } else {
        title.unwrap_or(tool_name).to_string()
    }
}

fn extract_diff_from_metadata(metadata: &Value) -> Option<&str> {
    metadata.get("diff").and_then(Value::as_str).or_else(|| {
        metadata
            .get("results")
            .and_then(Value::as_array)
            .and_then(|results| results.last())
            .and_then(|last| last.get("diff"))
            .and_then(Value::as_str)
    })
}
