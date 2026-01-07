use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use async_trait::async_trait;
use executors::{
    approvals::{ExecutorApprovalError, ExecutorApprovalService},
    env::ExecutionEnv,
    executors::{ExecutorExitResult, StandardCodingAgentExecutor, opencode::Opencode},
    logs::{
        NormalizedEntry, NormalizedEntryType, utils::patch::extract_normalized_entry_from_patch,
    },
};
use futures::StreamExt;
use serde_json::Value;
use tokio_util::io::ReaderStream;
use workspace_utils::{approvals::ApprovalStatus, msg_store::MsgStore};

#[derive(Debug, Default)]
struct ReadOnlyApprovalService;

#[async_trait]
impl ExecutorApprovalService for ReadOnlyApprovalService {
    async fn request_tool_approval(
        &self,
        tool_name: &str,
        _tool_input: Value,
        _tool_call_id: &str,
    ) -> Result<ApprovalStatus, ExecutorApprovalError> {
        let tool = tool_name.trim().to_lowercase();

        let allow = matches!(
            tool.as_str(),
            // file read/search
            "read" | "grep" | "glob"
            // web/network read-only
            | "webfetch" | "websearch" | "codesearch"
            // internal-only, should not mutate repo
            | "todoread" | "todowrite"
        );

        if allow {
            return Ok(ApprovalStatus::Approved);
        }

        Ok(ApprovalStatus::Denied {
            reason: Some(format!(
                "Debug helper: denying `{tool_name}` (read-only run)"
            )),
        })
    }
}

fn build_read_only_prompt() -> String {
    [
        "Read-only debugging run.",
        "Summarize what this repository does in 5 bullet points.",
        "You may ONLY use read-only tools (read/glob/grep/webfetch/websearch).",
        "Do NOT run bash/commands. Do NOT edit/write/patch any files.",
        "If a tool is denied, continue without it.",
    ]
    .join("\n")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let current_dir = std::env::current_dir()?;
    let worktree_path = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| current_dir.to_string_lossy().into_owned().to_string()),
    );

    let mut env = ExecutionEnv::default();
    env.insert(
        "OPENCODE_PERMISSION",
        r#"{"read":"ask","grep":"ask","glob":"ask","webfetch":"ask","websearch":"ask","codesearch":"ask","edit":"ask","write":"ask","multiedit":"ask","bash":"ask","external_directory":"ask","doom_loop":"ask"}"#,
    );

    let mut opencode = Opencode {
        append_prompt: Default::default(),
        model: Some("google-vertex/gemini-3-pro-preview".to_string()),
        mode: Some("build".to_string()),
        auto_approve: false,
        cmd: Default::default(),
        approvals: None,
    };
    opencode.use_approvals(Arc::new(ReadOnlyApprovalService::default()));

    let msg_store = Arc::new(MsgStore::new());
    opencode.normalize_logs(msg_store.clone(), &worktree_path);

    let prompt = build_read_only_prompt();
    let mut spawned = opencode.spawn(&worktree_path, &prompt, &env).await?;

    let stdout = spawned
        .child
        .inner()
        .stdout
        .take()
        .expect("opencode child missing stdout");
    let stderr = spawned
        .child
        .inner()
        .stderr
        .take()
        .expect("opencode child missing stderr");

    {
        let store = msg_store.clone();
        tokio::spawn(async move {
            let mut stream = ReaderStream::new(stdout);
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(bytes) => store.push_stdout(String::from_utf8_lossy(&bytes).into_owned()),
                    Err(err) => store.push_stderr(format!("stdout read error: {err}")),
                }
            }
        });
    }

    {
        let store = msg_store.clone();
        tokio::spawn(async move {
            let mut stream = ReaderStream::new(stderr);
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(bytes) => store.push_stderr(String::from_utf8_lossy(&bytes).into_owned()),
                    Err(err) => store.push_stderr(format!("stderr read error: {err}")),
                }
            }
        });
    }

    let exit_result = match spawned.exit_signal.take() {
        Some(rx) => tokio::time::timeout(Duration::from_secs(180), rx)
            .await
            .ok()
            .and_then(|res| res.ok())
            .unwrap_or(ExecutorExitResult::Failure),
        None => ExecutorExitResult::Success,
    };

    let _ = spawned.child.kill().await;
    let _ = spawned.child.wait().await;

    msg_store.push_finished();
    tokio::time::sleep(Duration::from_millis(200)).await;

    let (session_id, entries, errors) = collect_normalized_entries(&msg_store);
    let sdk_event_counts = collect_sdk_event_types(&msg_store);

    eprintln!(
        "session_id: {}",
        session_id.unwrap_or_else(|| "<none>".to_string())
    );
    eprintln!("exit_result: {exit_result:?}");
    eprintln!("entries: {}", entries.len());
    eprintln!(
        "counts: assistant={} thinking={} tool={} error={} feedback={} system={}",
        count_type(&entries, |t| matches!(
            t,
            NormalizedEntryType::AssistantMessage
        )),
        count_type(&entries, |t| matches!(t, NormalizedEntryType::Thinking)),
        count_type(&entries, |t| matches!(
            t,
            NormalizedEntryType::ToolUse { .. }
        )),
        count_type(&entries, |t| matches!(
            t,
            NormalizedEntryType::ErrorMessage { .. }
        )),
        count_type(&entries, |t| matches!(
            t,
            NormalizedEntryType::UserFeedback { .. }
        )),
        count_type(&entries, |t| matches!(
            t,
            NormalizedEntryType::SystemMessage
        )),
    );

    if let Some(sys) = entries
        .first()
        .filter(|e| matches!(e.entry_type, NormalizedEntryType::SystemMessage))
        .or_else(|| {
            entries
                .iter()
                .find(|e| matches!(e.entry_type, NormalizedEntryType::SystemMessage))
        })
        .map(|e| e.content.trim().to_string())
        .filter(|s| !s.is_empty())
    {
        eprintln!("\nSystem message:");
        eprintln!("{sys}");
    }

    if !errors.is_empty() {
        eprintln!("\nNormalized errors:");
        for err in errors {
            eprintln!("- {}", err.replace('\n', "\\n"));
        }
    }

    if !sdk_event_counts.is_empty() {
        eprintln!("\nSDK event types (top 12):");
        let mut counts: Vec<(String, usize)> = sdk_event_counts.into_iter().collect();
        counts.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        for (ty, n) in counts.into_iter().take(12) {
            eprintln!("- {ty}: {n}");
        }
    }

    if entries.is_empty() {
        eprintln!("\nSDK message event debug (since entries=0):");
        dump_message_events(&msg_store);
        eprintln!("\nstderr tail (last 30 lines):");
        dump_stderr_tail(&msg_store, 30);
    }

    if let Some(last_assistant) = entries
        .iter()
        .rev()
        .find(|e| matches!(e.entry_type, NormalizedEntryType::AssistantMessage))
        .map(|e| e.content.trim().to_string())
        .filter(|s| !s.is_empty())
    {
        eprintln!("\nLast assistant message (trimmed):");
        eprintln!("{last_assistant}");
    }

    Ok(())
}

fn count_type(entries: &[NormalizedEntry], f: impl Fn(&NormalizedEntryType) -> bool) -> usize {
    entries.iter().filter(|e| f(&e.entry_type)).count()
}

fn collect_normalized_entries(
    msg_store: &Arc<MsgStore>,
) -> (Option<String>, Vec<NormalizedEntry>, Vec<String>) {
    let mut session_id: Option<String> = None;
    let mut by_index: HashMap<usize, NormalizedEntry> = HashMap::new();

    for msg in msg_store.get_history() {
        match msg {
            workspace_utils::log_msg::LogMsg::SessionId(id) => {
                if session_id.is_none() {
                    session_id = Some(id);
                }
            }
            workspace_utils::log_msg::LogMsg::JsonPatch(patch) => {
                if let Some((idx, entry)) = extract_normalized_entry_from_patch(&patch) {
                    by_index.insert(idx, entry);
                }
            }
            _ => {}
        }
    }

    let mut entries: Vec<(usize, NormalizedEntry)> = by_index.into_iter().collect();
    entries.sort_by_key(|(idx, _)| *idx);
    let entries: Vec<NormalizedEntry> = entries.into_iter().map(|(_, e)| e).collect();

    let errors = entries
        .iter()
        .filter_map(|e| match e.entry_type {
            NormalizedEntryType::ErrorMessage { .. } => Some(e.content.clone()),
            _ => None,
        })
        .collect();

    (session_id, entries, errors)
}

fn collect_sdk_event_types(msg_store: &Arc<MsgStore>) -> HashMap<String, usize> {
    let mut stdout = String::new();
    for msg in msg_store.get_history() {
        if let workspace_utils::log_msg::LogMsg::Stdout(chunk) = msg {
            stdout.push_str(&chunk);
        }
    }

    let mut counts: HashMap<String, usize> = HashMap::new();
    for line in stdout.lines() {
        let Ok(value) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };

        if value.get("type").and_then(Value::as_str) != Some("sdk_event") {
            continue;
        }

        let Some(event_type) = value
            .get("event")
            .and_then(|e| e.get("type"))
            .and_then(Value::as_str)
        else {
            continue;
        };

        *counts.entry(event_type.to_string()).or_insert(0) += 1;
    }

    counts
}

fn dump_message_events(msg_store: &Arc<MsgStore>) {
    let mut stdout = String::new();
    for msg in msg_store.get_history() {
        if let workspace_utils::log_msg::LogMsg::Stdout(chunk) = msg {
            stdout.push_str(&chunk);
        }
    }

    for line in stdout.lines() {
        let Ok(value) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };

        if value.get("type").and_then(Value::as_str) != Some("sdk_event") {
            continue;
        }

        let Some(event) = value.get("event") else {
            continue;
        };

        let Some(event_type) = event.get("type").and_then(Value::as_str) else {
            continue;
        };

        match event_type {
            "session.status" => {
                let status = event
                    .pointer("/properties/status/type")
                    .and_then(Value::as_str)
                    .unwrap_or("<none>");
                eprintln!("- session.status type={status}");
            }
            "message.updated" => {
                let info = event
                    .pointer("/properties/info")
                    .cloned()
                    .unwrap_or(Value::Null);
                let id = info.get("id").and_then(Value::as_str).unwrap_or("<none>");
                let role = info.get("role").and_then(Value::as_str).unwrap_or("<none>");
                let session_id = info
                    .get("sessionID")
                    .or_else(|| info.get("sessionId"))
                    .and_then(Value::as_str)
                    .unwrap_or("<none>");
                eprintln!("- message.updated id={id} role={role} session={session_id}");
            }
            "message.part.updated" => {
                let part = event
                    .pointer("/properties/part")
                    .cloned()
                    .unwrap_or(Value::Null);
                let part_type = part.get("type").and_then(Value::as_str).unwrap_or("<none>");
                let msg_id = part
                    .get("messageID")
                    .or_else(|| part.get("messageId"))
                    .and_then(Value::as_str)
                    .unwrap_or("<none>");
                let call_id = part
                    .get("callID")
                    .or_else(|| part.get("callId"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let tool = part.get("tool").and_then(Value::as_str).unwrap_or("");
                let delta_len = event
                    .pointer("/properties/delta")
                    .and_then(Value::as_str)
                    .map(|s| s.len())
                    .unwrap_or(0);

                eprintln!(
                    "- message.part.updated messageID={msg_id} part_type={part_type} tool={tool} callID={call_id} delta_len={delta_len}"
                );
            }
            _ => {}
        }
    }
}

fn dump_stderr_tail(msg_store: &Arc<MsgStore>, max_lines: usize) {
    let mut stderr = String::new();
    for msg in msg_store.get_history() {
        if let workspace_utils::log_msg::LogMsg::Stderr(chunk) = msg {
            stderr.push_str(&chunk);
        }
    }

    let lines: Vec<&str> = stderr.lines().collect();
    let start = lines.len().saturating_sub(max_lines);
    for line in &lines[start..] {
        eprintln!("- {}", line);
    }
}
