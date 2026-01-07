use std::{path::Path, sync::Arc, time::Duration};

use async_trait::async_trait;
use command_group::AsyncCommandGroup;
use derivative::Derivative;
use futures::StreamExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio_util::io::ReaderStream;
use ts_rs::TS;
use workspace_utils::msg_store::MsgStore;

use crate::{
    approvals::ExecutorApprovalService,
    command::{CmdOverrides, CommandBuilder, apply_overrides},
    env::ExecutionEnv,
    executors::{
        AppendPrompt, AvailabilityInfo, ExecutorError, ExecutorExitResult, SpawnedChild,
        StandardCodingAgentExecutor,
    },
    stdout_dup::{create_stderr_pipe_writer, create_stdout_pipe_writer},
};

mod normalize_logs;
mod sdk;
mod types;

use sdk::{LogWriter, RunConfig, run_session};

#[derive(Derivative, Clone, Serialize, Deserialize, TS, JsonSchema)]
#[derivative(Debug, PartialEq)]
pub struct Opencode {
    #[serde(default)]
    pub append_prompt: AppendPrompt,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "agent")]
    pub mode: Option<String>,
    /// Auto-approve agent actions
    #[serde(default = "default_to_true")]
    pub auto_approve: bool,
    #[serde(flatten)]
    pub cmd: CmdOverrides,
    #[serde(skip)]
    #[ts(skip)]
    #[derivative(Debug = "ignore", PartialEq = "ignore")]
    pub approvals: Option<Arc<dyn ExecutorApprovalService>>,
}

impl Opencode {
    fn build_command_builder(&self) -> CommandBuilder {
        let builder = CommandBuilder::new("npx --yes opencode-ai@1.1.3")
            // Pass hostname/port as separate args so OpenCode treats them as explicitly set
            // (it checks `process.argv.includes(\"--port\")` / `\"--hostname\"`).
            .extend_params(["serve", "--hostname", "127.0.0.1", "--port", "0"]);
        apply_overrides(builder, &self.cmd)
    }

    async fn spawn_inner(
        &self,
        current_dir: &Path,
        prompt: &str,
        resume_session: Option<&str>,
        env: &ExecutionEnv,
    ) -> Result<SpawnedChild, ExecutorError> {
        let combined_prompt = self.append_prompt.combine_prompt(prompt);

        let command_parts = self.build_command_builder().build_initial()?;
        let (program_path, args) = command_parts.into_resolved().await?;

        let mut command = Command::new(program_path);
        command
            .kill_on_drop(true)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .current_dir(current_dir)
            .args(&args)
            .env("CI", "1")
            .env("NODE_NO_WARNINGS", "1")
            .env("NO_COLOR", "1");

        env.clone()
            .with_profile(&self.cmd)
            .apply_to_command(&mut command);

        let mut child = command.group_spawn()?;
        let server_stderr = child.inner().stderr.take().ok_or_else(|| {
            ExecutorError::Io(std::io::Error::other(
                "OpenCode server missing stderr (needed for startup diagnostics)",
            ))
        })?;
        let server_stdout = child.inner().stdout.take().ok_or_else(|| {
            ExecutorError::Io(std::io::Error::other(
                "OpenCode server missing stdout (needed to parse listening URL)",
            ))
        })?;

        // Keep a stderr handle for the supervisor to read (even though we drain the real server
        // stderr internally to avoid startup deadlocks).
        let _ = create_stderr_pipe_writer(&mut child)?;

        let stdout = create_stdout_pipe_writer(&mut child)?;
        let log_writer = LogWriter::new(stdout);

        let (exit_signal_tx, exit_signal_rx) = tokio::sync::oneshot::channel();

        let directory = current_dir.to_string_lossy().to_string();
        let base_url = wait_for_server_url(server_stdout, Some(server_stderr)).await?;
        let approvals = if self.auto_approve {
            None
        } else {
            self.approvals.clone()
        };

        let config = RunConfig {
            base_url,
            directory,
            prompt: combined_prompt,
            resume_session_id: resume_session.map(|s| s.to_string()),
            model: self.model.clone(),
            agent: self.mode.clone(),
            approvals,
            auto_approve: self.auto_approve,
        };

        tokio::spawn(async move {
            let result = run_session(config, log_writer.clone()).await;
            let exit_result = match result {
                Ok(()) => ExecutorExitResult::Success,
                Err(err) => {
                    let _ = log_writer
                        .log_error(format!("OpenCode executor error: {err}"))
                        .await;
                    ExecutorExitResult::Failure
                }
            };
            let _ = exit_signal_tx.send(exit_result);
        });

        Ok(SpawnedChild {
            child,
            exit_signal: Some(exit_signal_rx),
            interrupt_sender: None,
        })
    }
}

async fn wait_for_server_url(
    stdout: tokio::process::ChildStdout,
    stderr: Option<tokio::process::ChildStderr>,
) -> Result<String, ExecutorError> {
    #[derive(Clone, Copy)]
    enum StreamKind {
        Stdout,
        Stderr,
    }

    const URL_PREFIX: &str = "opencode server listening on ";
    const OUTPUT_TAIL_MAX: usize = 16 * 1024;
    const SEARCH_BUF_MAX: usize = 64 * 1024;
    const STARTUP_TIMEOUT: Duration = Duration::from_secs(120);

    fn truncate_front_to_max(buf: &mut String, max: usize) {
        if buf.len() <= max {
            return;
        }
        let excess = buf.len() - max;
        let mut cut_at = 0;
        for (idx, _) in buf.char_indices() {
            if idx >= excess {
                cut_at = idx;
                break;
            }
        }
        buf.drain(..cut_at);
    }

    fn try_extract_url(buf: &str) -> Option<String> {
        let idx = buf.rfind(URL_PREFIX)?;
        let after = &buf[idx + URL_PREFIX.len()..];
        let end = after.find(|c: char| c.is_whitespace())?;
        let url = after[..end].trim();
        if url.starts_with("http://") || url.starts_with("https://") {
            Some(url.to_string())
        } else {
            None
        }
    }

    let (tx, mut rx) = mpsc::unbounded_channel::<(StreamKind, String)>();

    // Drain stdout in a background task (needed to keep the server pipe open on all platforms),
    // while also forwarding chunks for URL detection.
    {
        let tx = tx.clone();
        tokio::spawn(async move {
            let mut stream = ReaderStream::new(stdout);
            while let Some(res) = stream.next().await {
                match res {
                    Ok(bytes) => {
                        let text = String::from_utf8_lossy(&bytes).into_owned();
                        let _ = tx.send((StreamKind::Stdout, text));
                    }
                    Err(err) => {
                        let _ = tx.send((StreamKind::Stdout, format!("[read error] {err}")));
                        break;
                    }
                }
            }
        });
    }

    // Drain stderr too (npm/npx progress output is often stderr on Windows).
    if let Some(stderr) = stderr {
        let tx = tx.clone();
        tokio::spawn(async move {
            let mut stream = ReaderStream::new(stderr);
            while let Some(res) = stream.next().await {
                match res {
                    Ok(bytes) => {
                        let text = String::from_utf8_lossy(&bytes).into_owned();
                        let _ = tx.send((StreamKind::Stderr, text));
                    }
                    Err(err) => {
                        let _ = tx.send((StreamKind::Stderr, format!("[read error] {err}")));
                        break;
                    }
                }
            }
        });
    }

    drop(tx);

    let deadline = tokio::time::Instant::now() + STARTUP_TIMEOUT;
    let mut search_buf = String::new();
    let mut output_tail = String::new();

    let deadline_sleep = tokio::time::sleep_until(deadline);
    tokio::pin!(deadline_sleep);

    loop {
        tokio::select! {
            _ = &mut deadline_sleep => {
                let tail = output_tail.trim().to_string();
                return Err(ExecutorError::Io(std::io::Error::other(format!(
                    "Timed out waiting for OpenCode server to print listening URL.\nServer output tail:\n{tail}"
                ))));
            }
            chunk = rx.recv() => match chunk {
                Some((kind, text)) => {
                    search_buf.push_str(&text);
                    truncate_front_to_max(&mut search_buf, SEARCH_BUF_MAX);

                    match kind {
                        StreamKind::Stdout => output_tail.push_str("[stdout] "),
                        StreamKind::Stderr => output_tail.push_str("[stderr] "),
                    }
                    output_tail.push_str(&text);
                    truncate_front_to_max(&mut output_tail, OUTPUT_TAIL_MAX);

                    if let Some(url) = try_extract_url(&search_buf) {
                        return Ok(url);
                    }
                }
                None => {
                    let tail = output_tail.trim().to_string();
                    return Err(ExecutorError::Io(std::io::Error::other(format!(
                        "OpenCode server exited before printing listening URL.\nServer output tail:\n{tail}"
                    ))));
                }
            }
        }
    }
}

#[async_trait]
impl StandardCodingAgentExecutor for Opencode {
    fn use_approvals(&mut self, approvals: Arc<dyn ExecutorApprovalService>) {
        self.approvals = Some(approvals);
    }

    async fn spawn(
        &self,
        current_dir: &Path,
        prompt: &str,
        env: &ExecutionEnv,
    ) -> Result<SpawnedChild, ExecutorError> {
        let env = setup_approvals_env(self.auto_approve, env);
        self.spawn_inner(current_dir, prompt, None, &env).await
    }

    async fn spawn_follow_up(
        &self,
        current_dir: &Path,
        prompt: &str,
        session_id: &str,
        env: &ExecutionEnv,
    ) -> Result<SpawnedChild, ExecutorError> {
        let env = setup_approvals_env(self.auto_approve, env);
        self.spawn_inner(current_dir, prompt, Some(session_id), &env)
            .await
    }

    fn normalize_logs(&self, msg_store: Arc<MsgStore>, worktree_path: &Path) {
        normalize_logs::normalize_logs(msg_store, worktree_path);
    }

    fn default_mcp_config_path(&self) -> Option<std::path::PathBuf> {
        #[cfg(unix)]
        {
            xdg::BaseDirectories::with_prefix("opencode").get_config_file("opencode.json")
        }
        #[cfg(not(unix))]
        {
            dirs::config_dir().map(|config| config.join("opencode").join("opencode.json"))
        }
    }

    fn get_availability_info(&self) -> AvailabilityInfo {
        let mcp_config_found = self
            .default_mcp_config_path()
            .map(|p| p.exists())
            .unwrap_or(false);

        let installation_indicator_found = dirs::config_dir()
            .map(|config| config.join("opencode").exists())
            .unwrap_or(false);

        if mcp_config_found || installation_indicator_found {
            AvailabilityInfo::InstallationFound
        } else {
            AvailabilityInfo::NotFound
        }
    }
}

fn default_to_true() -> bool {
    true
}

fn setup_approvals_env(auto_approve: bool, env: &ExecutionEnv) -> ExecutionEnv {
    let mut env = env.clone();
    if !auto_approve && !env.contains_key("OPENCODE_PERMISSION") {
        env.insert("OPENCODE_PERMISSION", r#"{"edit": "ask", "bash": "ask", "webfetch": "ask", "doom_loop": "ask", "external_directory": "ask"}"#);
    }
    env
}
