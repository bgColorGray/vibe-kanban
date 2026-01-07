use std::{
    collections::HashSet,
    io,
    sync::{Arc, Once},
    time::Duration,
};

use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::{
    io::{AsyncWrite, AsyncWriteExt, BufWriter},
    sync::{Mutex, mpsc},
};
use workspace_utils::approvals::ApprovalStatus;

use super::types::OpencodeExecutorEvent;
use crate::{
    approvals::{ExecutorApprovalError, ExecutorApprovalService},
    executors::ExecutorError,
};

fn ensure_rustls_crypto_provider() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if let Err(err) = rustls::crypto::aws_lc_rs::default_provider().install_default() {
            tracing::debug!("rustls crypto provider install failed: {err:?}");
        }
    });
}

#[derive(Clone)]
pub struct LogWriter {
    writer: Arc<Mutex<BufWriter<Box<dyn AsyncWrite + Send + Unpin>>>>,
}

impl LogWriter {
    pub fn new(writer: impl AsyncWrite + Send + Unpin + 'static) -> Self {
        Self {
            writer: Arc::new(Mutex::new(BufWriter::new(Box::new(writer)))),
        }
    }

    pub async fn log_event(&self, event: &OpencodeExecutorEvent) -> Result<(), ExecutorError> {
        let raw =
            serde_json::to_string(event).map_err(|err| ExecutorError::Io(io::Error::other(err)))?;
        self.log_raw(&raw).await
    }

    pub async fn log_error(&self, message: String) -> Result<(), ExecutorError> {
        self.log_event(&OpencodeExecutorEvent::Error { message })
            .await
    }

    async fn log_raw(&self, raw: &str) -> Result<(), ExecutorError> {
        let mut guard = self.writer.lock().await;
        guard
            .write_all(raw.as_bytes())
            .await
            .map_err(ExecutorError::Io)?;
        guard.write_all(b"\n").await.map_err(ExecutorError::Io)?;
        guard.flush().await.map_err(ExecutorError::Io)?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct RunConfig {
    pub base_url: String,
    pub directory: String,
    pub prompt: String,
    pub resume_session_id: Option<String>,
    pub model: Option<String>,
    pub agent: Option<String>,
    pub approvals: Option<Arc<dyn ExecutorApprovalService>>,
    pub auto_approve: bool,
}

#[derive(Debug, Deserialize)]
struct HealthResponse {
    healthy: bool,
    version: String,
}

#[derive(Debug, Deserialize)]
struct SessionResponse {
    id: String,
}

#[derive(Debug, Serialize)]
struct PromptRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<ModelSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    agent: Option<String>,
    parts: Vec<TextPartInput>,
}

#[derive(Debug, Serialize, Clone)]
struct ModelSpec {
    #[serde(rename = "providerID")]
    provider_id: String,
    #[serde(rename = "modelID")]
    model_id: String,
}

#[derive(Debug, Serialize)]
struct TextPartInput {
    r#type: &'static str,
    text: String,
}

#[derive(Debug, Clone)]
enum ControlEvent {
    Idle,
    AuthRequired { message: String },
    SessionError { message: String },
    Disconnected,
}

pub async fn run_session(config: RunConfig, log_writer: LogWriter) -> Result<(), ExecutorError> {
    ensure_rustls_crypto_provider();
    let client = reqwest::Client::builder()
        .default_headers(build_default_headers(&config.directory))
        .build()
        .map_err(|err| ExecutorError::Io(io::Error::other(err)))?;

    wait_for_health(&client, &config.base_url).await?;

    let session_id = match config.resume_session_id.as_deref() {
        Some(existing) => {
            fork_session(&client, &config.base_url, &config.directory, existing).await?
        }
        None => create_session(&client, &config.base_url, &config.directory).await?,
    };

    log_writer
        .log_event(&OpencodeExecutorEvent::SessionStart {
            session_id: session_id.clone(),
        })
        .await?;

    let model = config.model.as_deref().and_then(parse_model);

    let (control_tx, mut control_rx) = mpsc::unbounded_channel::<ControlEvent>();

    // Establish the event stream connection before sending the prompt so we don't miss early
    // `message.updated` events (important for log normalization).
    let event_resp = connect_event_stream(&client, &config.base_url, &config.directory, None).await?;
    tokio::spawn(spawn_event_listener(
        client.clone(),
        config.base_url.clone(),
        config.directory.clone(),
        session_id.clone(),
        event_resp,
        log_writer.clone(),
        config.approvals.clone(),
        config.auto_approve,
        control_tx,
    ));

    run_prompt_with_control(
        &client,
        &config.base_url,
        &config.directory,
        &session_id,
        &config.prompt,
        model.clone(),
        config.agent.clone(),
        &mut control_rx,
    )
    .await?;

    log_writer.log_event(&OpencodeExecutorEvent::Done).await?;

    Ok(())
}

fn build_default_headers(directory: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    if let Ok(value) = HeaderValue::from_str(directory) {
        headers.insert("x-opencode-directory", value);
    }
    headers
}

async fn run_prompt_with_control(
    client: &reqwest::Client,
    base_url: &str,
    directory: &str,
    session_id: &str,
    prompt_text: &str,
    model: Option<ModelSpec>,
    agent: Option<String>,
    control_rx: &mut mpsc::UnboundedReceiver<ControlEvent>,
) -> Result<(), ExecutorError> {
    let mut idle_seen = false;
    let mut session_error: Option<String> = None;

    let mut prompt_fut = Box::pin(prompt(
        client,
        base_url,
        directory,
        session_id,
        prompt_text,
        model,
        agent,
    ));

    let prompt_result = loop {
        tokio::select! {
            res = &mut prompt_fut => break res,
            event = control_rx.recv() => match event {
                Some(ControlEvent::AuthRequired { message }) => return Err(ExecutorError::AuthRequired(message)),
                Some(ControlEvent::SessionError { message }) => {
                    match &mut session_error {
                        Some(existing) => {
                            existing.push('\n');
                            existing.push_str(&message);
                        }
                        None => session_error = Some(message),
                    }
                }
                Some(ControlEvent::Disconnected) => {
                    return Err(ExecutorError::Io(io::Error::other("OpenCode event stream disconnected while prompt was running")));
                }
                Some(ControlEvent::Idle) => idle_seen = true,
                None => {}
            }
        }
    };

    prompt_result?;

    if !idle_seen {
        // The OpenCode server streams events independently; wait for `session.idle` so we capture
        // tail updates reliably (e.g. final tool completion events).
        loop {
            match control_rx.recv().await {
                Some(ControlEvent::Idle) => break,
                Some(ControlEvent::AuthRequired { message }) => {
                    return Err(ExecutorError::AuthRequired(message));
                }
                Some(ControlEvent::SessionError { message }) => {
                    match &mut session_error {
                        Some(existing) => {
                            existing.push('\n');
                            existing.push_str(&message);
                        }
                        None => session_error = Some(message),
                    }
                }
                Some(ControlEvent::Disconnected) => {
                    return Err(ExecutorError::Io(io::Error::other(
                        "OpenCode event stream disconnected while waiting for session to go idle",
                    )));
                }
                None => break,
            }
        }
    }

    if let Some(message) = session_error {
        return Err(ExecutorError::Io(io::Error::other(message)));
    }

    Ok(())
}

async fn wait_for_health(client: &reqwest::Client, base_url: &str) -> Result<(), ExecutorError> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    let mut last_err: Option<String> = None;

    loop {
        if tokio::time::Instant::now() > deadline {
            return Err(ExecutorError::Io(io::Error::other(format!(
                "Timed out waiting for OpenCode server health: {}",
                last_err.unwrap_or_else(|| "unknown error".to_string())
            ))));
        }

        let resp = client.get(format!("{base_url}/global/health")).send().await;
        match resp {
            Ok(resp) => {
                if !resp.status().is_success() {
                    last_err = Some(format!("HTTP {}", resp.status()));
                } else if let Ok(body) = resp.json::<HealthResponse>().await {
                    if body.healthy {
                        return Ok(());
                    }
                    last_err = Some(format!("unhealthy server (version {})", body.version));
                } else {
                    last_err = Some("failed to parse health response".to_string());
                }
            }
            Err(err) => {
                last_err = Some(err.to_string());
            }
        }

        tokio::time::sleep(Duration::from_millis(150)).await;
    }
}

async fn create_session(
    client: &reqwest::Client,
    base_url: &str,
    directory: &str,
) -> Result<String, ExecutorError> {
    let resp = client
        .post(format!("{base_url}/session"))
        .query(&[("directory", directory)])
        .json(&serde_json::json!({}))
        .send()
        .await
        .map_err(|err| ExecutorError::Io(io::Error::other(err)))?;

    if !resp.status().is_success() {
        return Err(ExecutorError::Io(io::Error::other(format!(
            "OpenCode session.create failed: HTTP {}",
            resp.status()
        ))));
    }

    let session = resp
        .json::<SessionResponse>()
        .await
        .map_err(|err| ExecutorError::Io(io::Error::other(err)))?;
    Ok(session.id)
}

async fn fork_session(
    client: &reqwest::Client,
    base_url: &str,
    directory: &str,
    session_id: &str,
) -> Result<String, ExecutorError> {
    let resp = client
        .post(format!("{base_url}/session/{session_id}/fork"))
        .query(&[("directory", directory)])
        .json(&serde_json::json!({}))
        .send()
        .await
        .map_err(|err| ExecutorError::Io(io::Error::other(err)))?;

    if !resp.status().is_success() {
        return Err(ExecutorError::Io(io::Error::other(format!(
            "OpenCode session.fork failed: HTTP {}",
            resp.status()
        ))));
    }

    let session = resp
        .json::<SessionResponse>()
        .await
        .map_err(|err| ExecutorError::Io(io::Error::other(err)))?;
    Ok(session.id)
}

async fn prompt(
    client: &reqwest::Client,
    base_url: &str,
    directory: &str,
    session_id: &str,
    prompt: &str,
    model: Option<ModelSpec>,
    agent: Option<String>,
) -> Result<(), ExecutorError> {
    let req = PromptRequest {
        model,
        agent,
        parts: vec![TextPartInput {
            r#type: "text",
            text: prompt.to_string(),
        }],
    };

    let resp = client
        .post(format!("{base_url}/session/{session_id}/message"))
        .query(&[("directory", directory)])
        .json(&req)
        .send()
        .await
        .map_err(|err| ExecutorError::Io(io::Error::other(err)))?;

    let status = resp.status();
    let body = resp
        .text()
        .await
        .map_err(|err| ExecutorError::Io(io::Error::other(err)))?;

    // The OpenCode server uses streaming responses and may set the HTTP status early; validate
    // success using the response body shape as well.
    if !status.is_success() {
        return Err(ExecutorError::Io(io::Error::other(format!(
            "OpenCode session.prompt failed: HTTP {status} {body}"
        ))));
    }

    let trimmed = body.trim();
    if trimmed.is_empty() {
        return Err(ExecutorError::Io(io::Error::other(
            "OpenCode session.prompt returned empty response body",
        )));
    }

    let parsed: Value =
        serde_json::from_str(trimmed).map_err(|err| ExecutorError::Io(io::Error::other(err)))?;

    // Success response: { info, parts }
    if parsed.get("info").is_some() && parsed.get("parts").is_some() {
        return Ok(());
    }

    // Error response: { name, data }
    if let Some(name) = parsed.get("name").and_then(Value::as_str) {
        let message = parsed
            .pointer("/data/message")
            .and_then(Value::as_str)
            .unwrap_or(trimmed);
        return Err(ExecutorError::Io(io::Error::other(format!(
            "OpenCode session.prompt failed: {name}: {message}"
        ))));
    }

    Err(ExecutorError::Io(io::Error::other(format!(
        "OpenCode session.prompt returned unexpected response: {trimmed}"
    ))))
}

fn parse_model(model: &str) -> Option<ModelSpec> {
    let (provider_id, model_id) = match model.split_once('/') {
        Some((provider, rest)) => (provider.to_string(), rest.to_string()),
        None => (model.to_string(), String::new()),
    };

    Some(ModelSpec {
        provider_id,
        model_id,
    })
}

async fn connect_event_stream(
    client: &reqwest::Client,
    base_url: &str,
    directory: &str,
    last_event_id: Option<&str>,
) -> Result<reqwest::Response, ExecutorError> {
    let mut req = client
        .get(format!("{base_url}/event"))
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .query(&[("directory", directory)]);

    if let Some(last_event_id) = last_event_id {
        req = req.header("Last-Event-ID", last_event_id);
    }

    let resp = req
        .send()
        .await
        .map_err(|err| ExecutorError::Io(io::Error::other(err)))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp
            .text()
            .await
            .unwrap_or_else(|_| "<failed to read response body>".to_string());
        return Err(ExecutorError::Io(io::Error::other(format!(
            "OpenCode event stream failed: HTTP {status} {body}"
        ))));
    }

    Ok(resp)
}

async fn spawn_event_listener(
    client: reqwest::Client,
    base_url: String,
    directory: String,
    session_id: String,
    initial_resp: reqwest::Response,
    log_writer: LogWriter,
    approvals: Option<Arc<dyn ExecutorApprovalService>>,
    auto_approve: bool,
    control_tx: mpsc::UnboundedSender<ControlEvent>,
) {
    let mut seen_permissions: HashSet<String> = HashSet::new();
    let mut last_event_id: Option<String> = None;
    let mut base_retry_delay = Duration::from_millis(3000);
    let mut attempt: u32 = 0;
    let max_attempts: u32 = 20;
    let mut resp: Option<reqwest::Response> = Some(initial_resp);

    loop {
        let current_resp = match resp.take() {
            Some(r) => {
                attempt = 0;
                r
            }
            None => match connect_event_stream(
                &client,
                &base_url,
                &directory,
                last_event_id.as_deref(),
            )
            .await
            {
                Ok(r) => {
                    attempt = 0;
                    r
                }
                Err(err) => {
                    let _ = log_writer
                        .log_error(format!("OpenCode event stream reconnect failed: {err}"))
                        .await;
                    attempt += 1;
                    if attempt >= max_attempts {
                        let _ = control_tx.send(ControlEvent::Disconnected);
                        return;
                    }

                    let exp = attempt.saturating_sub(1).min(10);
                    let mult = 1u32 << exp;
                    let backoff = base_retry_delay
                        .checked_mul(mult)
                        .unwrap_or(Duration::from_secs(30))
                        .min(Duration::from_secs(30));
                    tokio::time::sleep(backoff).await;
                    continue;
                }
            },
        };

        let outcome = process_event_stream(
            &mut seen_permissions,
            &client,
            &base_url,
            &directory,
            &session_id,
            current_resp,
            &log_writer,
            approvals.clone(),
            auto_approve,
            &control_tx,
            &mut base_retry_delay,
            &mut last_event_id,
        )
        .await;

        match outcome {
            Ok(EventStreamOutcome::Idle) | Ok(EventStreamOutcome::Terminal) => return,
            Ok(EventStreamOutcome::Disconnected) | Err(_) => {
                attempt += 1;
                if attempt >= max_attempts {
                    let _ = control_tx.send(ControlEvent::Disconnected);
                    return;
                }
            }
        }

        let exp = attempt.saturating_sub(1).min(10);
        let mult = 1u32 << exp;
        let backoff = base_retry_delay
            .checked_mul(mult)
            .unwrap_or(Duration::from_secs(30))
            .min(Duration::from_secs(30));
        tokio::time::sleep(backoff).await;
        resp = None;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventStreamOutcome {
    Idle,
    Terminal,
    Disconnected,
}

async fn process_event_stream(
    seen_permissions: &mut HashSet<String>,
    client: &reqwest::Client,
    base_url: &str,
    directory: &str,
    session_id: &str,
    resp: reqwest::Response,
    log_writer: &LogWriter,
    approvals: Option<Arc<dyn ExecutorApprovalService>>,
    auto_approve: bool,
    control_tx: &mpsc::UnboundedSender<ControlEvent>,
    base_retry_delay: &mut Duration,
    last_event_id: &mut Option<String>,
) -> Result<EventStreamOutcome, ExecutorError> {
    let mut stream = resp.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|err| ExecutorError::Io(io::Error::other(err)))?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(raw_event) = next_sse_chunk(&mut buffer) {
            let parsed = parse_sse_event(&raw_event);
            if let Some(id) = parsed.id {
                *last_event_id = Some(id);
            }
            if let Some(retry) = parsed.retry {
                *base_retry_delay = retry;
            }

            if let Some(data) = parsed.data {
                let Some(event_type) = data.get("type").and_then(Value::as_str) else {
                    continue;
                };

                if !event_matches_session(event_type, &data, session_id) {
                    continue;
                }

                let _ = log_writer
                    .log_event(&OpencodeExecutorEvent::SdkEvent {
                        event: data.clone(),
                    })
                    .await;

                match event_type {
                    "session.idle" => {
                        let _ = control_tx.send(ControlEvent::Idle);
                        return Ok(EventStreamOutcome::Idle);
                    }
                    "session.error" => {
                        let error_type = data
                            .pointer("/properties/error/name")
                            .or_else(|| data.pointer("/properties/error/type"))
                            .and_then(Value::as_str)
                            .unwrap_or("unknown");
                        let message = data
                            .pointer("/properties/error/data/message")
                            .or_else(|| data.pointer("/properties/error/message"))
                            .and_then(Value::as_str)
                            .unwrap_or("OpenCode session error")
                            .to_string();

                        if error_type == "ProviderAuthError" {
                            let _ = control_tx.send(ControlEvent::AuthRequired { message });
                            return Ok(EventStreamOutcome::Terminal);
                        }

                        let _ = control_tx.send(ControlEvent::SessionError { message });
                    }
                    "permission.asked" => {
                        let request_id = data
                            .pointer("/properties/id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();

                        if request_id.is_empty() || !seen_permissions.insert(request_id.clone()) {
                            continue;
                        }

                        let tool_call_id = data
                            .pointer("/properties/tool/callID")
                            .and_then(Value::as_str)
                            .unwrap_or(&request_id)
                            .to_string();

                        let permission = data
                            .pointer("/properties/permission")
                            .and_then(Value::as_str)
                            .unwrap_or("tool")
                            .to_string();

                        let tool_input = data
                            .get("properties")
                            .cloned()
                            .unwrap_or_else(|| serde_json::json!({}));

                        let approvals = approvals.clone();
                        let client = client.clone();
                        let base_url = base_url.to_string();
                        let directory = directory.to_string();
                        let log_writer = log_writer.clone();
                        tokio::spawn(async move {
                            let status = request_permission_approval(
                                auto_approve,
                                approvals,
                                &permission,
                                tool_input,
                                &tool_call_id,
                            )
                            .await;

                            let _ = log_writer
                                .log_event(&OpencodeExecutorEvent::ApprovalResponse {
                                    tool_call_id: tool_call_id.clone(),
                                    status: status.clone(),
                                })
                                .await;

                            let (reply, message) = match status {
                                ApprovalStatus::Approved => ("once", None),
                                ApprovalStatus::Denied { reason } => {
                                    let msg = reason
                                        .unwrap_or_else(|| "User denied this tool use request".to_string())
                                        .trim()
                                        .to_string();
                                    let msg = if msg.is_empty() {
                                        "User denied this tool use request".to_string()
                                    } else {
                                        msg
                                    };
                                    ("reject", Some(msg))
                                }
                                ApprovalStatus::TimedOut => (
                                    "reject",
                                    Some(
                                        "Approval request timed out; proceed without using this tool call."
                                            .to_string(),
                                    ),
                                ),
                                ApprovalStatus::Pending => (
                                    "reject",
                                    Some(
                                        "Approval request could not be completed; proceed without using this tool call."
                                            .to_string(),
                                    ),
                                ),
                            };

                            // If we reject without a message, OpenCode treats it as a hard stop.
                            // Provide a message so the agent can continue with guidance.
                            let payload = if reply == "reject" {
                                serde_json::json!({ "reply": reply, "message": message.unwrap_or_else(|| "User denied this tool use request".to_string()) })
                            } else {
                                serde_json::json!({ "reply": reply })
                            };

                            let _ = client
                                .post(format!("{base_url}/permission/{request_id}/reply"))
                                .query(&[("directory", directory.as_str())])
                                .json(&payload)
                                .send()
                                .await;
                        });
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(EventStreamOutcome::Disconnected)
}

fn next_sse_chunk(buffer: &mut String) -> Option<String> {
    let nl = buffer.find("\n\n").map(|idx| (idx, 2));
    let crlf = buffer.find("\r\n\r\n").map(|idx| (idx, 4));

    let (idx, delim_len) = match (nl, crlf) {
        (Some(a), Some(b)) => if a.0 <= b.0 { a } else { b },
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => return None,
    };

    let raw = buffer[..idx].to_string();
    buffer.drain(..idx + delim_len);
    while buffer.starts_with('\n') || buffer.starts_with('\r') {
        buffer.remove(0);
    }
    Some(raw)
}

struct ParsedSseEvent {
    data: Option<Value>,
    id: Option<String>,
    retry: Option<Duration>,
}

fn parse_sse_event(raw_event: &str) -> ParsedSseEvent {
    let mut data_lines: Vec<&str> = Vec::new();
    let mut id: Option<String> = None;
    let mut retry: Option<Duration> = None;
    for line in raw_event.lines() {
        let line = line.trim_end();
        if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.trim_start());
        } else if let Some(rest) = line.strip_prefix("id:") {
            let rest = rest.trim_start().trim();
            if !rest.is_empty() {
                id = Some(rest.to_string());
            }
        } else if let Some(rest) = line.strip_prefix("retry:") {
            let rest = rest.trim_start().trim();
            if let Ok(ms) = rest.parse::<u64>() {
                retry = Some(Duration::from_millis(ms));
            }
        }
    }

    let data = if data_lines.is_empty() {
        None
    } else {
        let joined = data_lines.join("\n");
        serde_json::from_str(&joined).ok()
    };

    ParsedSseEvent { data, id, retry }
}

fn event_matches_session(event_type: &str, event: &Value, session_id: &str) -> bool {
    let extracted = match event_type {
        "message.updated" => event
            .pointer("/properties/info/sessionID")
            .and_then(Value::as_str),
        "message.part.updated" => event
            .pointer("/properties/part/sessionID")
            .and_then(Value::as_str),
        "permission.asked" | "permission.replied" | "session.idle" | "session.error" => event
            .pointer("/properties/sessionID")
            .and_then(Value::as_str),
        _ => event
            .pointer("/properties/sessionID")
            .and_then(Value::as_str)
            .or_else(|| {
                event
                    .pointer("/properties/info/sessionID")
                    .and_then(Value::as_str)
            })
            .or_else(|| {
                event
                    .pointer("/properties/part/sessionID")
                    .and_then(Value::as_str)
            }),
    };

    extracted == Some(session_id)
}

async fn request_permission_approval(
    auto_approve: bool,
    approvals: Option<Arc<dyn ExecutorApprovalService>>,
    tool_name: &str,
    tool_input: Value,
    tool_call_id: &str,
) -> ApprovalStatus {
    if auto_approve {
        return ApprovalStatus::Approved;
    }

    let Some(approvals) = approvals else {
        return ApprovalStatus::Approved;
    };

    match approvals
        .request_tool_approval(tool_name, tool_input, tool_call_id)
        .await
    {
        Ok(status) => status,
        Err(
            ExecutorApprovalError::ServiceUnavailable | ExecutorApprovalError::SessionNotRegistered,
        ) => ApprovalStatus::Approved,
        Err(err) => ApprovalStatus::Denied {
            reason: Some(format!("Approval request failed: {err}")),
        },
    }
}
