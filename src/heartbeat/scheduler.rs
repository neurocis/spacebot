//! Heartbeat scheduler: timer management and execution.
//!
//! Each heartbeat gets its own tokio task that fires on an interval.
//! When a heartbeat fires, it creates a fresh short-lived channel,
//! runs the heartbeat prompt through the LLM, and delivers the result
//! to the delivery target via the messaging system.

use crate::agent::channel::Channel;
use crate::error::Result;
use crate::heartbeat::store::HeartbeatStore;
use crate::messaging::MessagingManager;
use crate::{AgentDeps, InboundMessage, MessageContent, OutboundResponse};
use chrono::Timelike;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{interval, Duration};

/// A heartbeat definition loaded from the database.
#[derive(Debug, Clone)]
pub struct Heartbeat {
    pub id: String,
    pub prompt: String,
    pub interval_secs: u64,
    pub delivery_target: DeliveryTarget,
    pub active_hours: Option<(u8, u8)>,
    pub enabled: bool,
    pub consecutive_failures: u32,
}

/// Where to send heartbeat results.
#[derive(Debug, Clone)]
pub struct DeliveryTarget {
    /// Messaging adapter name (e.g. "discord").
    pub adapter: String,
    /// Platform-specific target (e.g. a Discord channel ID).
    pub target: String,
}

impl DeliveryTarget {
    /// Parse a delivery target string in the format "adapter:target".
    pub fn parse(raw: &str) -> Option<Self> {
        let (adapter, target) = raw.split_once(':')?;
        if adapter.is_empty() || target.is_empty() {
            return None;
        }
        Some(Self {
            adapter: adapter.to_string(),
            target: target.to_string(),
        })
    }
}

impl std::fmt::Display for DeliveryTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.adapter, self.target)
    }
}

/// Serializable heartbeat config (for storage and TOML parsing).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HeartbeatConfig {
    pub id: String,
    pub prompt: String,
    #[serde(default = "default_interval")]
    pub interval_secs: u64,
    /// Delivery target in "adapter:target" format (e.g. "discord:123456789").
    pub delivery_target: String,
    pub active_hours: Option<(u8, u8)>,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_interval() -> u64 {
    3600
}

fn default_true() -> bool {
    true
}

/// Context needed to execute a heartbeat (agent resources + messaging).
///
/// Prompts, identity, browser config, and skills are read from
/// `deps.runtime_config` on each heartbeat firing so changes propagate
/// without restarting the scheduler.
#[derive(Clone)]
pub struct HeartbeatContext {
    pub deps: AgentDeps,
    pub screenshot_dir: std::path::PathBuf,
    pub logs_dir: std::path::PathBuf,
    pub messaging_manager: Arc<MessagingManager>,
    pub store: Arc<HeartbeatStore>,
}

const MAX_CONSECUTIVE_FAILURES: u32 = 3;

/// Scheduler that manages heartbeat timers and execution.
pub struct Scheduler {
    heartbeats: Arc<RwLock<HashMap<String, Heartbeat>>>,
    timers: Arc<RwLock<HashMap<String, tokio::task::JoinHandle<()>>>>,
    context: HeartbeatContext,
}

impl std::fmt::Debug for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scheduler").finish_non_exhaustive()
    }
}

impl Scheduler {
    pub fn new(context: HeartbeatContext) -> Self {
        Self {
            heartbeats: Arc::new(RwLock::new(HashMap::new())),
            timers: Arc::new(RwLock::new(HashMap::new())),
            context,
        }
    }

    /// Register and start a heartbeat from config.
    pub async fn register(&self, config: HeartbeatConfig) -> Result<()> {
        let delivery_target = DeliveryTarget::parse(&config.delivery_target).unwrap_or_else(|| {
            tracing::warn!(
                heartbeat_id = %config.id,
                raw_target = %config.delivery_target,
                "invalid delivery target format, expected 'adapter:target'"
            );
            DeliveryTarget {
                adapter: "unknown".into(),
                target: config.delivery_target.clone(),
            }
        });

        let heartbeat = Heartbeat {
            id: config.id.clone(),
            prompt: config.prompt,
            interval_secs: config.interval_secs,
            delivery_target,
            active_hours: config.active_hours,
            enabled: config.enabled,
            consecutive_failures: 0,
        };

        {
            let mut heartbeats = self.heartbeats.write().await;
            heartbeats.insert(config.id.clone(), heartbeat);
        }

        if config.enabled {
            self.start_timer(&config.id).await;
        }

        tracing::info!(heartbeat_id = %config.id, interval_secs = config.interval_secs, "heartbeat registered");
        Ok(())
    }

    /// Start a timer loop for a heartbeat.
    async fn start_timer(&self, heartbeat_id: &str) {
        let heartbeat_id_for_map = heartbeat_id.to_string();
        let heartbeat_id = heartbeat_id.to_string();
        let heartbeats = self.heartbeats.clone();
        let context = self.context.clone();

        let handle = tokio::spawn(async move {
            // Look up interval before entering the loop
            let interval_secs = {
                let hb = heartbeats.read().await;
                hb.get(&heartbeat_id)
                    .map(|h| h.interval_secs)
                    .unwrap_or(3600)
            };

            let mut ticker = interval(Duration::from_secs(interval_secs));
            // Skip the immediate first tick — heartbeats should wait for the first interval
            ticker.tick().await;

            loop {
                ticker.tick().await;

                let heartbeat = {
                    let hb = heartbeats.read().await;
                    match hb.get(&heartbeat_id) {
                        Some(h) if !h.enabled => {
                            tracing::debug!(heartbeat_id = %heartbeat_id, "heartbeat disabled, stopping timer");
                            break;
                        }
                        Some(h) => h.clone(),
                        None => {
                            tracing::debug!(heartbeat_id = %heartbeat_id, "heartbeat removed, stopping timer");
                            break;
                        }
                    }
                };

                // Check active hours window
                if let Some((start, end)) = heartbeat.active_hours {
                    let current_hour = chrono::Local::now().hour() as u8;
                    let in_window = if start <= end {
                        current_hour >= start && current_hour < end
                    } else {
                        // Wraps midnight (e.g. 22:00 - 06:00)
                        current_hour >= start || current_hour < end
                    };
                    if !in_window {
                        tracing::debug!(
                            heartbeat_id = %heartbeat_id,
                            current_hour,
                            start,
                            end,
                            "outside active hours, skipping"
                        );
                        continue;
                    }
                }

                tracing::info!(heartbeat_id = %heartbeat_id, "heartbeat firing");

                match run_heartbeat(&heartbeat, &context).await {
                    Ok(()) => {
                        // Reset failure count on success
                        let mut hb = heartbeats.write().await;
                        if let Some(h) = hb.get_mut(&heartbeat_id) {
                            h.consecutive_failures = 0;
                        }
                    }
                    Err(error) => {
                        tracing::error!(
                            heartbeat_id = %heartbeat_id,
                            %error,
                            "heartbeat execution failed"
                        );

                        let should_disable = {
                            let mut hb = heartbeats.write().await;
                            if let Some(h) = hb.get_mut(&heartbeat_id) {
                                h.consecutive_failures += 1;
                                h.consecutive_failures >= MAX_CONSECUTIVE_FAILURES
                            } else {
                                false
                            }
                        };

                        if should_disable {
                            tracing::warn!(
                                heartbeat_id = %heartbeat_id,
                                "circuit breaker tripped after {MAX_CONSECUTIVE_FAILURES} consecutive failures, disabling"
                            );

                            {
                                let mut hb = heartbeats.write().await;
                                if let Some(h) = hb.get_mut(&heartbeat_id) {
                                    h.enabled = false;
                                }
                            }

                            // Persist the disabled state
                            if let Err(error) = context.store.update_enabled(&heartbeat_id, false).await {
                                tracing::error!(%error, "failed to persist heartbeat disabled state");
                            }

                            break;
                        }
                    }
                }
            }
        });

        let mut timers = self.timers.write().await;
        timers.insert(heartbeat_id_for_map, handle);
    }

    /// Shutdown all heartbeat timers and wait for them to finish.
    pub async fn shutdown(&self) {
        let handles: Vec<(String, tokio::task::JoinHandle<()>)> = {
            let mut timers = self.timers.write().await;
            timers.drain().collect()
        };

        for (id, handle) in handles {
            handle.abort();
            let _ = handle.await;
            tracing::debug!(heartbeat_id = %id, "heartbeat timer stopped");
        }
    }
}

/// Execute a single heartbeat: create a fresh channel, run the prompt, deliver the result.
async fn run_heartbeat(heartbeat: &Heartbeat, context: &HeartbeatContext) -> Result<()> {
    let channel_id: crate::ChannelId = Arc::from(format!("heartbeat:{}", heartbeat.id).as_str());

    // Create the outbound response channel to collect whatever the channel produces
    let (response_tx, mut response_rx) = tokio::sync::mpsc::channel::<OutboundResponse>(32);

    // Subscribe to the agent's event bus (the channel needs this for branch/worker events)
    let event_rx = context.deps.event_tx.subscribe();

    let (channel, channel_tx) = Channel::new(
        channel_id.clone(),
        context.deps.clone(),
        response_tx,
        event_rx,
        context.screenshot_dir.clone(),
        context.logs_dir.clone(),
    );

    // Spawn the channel's event loop
    let channel_handle = tokio::spawn(async move {
        if let Err(error) = channel.run().await {
            tracing::error!(%error, "heartbeat channel failed");
        }
    });

    // Send the heartbeat prompt as a synthetic message
    let message = InboundMessage {
        id: uuid::Uuid::new_v4().to_string(),
        source: "heartbeat".into(),
        conversation_id: format!("heartbeat:{}", heartbeat.id),
        sender_id: "system".into(),
        agent_id: Some(context.deps.agent_id.clone()),
        content: MessageContent::Text(heartbeat.prompt.clone()),
        timestamp: chrono::Utc::now(),
        metadata: HashMap::new(),
    };

    channel_tx
        .send(message)
        .await
        .map_err(|error| anyhow::anyhow!("failed to send heartbeat prompt to channel: {error}"))?;

    // Collect responses with a timeout. The channel may produce multiple messages
    // (e.g. status updates, then text). We only care about text responses.
    let mut collected_text = Vec::new();
    let timeout = Duration::from_secs(120);

    // Drop the sender so the channel knows no more messages are coming.
    // The channel will process the one message and then its event loop will end
    // when the sender is dropped (message_rx returns None).
    drop(channel_tx);

    loop {
        match tokio::time::timeout(timeout, response_rx.recv()).await {
            Ok(Some(OutboundResponse::Text(text))) => {
                collected_text.push(text);
            }
            Ok(Some(_)) => {
                // Status updates, stream chunks, etc. — ignore for heartbeats
            }
            Ok(None) => {
                // Channel finished (response_tx dropped)
                break;
            }
            Err(_) => {
                tracing::warn!(heartbeat_id = %heartbeat.id, "heartbeat timed out after {timeout:?}");
                channel_handle.abort();
                break;
            }
        }
    }

    // Wait for the channel task to finish (it should already be done since we dropped channel_tx)
    let _ = channel_handle.await;

    let result_text = collected_text.join("\n\n");
    let has_result = !result_text.trim().is_empty();

    // Log execution
    let summary = if has_result {
        Some(result_text.as_str())
    } else {
        None
    };
    if let Err(error) = context
        .store
        .log_execution(&heartbeat.id, true, summary)
        .await
    {
        tracing::warn!(%error, "failed to log heartbeat execution");
    }

    // Deliver result to target (only if there's something to say)
    if has_result {
        if let Err(error) = context
            .messaging_manager
            .broadcast(
                &heartbeat.delivery_target.adapter,
                &heartbeat.delivery_target.target,
                OutboundResponse::Text(result_text),
            )
            .await
        {
            tracing::error!(
                heartbeat_id = %heartbeat.id,
                target = %heartbeat.delivery_target,
                %error,
                "failed to deliver heartbeat result"
            );
            // Log the delivery failure
            let _ = context
                .store
                .log_execution(&heartbeat.id, false, Some(&error.to_string()))
                .await;
            return Err(error);
        }

        tracing::info!(
            heartbeat_id = %heartbeat.id,
            target = %heartbeat.delivery_target,
            "heartbeat result delivered"
        );
    } else {
        tracing::debug!(heartbeat_id = %heartbeat.id, "heartbeat produced no output, skipping delivery");
    }

    Ok(())
}
