//! Messaging adapters (Discord, Slack, Telegram, Twitch, Webhook, WebChat).

pub mod traits;
pub mod manager;
pub mod discord;
pub mod slack;
pub mod telegram;
pub mod twitch;
pub mod webhook;
pub mod webchat;

pub use traits::Messaging;
pub use manager::MessagingManager;
