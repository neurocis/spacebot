use crate::error::Result;
use anyhow::Context;
use minijinja::{context, Environment, Value};
use std::collections::HashMap;
use std::sync::Arc;

/// Template engine for rendering system prompts with dynamic variables.
///
/// Prompts are bundled in the binary as `include_str!` embedded templates.
/// Language selection is done at initialization and templates are not
/// reloadable at runtime (no file watching, no hot reload).
#[derive(Clone)]
pub struct PromptEngine {
    /// The MiniJinja environment holding all templates for the configured language.
    /// Wrapped in Arc to make PromptEngine Clone.
    env: Arc<Environment<'static>>,
    /// Selected language code (e.g., "en").
    language: String,
}

impl PromptEngine {
    /// Create a new engine with templates for the given language.
    ///
    /// Currently only "en" (English) is fully implemented.
    /// The language parameter exists for future i18n expansion.
    pub fn new(language: &str) -> anyhow::Result<Self> {
        if language != "en" {
            tracing::warn!(
                language = language,
                "non-English language requested, falling back to English"
            );
        }

        let mut env = Environment::new();

        // Register all English templates
        // Process prompts
        env.add_template("channel", CHANNEL_TEMPLATE)?;
        env.add_template("branch", BRANCH_TEMPLATE)?;
        env.add_template("worker", WORKER_TEMPLATE)?;
        env.add_template("cortex", CORTEX_TEMPLATE)?;
        env.add_template("cortex_bulletin", CORTEX_BULLETIN_TEMPLATE)?;
        env.add_template("compactor", COMPACTOR_TEMPLATE)?;
        env.add_template("memory_persistence", MEMORY_PERSISTENCE_TEMPLATE)?;
        env.add_template("ingestion", INGESTION_TEMPLATE)?;

        // Fragment templates for inline strings
        env.add_template(
            "fragments/worker_capabilities",
            FRAGMENT_WORKER_CAPABILITIES,
        )?;
        env.add_template(
            "fragments/conversation_context",
            FRAGMENT_CONVERSATION_CONTEXT,
        )?;
        env.add_template("fragments/skills_channel", FRAGMENT_SKILLS_CHANNEL)?;
        env.add_template("fragments/skills_worker", FRAGMENT_SKILLS_WORKER)?;

        // System message fragments
        env.add_template("fragments/system/retrigger", FRAGMENT_SYSTEM_RETRIGGER)?;
        env.add_template("fragments/system/truncation", FRAGMENT_SYSTEM_TRUNCATION)?;
        env.add_template(
            "fragments/system/worker_overflow",
            FRAGMENT_SYSTEM_WORKER_OVERFLOW,
        )?;
        env.add_template(
            "fragments/system/worker_compact",
            FRAGMENT_SYSTEM_WORKER_COMPACT,
        )?;
        env.add_template(
            "fragments/system/memory_persistence",
            FRAGMENT_SYSTEM_MEMORY_PERSISTENCE,
        )?;
        env.add_template(
            "fragments/system/cortex_synthesis",
            FRAGMENT_SYSTEM_CORTEX_SYNTHESIS,
        )?;
        env.add_template(
            "fragments/system/ingestion_chunk",
            FRAGMENT_SYSTEM_INGESTION_CHUNK,
        )?;

        Ok(Self {
            env: Arc::new(env),
            language: language.to_string(),
        })
    }

    /// Render a template by name with the given context variables.
    ///
    /// # Arguments
    /// * `template_name` - Name of the template to render (e.g., "channel", "fragments/worker_capabilities")
    /// * `context` - MiniJinja Value containing template variables
    ///
    /// # Example
    /// ```rust
    /// let ctx = context! {
    ///     identity_context => "Some identity text",
    ///     browser_enabled => true,
    /// };
    /// let rendered = engine.render("channel", ctx)?;
    /// ```
    pub fn render(&self, template_name: &str, context: Value) -> Result<String> {
        let template = self
            .env
            .get_template(template_name)
            .with_context(|| format!("template '{}' not found", template_name))?;

        template
            .render(context)
            .with_context(|| format!("failed to render template '{}'", template_name))
            .map_err(Into::into)
    }

    /// Render a template with a HashMap of context variables.
    pub fn render_map(&self, template_name: &str, vars: HashMap<String, Value>) -> Result<String> {
        let context = Value::from_object(vars);
        self.render(template_name, context)
    }

    /// Convenience method for rendering simple templates with no variables.
    pub fn render_static(&self, template_name: &str) -> Result<String> {
        self.render(template_name, Value::UNDEFINED)
    }

    /// Convenience method for rendering worker capabilities fragment.
    pub fn render_worker_capabilities(
        &self,
        browser_enabled: bool,
        web_search_enabled: bool,
    ) -> Result<String> {
        self.render(
            "fragments/worker_capabilities",
            context! {
                browser_enabled => browser_enabled,
                web_search_enabled => web_search_enabled,
            },
        )
    }

    /// Convenience method for rendering conversation context fragment.
    pub fn render_conversation_context(
        &self,
        platform: &str,
        server_name: Option<&str>,
        channel_name: Option<&str>,
    ) -> Result<String> {
        self.render(
            "fragments/conversation_context",
            context! {
                platform => platform,
                server_name => server_name,
                channel_name => channel_name,
            },
        )
    }

    /// Convenience method for rendering skills channel fragment.
    pub fn render_skills_channel(&self, skills: Vec<SkillInfo>) -> Result<String> {
        self.render(
            "fragments/skills_channel",
            context! {
                skills => skills,
            },
        )
    }

    /// Convenience method for rendering skills worker fragment.
    pub fn render_skills_worker(&self, skill_name: &str, skill_content: &str) -> Result<String> {
        self.render(
            "fragments/skills_worker",
            context! {
                skill_name => skill_name,
                skill_content => skill_content,
            },
        )
    }

    /// Convenience method for rendering system retrigger message.
    pub fn render_system_retrigger(&self) -> Result<String> {
        self.render_static("fragments/system/retrigger")
    }

    /// Convenience method for rendering truncation marker.
    pub fn render_system_truncation(&self, remove_count: usize) -> Result<String> {
        self.render(
            "fragments/system/truncation",
            context! {
                remove_count => remove_count,
            },
        )
    }

    /// Convenience method for rendering worker overflow recovery message.
    pub fn render_system_worker_overflow(&self) -> Result<String> {
        self.render_static("fragments/system/worker_overflow")
    }

    /// Convenience method for rendering worker compaction message.
    pub fn render_system_worker_compact(&self, remove_count: usize, recap: &str) -> Result<String> {
        self.render(
            "fragments/system/worker_compact",
            context! {
                remove_count => remove_count,
                recap => recap,
            },
        )
    }

    /// Convenience method for rendering memory persistence prompt.
    pub fn render_system_memory_persistence(&self) -> Result<String> {
        self.render_static("fragments/system/memory_persistence")
    }

    /// Convenience method for rendering cortex synthesis prompt.
    pub fn render_system_cortex_synthesis(
        &self,
        max_words: usize,
        raw_sections: &str,
    ) -> Result<String> {
        self.render(
            "fragments/system/cortex_synthesis",
            context! {
                max_words => max_words,
                raw_sections => raw_sections,
            },
        )
    }

    /// Convenience method for rendering ingestion chunk prompt.
    pub fn render_system_ingestion_chunk(
        &self,
        filename: &str,
        chunk_number: usize,
        total_chunks: usize,
        chunk: &str,
    ) -> Result<String> {
        self.render(
            "fragments/system/ingestion_chunk",
            context! {
                filename => filename,
                chunk_number => chunk_number,
                total_chunks => total_chunks,
                chunk => chunk,
            },
        )
    }

    /// Render the complete channel system prompt with all dynamic components.
    pub fn render_channel_prompt(
        &self,
        identity_context: Option<String>,
        memory_bulletin: Option<String>,
        skills_prompt: Option<String>,
        worker_capabilities: String,
        conversation_context: Option<String>,
        status_text: Option<String>,
    ) -> Result<String> {
        self.render(
            "channel",
            context! {
                identity_context => identity_context,
                memory_bulletin => memory_bulletin,
                skills_prompt => skills_prompt,
                worker_capabilities => worker_capabilities,
                conversation_context => conversation_context,
                status_text => status_text,
            },
        )
    }

    /// Get the configured language code.
    pub fn language(&self) -> &str {
        &self.language
    }
}

/// Information about a skill for template rendering.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SkillInfo {
    pub name: String,
    pub description: String,
    pub location: String,
}

// ============================================================================
// Process Prompt Templates
// ============================================================================

const CHANNEL_TEMPLATE: &str = include_str!("../../prompts/en/channel.md.j2");
const BRANCH_TEMPLATE: &str = include_str!("../../prompts/en/branch.md.j2");
const WORKER_TEMPLATE: &str = include_str!("../../prompts/en/worker.md.j2");
const CORTEX_TEMPLATE: &str = include_str!("../../prompts/en/cortex.md.j2");
const CORTEX_BULLETIN_TEMPLATE: &str = include_str!("../../prompts/en/cortex_bulletin.md.j2");
const COMPACTOR_TEMPLATE: &str = include_str!("../../prompts/en/compactor.md.j2");
const MEMORY_PERSISTENCE_TEMPLATE: &str = include_str!("../../prompts/en/memory_persistence.md.j2");
const INGESTION_TEMPLATE: &str = include_str!("../../prompts/en/ingestion.md.j2");

// ============================================================================
// Fragment Templates
// ============================================================================

const FRAGMENT_WORKER_CAPABILITIES: &str = r#"
## Worker Capabilities

When you spawn a worker, it has access to the following tools:

- **shell** — run shell commands
- **file** — read, write, search, and list files  
- **exec** — run subprocesses with environment control
- **set_status** — update worker status visible in your status block
{%- if browser_enabled %}
- **browser** — browse web pages, take screenshots, click elements, fill forms
{%- endif %}
{%- if web_search_enabled %}
- **web_search** — search the web via Brave Search API
{%- endif %}

Workers do NOT have conversation context or memory access. Include all necessary context in the task description.
"#;

const FRAGMENT_CONVERSATION_CONTEXT: &str = r#"
Platform: {{ platform }}
{%- if server_name %}
Server: {{ server_name }}
{%- endif %}
{%- if channel_name %}
Channel: #{{ channel_name }}
{%- endif %}
Multiple users may be present. Each message is prefixed with [username].
"#;

const FRAGMENT_SKILLS_CHANNEL: &str = r#"
## Available Skills

You have access to the following skills. Skills contain specialized instructions for specific tasks. When a user's request matches a skill, spawn a worker to handle it and include the skill name in the task description so the worker knows which skill to follow.

To use a skill, spawn a worker with a task like: "Use the [skill-name] skill to [task]. Read the skill instructions at [path] first."

<available_skills>
{%- for skill in skills %}
  <skill>
    <name>{{ skill.name }}</name>
    <description>{{ skill.description }}</description>
    <location>{{ skill.location }}</location>
  </skill>
{%- endfor %}
</available_skills>
"#;

const FRAGMENT_SKILLS_WORKER: &str = r#"
## Skill Instructions

You are executing the **{{ skill_name }}** skill. Follow these instructions:

{{ skill_content }}
"#;

// ============================================================================
// System Message Fragments
// ============================================================================

const FRAGMENT_SYSTEM_RETRIGGER: &str = "[System: a background process has completed. Check your history and status block for the result, then respond to the user.]";

const FRAGMENT_SYSTEM_TRUNCATION: &str = "[System: {{ remove_count }} older messages were truncated due to context limits. Some conversation history has been lost.]";

const FRAGMENT_SYSTEM_WORKER_OVERFLOW: &str =
    "[System: Previous attempt exceeded context limit. Older history has been compacted.]";

const FRAGMENT_SYSTEM_WORKER_COMPACT: &str = r#"[System: Earlier work has been summarized to free up context. {{ remove_count }} messages compacted.]

## Work completed so far:

{{ recap }}"#;

const FRAGMENT_SYSTEM_MEMORY_PERSISTENCE: &str = "Review the recent conversation and persist any important information as memories. Start by recalling existing memories related to the topics discussed, then save new or updated memories with appropriate associations.";

const FRAGMENT_SYSTEM_CORTEX_SYNTHESIS: &str = r#"Synthesize the following memory data into a concise briefing of {{ max_words }} words or fewer.

## Raw Memory Data

{{ raw_sections }}"#;

const FRAGMENT_SYSTEM_INGESTION_CHUNK: &str = r#"## File: {{ filename }} (chunk {{ chunk_number }} of {{ total_chunks }})

Process the following text and extract any useful memories:

---

{{ chunk }}

---"#;
