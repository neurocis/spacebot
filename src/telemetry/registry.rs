//! Global metrics registry and metric handle definitions.

use prometheus::{
    Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGaugeVec, Opts, Registry,
};

use std::sync::LazyLock;

/// Global metrics instance. Initialized once, accessed from any call site.
static METRICS: LazyLock<Metrics> = LazyLock::new(Metrics::new);

/// All Prometheus metric handles for the Spacebot process.
///
/// Access via `Metrics::global()`. Metric handles are cheap to clone (Arc
/// internally) so call sites can grab references without threading state.
pub struct Metrics {
    pub(crate) registry: Registry,

    // -- Counters --

    /// Total LLM completion requests.
    /// Labels: agent_id, model, tier (e.g. "channel", "branch", "worker").
    pub llm_requests_total: IntCounterVec,

    /// Total tool calls executed across all processes.
    /// Labels: agent_id, tool_name.
    pub tool_calls_total: IntCounterVec,

    /// Total memory recall (read) operations.
    pub memory_reads_total: IntCounter,

    /// Total memory save (write) operations.
    pub memory_writes_total: IntCounter,

    // -- Histograms --

    /// LLM request duration in seconds.
    pub llm_request_duration_seconds: HistogramVec,

    /// Tool call duration in seconds.
    pub tool_call_duration_seconds: Histogram,

    // -- Gauges --

    /// Currently active workers per agent.
    /// Label: agent_id.
    pub active_workers: IntGaugeVec,

    /// Total memory entries per agent.
    /// Label: agent_id.
    // TODO: Not wired to any call site. Needs periodic store queries or
    // inc/dec in MemoryStore::save()/delete() to reflect actual counts.
    pub memory_entry_count: IntGaugeVec,
}

impl Metrics {
    fn new() -> Self {
        let registry = Registry::new();

        let llm_requests_total = IntCounterVec::new(
            Opts::new("spacebot_llm_requests_total", "Total LLM completion requests"),
            &["agent_id", "model", "tier"],
        )
        .expect("hardcoded metric descriptor");

        let tool_calls_total = IntCounterVec::new(
            Opts::new("spacebot_tool_calls_total", "Total tool calls executed"),
            &["agent_id", "tool_name"],
        )
        .expect("hardcoded metric descriptor");

        let memory_reads_total = IntCounter::new(
            "spacebot_memory_reads_total",
            "Total memory recall operations",
        )
        .expect("hardcoded metric descriptor");

        let memory_writes_total = IntCounter::new(
            "spacebot_memory_writes_total",
            "Total memory save operations",
        )
        .expect("hardcoded metric descriptor");

        let llm_request_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "spacebot_llm_request_duration_seconds",
                "LLM request duration in seconds",
            )
            // TODO: Max bucket 10s is too low for LLM requests. Completions with
            // retries and fallback chains routinely take 15-60s. Add upper buckets
            // (e.g. 15, 30, 60, 120) so p99 latency doesn't collapse into +Inf.
            .buckets(vec![0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]),
            &["agent_id", "model", "tier"],
        )
        .expect("hardcoded metric descriptor");

        let tool_call_duration_seconds = Histogram::with_opts(
            HistogramOpts::new(
                "spacebot_tool_call_duration_seconds",
                "Tool call duration in seconds",
            )
            .buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]),
        )
        .expect("hardcoded metric descriptor");

        let active_workers = IntGaugeVec::new(
            Opts::new("spacebot_active_workers", "Currently active workers"),
            &["agent_id"],
        )
        .expect("hardcoded metric descriptor");

        let memory_entry_count = IntGaugeVec::new(
            Opts::new("spacebot_memory_entry_count", "Total memory entries per agent"),
            &["agent_id"],
        )
        .expect("hardcoded metric descriptor");

        registry.register(Box::new(llm_requests_total.clone())).expect("hardcoded metric");
        registry.register(Box::new(tool_calls_total.clone())).expect("hardcoded metric");
        registry.register(Box::new(memory_reads_total.clone())).expect("hardcoded metric");
        registry.register(Box::new(memory_writes_total.clone())).expect("hardcoded metric");
        registry.register(Box::new(llm_request_duration_seconds.clone())).expect("hardcoded metric");
        registry.register(Box::new(tool_call_duration_seconds.clone())).expect("hardcoded metric");
        registry.register(Box::new(active_workers.clone())).expect("hardcoded metric");
        registry.register(Box::new(memory_entry_count.clone())).expect("hardcoded metric");

        Self {
            registry,
            llm_requests_total,
            tool_calls_total,
            memory_reads_total,
            memory_writes_total,
            llm_request_duration_seconds,
            tool_call_duration_seconds,
            active_workers,
            memory_entry_count,
        }
    }

    /// Access the global metrics instance.
    pub fn global() -> &'static Self {
        &METRICS
    }
}
