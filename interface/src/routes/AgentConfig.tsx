import { useCallback, useEffect, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, type AgentConfigResponse, type AgentConfigUpdateRequest } from "@/api/client";
import { Button, SettingSidebarButton } from "@/ui";
import { PlusSignIcon, MinusSignIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

type SectionId = "soul" | "identity" | "user" | "routing" | "tuning" | "compaction" | "cortex" | "coalesce" | "memory" | "browser" | "discord";

const SECTIONS: {
	id: SectionId;
	label: string;
	group: "identity" | "config";
	description: string;
	detail: string;
}[] = [
	{ id: "soul", label: "Soul", group: "identity", description: "SOUL.md", detail: "Defines the agent's personality, values, communication style, and behavioral boundaries. This is the core of who the agent is." },
	{ id: "identity", label: "Identity", group: "identity", description: "IDENTITY.md", detail: "The agent's name, nature, and purpose. How it introduces itself and what it understands its role to be." },
	{ id: "user", label: "User", group: "identity", description: "USER.md", detail: "Information about the human this agent interacts with. Name, preferences, context, and anything that helps the agent personalize responses." },
	{ id: "routing", label: "Model Routing", group: "config", description: "Which models each process uses", detail: "Controls which LLM model is used for each process type. Channels handle user-facing conversation, branches do thinking, workers execute tasks, the compactor summarizes context, and the cortex observes system state." },
	{ id: "tuning", label: "Tuning", group: "config", description: "Turn limits, context window, branches", detail: "Core limits that control how much work the agent does per message. Max turns caps LLM iterations per channel message. Context window sets the token budget. Branch limits control parallel thinking." },
	{ id: "compaction", label: "Compaction", group: "config", description: "Context compaction thresholds", detail: "Thresholds that trigger context summarization as the conversation grows. Background kicks in early, aggressive compresses harder, and emergency truncates without LLM involvement. All values are fractions of the context window." },
	{ id: "cortex", label: "Cortex", group: "config", description: "System observer settings", detail: "The cortex monitors active processes and generates memory bulletins. Tick interval controls observation frequency. Timeouts determine when stuck workers or branches get cancelled. The circuit breaker auto-disables after consecutive failures." },
	{ id: "coalesce", label: "Coalesce", group: "config", description: "Message batching", detail: "When multiple messages arrive in quick succession, coalescing batches them into a single LLM turn. This prevents the agent from responding to each message individually in fast-moving conversations." },
	{ id: "memory", label: "Memory Persistence", group: "config", description: "Auto-save interval", detail: "Spawns a silent background branch at regular intervals to recall existing memories and save new ones from the recent conversation. Runs without blocking the channel." },
	{ id: "browser", label: "Browser", group: "config", description: "Chrome automation", detail: "Controls browser automation tools available to workers. When enabled, workers can navigate web pages, take screenshots, and interact with sites. JavaScript evaluation is a separate permission." },
	{ id: "discord", label: "Discord", group: "config", description: "Discord adapter settings", detail: "Instance-level Discord adapter configuration. Controls how the bot interacts with messages from other bots on Discord. Self-messages are always ignored regardless of settings." },
];

interface AgentConfigProps {
	agentId: string;
}

const isIdentityField = (id: SectionId): id is "soul" | "identity" | "user" => {
	return id === "soul" || id === "identity" || id === "user";
};

const getIdentityField = (data: { soul: string | null; identity: string | null; user: string | null }, field: SectionId): string | null => {
	if (isIdentityField(field)) {
		return data[field];
	}
	return null;
};

export function AgentConfig({ agentId }: AgentConfigProps) {
	const queryClient = useQueryClient();
	const [activeSection, setActiveSection] = useState<SectionId>("soul");

	const identityQuery = useQuery({
		queryKey: ["agent-identity", agentId],
		queryFn: () => api.agentIdentity(agentId),
		staleTime: 10_000,
	});

	const configQuery = useQuery({
		queryKey: ["agent-config", agentId],
		queryFn: () => api.agentConfig(agentId),
		staleTime: 10_000,
	});

	const identityMutation = useMutation({
		mutationFn: (update: { field: "soul" | "identity" | "user"; content: string }) =>
			api.updateIdentity({
				agent_id: agentId,
				[update.field]: update.content,
			}),
		onSuccess: (result) => {
			queryClient.setQueryData(["agent-identity", agentId], result);
		},
	});

	const configMutation = useMutation({
		mutationFn: (update: AgentConfigUpdateRequest) => api.updateAgentConfig(update),
		onSuccess: (result) => {
			queryClient.setQueryData(["agent-config", agentId], result);
		},
	});

	const isLoading = identityQuery.isLoading || configQuery.isLoading;
	const isError = identityQuery.isError || configQuery.isError;

	if (isLoading) {
		return (
			<div className="flex h-full items-center justify-center">
				<div className="flex items-center gap-2 text-ink-dull">
					<div className="h-2 w-2 animate-pulse rounded-full bg-accent" />
					Loading configuration...
				</div>
			</div>
		);
	}

	if (isError) {
		return (
			<div className="flex h-full items-center justify-center">
				<p className="text-sm text-red-400">Failed to load configuration</p>
			</div>
		);
	}

	const active = SECTIONS.find((s) => s.id === activeSection)!;
	const isIdentitySection = active.group === "identity";

	return (
		<div className="flex h-full">
			{/* Sidebar */}
			<div className="flex w-52 flex-shrink-0 flex-col border-r border-app-line/50 bg-app-darkBox/20 overflow-y-auto">
				{/* Identity Group */}
				<div className="px-3 pb-1 pt-4">
					<span className="text-tiny font-medium uppercase tracking-wider text-ink-faint">Identity</span>
				</div>
					<div className="flex flex-col gap-0.5 px-2">
					{SECTIONS.filter((s) => s.group === "identity").map((section) => {
						const isActive = activeSection === section.id;
						const hasContent = !!getIdentityField(identityQuery.data ?? { soul: null, identity: null, user: null }, section.id)?.trim();
						return (
							<SettingSidebarButton
								key={section.id}
								onClick={() => setActiveSection(section.id)}
								active={isActive}
							>
								<span className="flex-1">{section.label}</span>
								{!hasContent && (
									<span className="rounded bg-amber-500/10 px-1 py-0.5 text-tiny text-amber-400/70">empty</span>
								)}
							</SettingSidebarButton>
						);
					})}
				</div>

				{/* Config Group */}
				<div className="px-3 pb-1 pt-4 mt-2">
					<span className="text-tiny font-medium uppercase tracking-wider text-ink-faint">Configuration</span>
				</div>
					<div className="flex flex-col gap-0.5 px-2">
					{SECTIONS.filter((s) => s.group === "config").map((section) => {
						const isActive = activeSection === section.id;
						return (
							<SettingSidebarButton
								key={section.id}
								onClick={() => setActiveSection(section.id)}
								active={isActive}
							>
								<span className="flex-1">{section.label}</span>
							</SettingSidebarButton>
						);
					})}
				</div>
			</div>

			{/* Editor */}
			<div className="flex flex-1 flex-col overflow-hidden">
				{isIdentitySection ? (
				<IdentityEditor
					key={active.id}
					label={active.label}
					description={active.description}
					content={getIdentityField(identityQuery.data ?? { soul: null, identity: null, user: null }, active.id)}
					saving={identityMutation.isPending}
					onSave={(content) => {
						// Only mutate for identity sections
						if (isIdentityField(active.id)) {
							identityMutation.mutate({ field: active.id, content });
						}
					}}
				/>
				) : (
					<ConfigSectionEditor
						sectionId={active.id}
						label={active.label}
						description={active.description}
						detail={active.detail}
						config={configQuery.data!}
						saving={configMutation.isPending}
						onSave={(update) => configMutation.mutate({ agent_id: agentId, ...update })}
					/>
				)}
			</div>
		</div>
	);
}

// -- Identity Editor --

interface IdentityEditorProps {
	label: string;
	description: string;
	content: string | null;
	saving: boolean;
	onSave: (content: string) => void;
}

function IdentityEditor({ label, description, content, saving, onSave }: IdentityEditorProps) {
	const [value, setValue] = useState(content ?? "");
	const [dirty, setDirty] = useState(false);

	useEffect(() => {
		if (!dirty) {
			setValue(content ?? "");
		}
	}, [content, dirty]);

	const handleChange = useCallback((event: React.ChangeEvent<HTMLTextAreaElement>) => {
		setValue(event.target.value);
		setDirty(true);
	}, []);

	const handleSave = useCallback(() => {
		onSave(value);
		setDirty(false);
	}, [onSave, value]);

	const handleKeyDown = useCallback(
		(event: React.KeyboardEvent) => {
			if ((event.metaKey || event.ctrlKey) && event.key === "s") {
				event.preventDefault();
				if (dirty) handleSave();
			}
		},
		[dirty, handleSave],
	);

	const handleRevert = useCallback(() => {
		setValue(content ?? "");
		setDirty(false);
	}, [content]);

	return (
		<>
			<div className="flex items-center justify-between border-b border-app-line/50 bg-app-darkBox/20 px-5 py-2.5">
				<div className="flex items-center gap-3">
					<h3 className="text-sm font-medium text-ink">{label}</h3>
					<span className="rounded bg-app-darkBox px-1.5 py-0.5 font-mono text-tiny text-ink-faint">{description}</span>
				</div>
			<div className="flex items-center gap-2">
				{dirty && (
					<>
						<Button
							onClick={handleRevert}
							variant="ghost"
							size="sm"
							className="h-6 px-2 text-tiny"
						>
							Revert
						</Button>
						<Button
							onClick={handleSave}
							size="sm"
							loading={saving}
							className="h-6 bg-accent/15 px-2 text-tiny text-accent hover:bg-accent/25"
						>
							Save
						</Button>
					</>
				)}
				{!dirty && <span className="text-tiny text-ink-faint/50">Cmd+S to save</span>}
			</div>
			</div>
			<div className="flex-1 overflow-y-auto p-4">
				<textarea
					value={value}
					onChange={handleChange}
					onKeyDown={handleKeyDown}
					placeholder={`Write ${label.toLowerCase()} content here...`}
					className="h-full w-full resize-none rounded-md border border-transparent bg-app-darkBox/30 px-4 py-3 font-mono text-sm leading-relaxed text-ink-dull placeholder:text-ink-faint/40 focus:border-accent/30 focus:outline-none"
					spellCheck={false}
				/>
			</div>
		</>
	);
}

// -- Config Section Editors --

interface ConfigSectionEditorProps {
	sectionId: SectionId;
	label: string;
	description: string;
	detail: string;
	config: AgentConfigResponse;
	saving: boolean;
	onSave: (update: Partial<AgentConfigUpdateRequest>) => void;
}

function ConfigSectionEditor({ sectionId, label, description, detail, config, saving, onSave }: ConfigSectionEditorProps) {
	const [localValues, setLocalValues] = useState<Record<string, string | number | boolean>>(() => {
		// Initialize from config based on section
		switch (sectionId) {
			case "routing":
				return { ...config.routing };
			case "tuning":
				return { ...config.tuning };
			case "compaction":
				return { ...config.compaction };
			case "cortex":
				return { ...config.cortex };
			case "coalesce":
				return { ...config.coalesce };
			case "memory":
				return { ...config.memory_persistence };
			case "browser":
				return { ...config.browser };
			case "discord":
				return { ...config.discord };
			default:
				return {};
		}
	});

	const [dirty, setDirty] = useState(false);

	// Reset local values when config changes externally
	useEffect(() => {
		if (!dirty) {
			switch (sectionId) {
				case "routing":
					setLocalValues({ ...config.routing });
					break;
				case "tuning":
					setLocalValues({ ...config.tuning });
					break;
				case "compaction":
					setLocalValues({ ...config.compaction });
					break;
				case "cortex":
					setLocalValues({ ...config.cortex });
					break;
				case "coalesce":
					setLocalValues({ ...config.coalesce });
					break;
				case "memory":
					setLocalValues({ ...config.memory_persistence });
					break;
				case "browser":
					setLocalValues({ ...config.browser });
					break;
				case "discord":
					setLocalValues({ ...config.discord });
					break;
			}
		}
	}, [config, sectionId, dirty]);

	const handleChange = useCallback((field: string, value: string | number | boolean) => {
		setLocalValues((prev) => ({ ...prev, [field]: value }));
		setDirty(true);
	}, []);

	const handleSave = useCallback(() => {
		onSave({ [sectionId]: localValues });
		setDirty(false);
	}, [onSave, sectionId, localValues]);

	const handleRevert = useCallback(() => {
		switch (sectionId) {
			case "routing":
				setLocalValues({ ...config.routing });
				break;
			case "tuning":
				setLocalValues({ ...config.tuning });
				break;
			case "compaction":
				setLocalValues({ ...config.compaction });
				break;
			case "cortex":
				setLocalValues({ ...config.cortex });
				break;
			case "coalesce":
				setLocalValues({ ...config.coalesce });
				break;
			case "memory":
				setLocalValues({ ...config.memory_persistence });
				break;
			case "browser":
				setLocalValues({ ...config.browser });
				break;
			case "discord":
				setLocalValues({ ...config.discord });
				break;
		}
		setDirty(false);
	}, [config, sectionId]);

	const renderFields = () => {
		switch (sectionId) {
			case "routing":
				return (
					<div className="grid gap-4">
						<ConfigField
							label="Channel Model"
							description="Model for user-facing channels"
							value={localValues.channel as string}
							onChange={(v) => handleChange("channel", v)}
						/>
						<ConfigField
							label="Branch Model"
							description="Model for thinking branches"
							value={localValues.branch as string}
							onChange={(v) => handleChange("branch", v)}
						/>
						<ConfigField
							label="Worker Model"
							description="Model for task workers"
							value={localValues.worker as string}
							onChange={(v) => handleChange("worker", v)}
						/>
						<ConfigField
							label="Compactor Model"
							description="Model for summarization"
							value={localValues.compactor as string}
							onChange={(v) => handleChange("compactor", v)}
						/>
						<ConfigField
							label="Cortex Model"
							description="Model for system observation"
							value={localValues.cortex as string}
							onChange={(v) => handleChange("cortex", v)}
						/>
						<ConfigNumberField
							label="Rate Limit Cooldown"
							description="Seconds to deprioritize rate-limited models"
							value={localValues.rate_limit_cooldown_secs as number}
							onChange={(v) => handleChange("rate_limit_cooldown_secs", v)}
							min={0}
							suffix="s"
						/>
					</div>
				);
			case "tuning":
				return (
					<div className="grid gap-4">
						<ConfigNumberField
							label="Max Concurrent Branches"
							description="Maximum branches per channel"
							value={localValues.max_concurrent_branches as number}
							onChange={(v) => handleChange("max_concurrent_branches", v)}
							min={1}
							max={20}
						/>
						<ConfigNumberField
							label="Max Concurrent Workers"
							description="Maximum workers per channel"
							value={localValues.max_concurrent_workers as number}
							onChange={(v) => handleChange("max_concurrent_workers", v)}
							min={1}
							max={20}
						/>
						<ConfigNumberField
							label="Max Turns"
							description="Max LLM turns per channel message"
							value={localValues.max_turns as number}
							onChange={(v) => handleChange("max_turns", v)}
							min={1}
							max={50}
						/>
						<ConfigNumberField
							label="Branch Max Turns"
							description="Max turns for thinking branches"
							value={localValues.branch_max_turns as number}
							onChange={(v) => handleChange("branch_max_turns", v)}
							min={1}
							max={100}
						/>
						<ConfigNumberField
							label="Context Window"
							description="Context window size in tokens"
							value={localValues.context_window as number}
							onChange={(v) => handleChange("context_window", v)}
							min={1000}
							step={1000}
							suffix=" tokens"
						/>
						<ConfigNumberField
							label="History Backfill"
							description="Messages to fetch on new channel"
							value={localValues.history_backfill_count as number}
							onChange={(v) => handleChange("history_backfill_count", v)}
							min={0}
							max={500}
							suffix=" messages"
						/>
					</div>
				);
			case "compaction":
				return (
					<div className="grid gap-4">
						<ConfigFloatField
							label="Background Threshold"
							description="Start background summarization (fraction of context window)"
							value={localValues.background_threshold as number}
							onChange={(v) => handleChange("background_threshold", v)}
							min={0}
							max={1}
							step={0.01}
						/>
						<ConfigFloatField
							label="Aggressive Threshold"
							description="Start aggressive summarization"
							value={localValues.aggressive_threshold as number}
							onChange={(v) => handleChange("aggressive_threshold", v)}
							min={0}
							max={1}
							step={0.01}
						/>
						<ConfigFloatField
							label="Emergency Threshold"
							description="Emergency truncation (no LLM, drop oldest 50%)"
							value={localValues.emergency_threshold as number}
							onChange={(v) => handleChange("emergency_threshold", v)}
							min={0}
							max={1}
							step={0.01}
						/>
					</div>
				);
			case "cortex":
				return (
					<div className="grid gap-4">
						<ConfigNumberField
							label="Tick Interval"
							description="How often the cortex checks system state"
							value={localValues.tick_interval_secs as number}
							onChange={(v) => handleChange("tick_interval_secs", v)}
							min={1}
							suffix="s"
						/>
						<ConfigNumberField
							label="Worker Timeout"
							description="Worker timeout before cancellation"
							value={localValues.worker_timeout_secs as number}
							onChange={(v) => handleChange("worker_timeout_secs", v)}
							min={10}
							suffix="s"
						/>
						<ConfigNumberField
							label="Branch Timeout"
							description="Branch timeout before cancellation"
							value={localValues.branch_timeout_secs as number}
							onChange={(v) => handleChange("branch_timeout_secs", v)}
							min={5}
							suffix="s"
						/>
						<ConfigNumberField
							label="Circuit Breaker"
							description="Consecutive failures before auto-disable"
							value={localValues.circuit_breaker_threshold as number}
							onChange={(v) => handleChange("circuit_breaker_threshold", v)}
							min={1}
							max={10}
						/>
						<ConfigNumberField
							label="Bulletin Interval"
							description="Seconds between memory bulletin refreshes"
							value={localValues.bulletin_interval_secs as number}
							onChange={(v) => handleChange("bulletin_interval_secs", v)}
							min={60}
							suffix="s"
						/>
						<ConfigNumberField
							label="Bulletin Max Words"
							description="Target word count for memory bulletin"
							value={localValues.bulletin_max_words as number}
							onChange={(v) => handleChange("bulletin_max_words", v)}
							min={100}
							max={5000}
							suffix=" words"
						/>
						<ConfigNumberField
							label="Bulletin Max Turns"
							description="Max LLM turns for bulletin generation"
							value={localValues.bulletin_max_turns as number}
							onChange={(v) => handleChange("bulletin_max_turns", v)}
							min={5}
							max={50}
						/>
					</div>
				);
			case "coalesce":
				return (
					<div className="grid gap-4">
						<ConfigToggleField
							label="Enabled"
							description="Enable message coalescing for multi-user channels"
							value={localValues.enabled as boolean}
							onChange={(v) => handleChange("enabled", v)}
						/>
						<ConfigNumberField
							label="Debounce"
							description="Initial debounce window after first message"
							value={localValues.debounce_ms as number}
							onChange={(v) => handleChange("debounce_ms", v)}
							min={100}
							max={10000}
							suffix="ms"
						/>
						<ConfigNumberField
							label="Max Wait"
							description="Maximum time to wait before flushing"
							value={localValues.max_wait_ms as number}
							onChange={(v) => handleChange("max_wait_ms", v)}
							min={500}
							max={30000}
							suffix="ms"
						/>
						<ConfigNumberField
							label="Min Messages"
							description="Min messages to trigger coalesce mode"
							value={localValues.min_messages as number}
							onChange={(v) => handleChange("min_messages", v)}
							min={1}
							max={10}
						/>
						<ConfigToggleField
							label="Multi-User Only"
							description="Apply only to multi-user conversations (skip for DMs)"
							value={localValues.multi_user_only as boolean}
							onChange={(v) => handleChange("multi_user_only", v)}
						/>
					</div>
				);
			case "memory":
				return (
					<div className="grid gap-4">
						<ConfigToggleField
							label="Enabled"
							description="Enable automatic memory persistence branches"
							value={localValues.enabled as boolean}
							onChange={(v) => handleChange("enabled", v)}
						/>
						<ConfigNumberField
							label="Message Interval"
							description="Number of user messages between automatic saves"
							value={localValues.message_interval as number}
							onChange={(v) => handleChange("message_interval", v)}
							min={1}
							max={200}
							suffix=" messages"
						/>
					</div>
				);
			case "browser":
				return (
					<div className="grid gap-4">
						<ConfigToggleField
							label="Enabled"
							description="Enable browser automation tools for workers"
							value={localValues.enabled as boolean}
							onChange={(v) => handleChange("enabled", v)}
						/>
						<ConfigToggleField
							label="Headless"
							description="Run Chrome in headless mode"
							value={localValues.headless as boolean}
							onChange={(v) => handleChange("headless", v)}
						/>
						<ConfigToggleField
							label="JavaScript Evaluation"
							description="Allow JavaScript evaluation via browser tool"
							value={localValues.evaluate_enabled as boolean}
							onChange={(v) => handleChange("evaluate_enabled", v)}
						/>
					</div>
				);
			case "discord":
				return (
					<div className="grid gap-4">
						<ConfigToggleField
							label="Allow Bot Messages"
							description="Process messages from other Discord bots (self-messages are always ignored)"
							value={localValues.allow_bot_messages as boolean}
							onChange={(v) => handleChange("allow_bot_messages", v)}
						/>
					</div>
				);
			default:
				return null;
		}
	};

	return (
		<>
			<div className="flex items-center justify-between border-b border-app-line/50 bg-app-darkBox/20 px-5 py-2.5">
				<div className="flex items-center gap-3">
					<h3 className="text-sm font-medium text-ink">{label}</h3>
					<span className="text-tiny text-ink-faint">{description}</span>
				</div>
			<div className="flex items-center gap-2">
				{dirty && (
					<>
						<Button
							onClick={handleRevert}
							variant="ghost"
							size="sm"
							className="h-6 px-2 text-tiny"
						>
							Revert
						</Button>
						<Button
							onClick={handleSave}
							size="sm"
							loading={saving}
							className="h-6 bg-accent/15 px-2 text-tiny text-accent hover:bg-accent/25"
						>
							Save
						</Button>
					</>
				)}
				{!dirty && <span className="text-tiny text-ink-faint/50">Changes auto-saved to config.toml</span>}
			</div>
			</div>
			<div className="flex-1 overflow-y-auto px-8 py-8">
				<div className="mb-6 rounded-lg border border-app-line/30 bg-app-darkBox/20 px-5 py-4">
					<p className="text-sm leading-relaxed text-ink-dull">{detail}</p>
				</div>
				{renderFields()}
			</div>
		</>
	);
}

// -- Form Field Components --

interface ConfigFieldProps {
	label: string;
	description: string;
	value: string;
	onChange: (value: string) => void;
}

function ConfigField({ label, description, value, onChange }: ConfigFieldProps) {
	return (
		<div className="flex flex-col gap-1.5">
			<label className="text-sm font-medium text-ink">{label}</label>
			<p className="text-tiny text-ink-faint">{description}</p>
			<input
				type="text"
				value={value}
				onChange={(e) => onChange(e.target.value)}
				className="mt-1 rounded-md border border-app-line/50 bg-app-darkBox/30 px-3 py-2 text-sm text-ink-dull focus:border-accent/30 focus:outline-none"
			/>
		</div>
	);
}

interface ConfigNumberFieldProps {
	label: string;
	description: string;
	value: number;
	onChange: (value: number) => void;
	min?: number;
	max?: number;
	step?: number;
	suffix?: string;
}

function ConfigNumberField({ label, description, value, onChange, min, max, step = 1, suffix }: ConfigNumberFieldProps) {
	const safeValue = value ?? 0;

	const clamp = (v: number) => {
		if (min !== undefined && v < min) return min;
		if (max !== undefined && v > max) return max;
		return v;
	};

	const increment = () => onChange(clamp(safeValue + step));
	const decrement = () => onChange(clamp(safeValue - step));

	const handleInput = (e: React.ChangeEvent<HTMLInputElement>) => {
		const raw = e.target.value;
		if (raw === "" || raw === "-") return;
		const parsed = Number(raw);
		if (!Number.isNaN(parsed)) onChange(clamp(parsed));
	};

	return (
		<div className="flex flex-col gap-1.5">
			<label className="text-sm font-medium text-ink">{label}</label>
			<p className="text-tiny text-ink-faint">{description}</p>
			<div className="flex items-center gap-2.5 mt-1">
			<div className="flex items-stretch rounded-md border border-app-line/50 bg-app-darkBox/30 overflow-hidden">
				<Button
					type="button"
					onClick={decrement}
					variant="ghost"
					size="icon"
					className="h-8 w-8 rounded-none"
				>
					<HugeiconsIcon icon={MinusSignIcon} className="h-3 w-3" />
				</Button>
				<input
					type="text"
					inputMode="numeric"
					value={safeValue}
					onChange={handleInput}
					className="w-20 border-x border-app-line/50 bg-transparent px-2 py-1.5 text-center font-mono text-sm text-ink-dull focus:outline-none"
				/>
				<Button
					type="button"
					onClick={increment}
					variant="ghost"
					size="icon"
					className="h-8 w-8 rounded-none"
				>
					<HugeiconsIcon icon={PlusSignIcon} className="h-3 w-3" />
				</Button>
			</div>
				{suffix && <span className="text-tiny text-ink-faint">{suffix}</span>}
			</div>
		</div>
	);
}

interface ConfigFloatFieldProps {
	label: string;
	description: string;
	value: number;
	onChange: (value: number) => void;
	min?: number;
	max?: number;
	step?: number;
}

function ConfigFloatField({ label, description, value, onChange, min = 0, max = 1, step = 0.01 }: ConfigFloatFieldProps) {
	const safeValue = value ?? 0;

	const clamp = (v: number) => {
		if (v < min) return min;
		if (v > max) return max;
		return Math.round(v / step) * step;
	};

	const increment = () => onChange(clamp(safeValue + step));
	const decrement = () => onChange(clamp(safeValue - step));

	const handleInput = (e: React.ChangeEvent<HTMLInputElement>) => {
		const raw = e.target.value;
		if (raw === "" || raw === "0." || raw === ".") return;
		const parsed = Number(raw);
		if (!Number.isNaN(parsed)) onChange(clamp(parsed));
	};

	const pct = ((safeValue - min) / (max - min)) * 100;

	return (
		<div className="flex flex-col gap-1.5">
			<label className="text-sm font-medium text-ink">{label}</label>
			<p className="text-tiny text-ink-faint">{description}</p>
			<div className="flex items-center gap-3 mt-1">
			<div className="flex items-stretch rounded-md border border-app-line/50 bg-app-darkBox/30 overflow-hidden">
				<Button
					type="button"
					onClick={decrement}
					variant="ghost"
					size="icon"
					className="h-8 w-8 rounded-none"
				>
					<HugeiconsIcon icon={MinusSignIcon} className="h-3 w-3" />
				</Button>
				<input
					type="text"
					inputMode="decimal"
					value={safeValue.toFixed(2)}
					onChange={handleInput}
					className="w-16 border-x border-app-line/50 bg-transparent px-1.5 py-1.5 text-center font-mono text-sm text-ink-dull focus:outline-none"
				/>
				<Button
					type="button"
					onClick={increment}
					variant="ghost"
					size="icon"
					className="h-8 w-8 rounded-none"
				>
					<HugeiconsIcon icon={PlusSignIcon} className="h-3 w-3" />
				</Button>
			</div>
				{/* Progress bar */}
				<div className="h-1.5 w-32 overflow-hidden rounded-full bg-app-darkBox">
					<div
						className="h-full rounded-full bg-accent/50 transition-all"
						style={{ width: `${pct}%` }}
					/>
				</div>
			</div>
		</div>
	);
}

interface ConfigToggleFieldProps {
	label: string;
	description: string;
	value: boolean;
	onChange: (value: boolean) => void;
}

function ConfigToggleField({ label, description, value, onChange }: ConfigToggleFieldProps) {
	return (
		<div className="flex items-center justify-between py-2">
			<div className="flex flex-col gap-0.5">
				<label className="text-sm font-medium text-ink">{label}</label>
				<p className="text-tiny text-ink-faint">{description}</p>
			</div>
			<Button
				onClick={() => onChange(!value)}
				variant="ghost"
				size="icon"
				className={`relative h-6 w-11 rounded-full p-0 ${
					value ? "bg-accent/60" : "bg-app-darkBox"
				}`}
			>
				<span
					className={`absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white transition-transform ${
						value ? "translate-x-5" : "translate-x-0"
					}`}
				/>
			</Button>
		</div>
	);
}
