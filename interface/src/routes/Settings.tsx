import {useState} from "react";
import {useQuery, useMutation, useQueryClient} from "@tanstack/react-query";
import {api} from "@/api/client";
import {Button, Input, SettingSidebarButton, Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter} from "@/ui";

type SectionId = "providers";

const SECTIONS = [
	{
		id: "providers" as const,
		label: "Providers",
		group: "general" as const,
		description: "LLM provider API keys",
	},
] satisfies {
	id: SectionId;
	label: string;
	group: string;
	description: string;
}[];

const PROVIDERS = [
	{
		id: "anthropic",
		name: "Anthropic",
		description: "Claude models (Sonnet, Opus, Haiku)",
		placeholder: "sk-ant-...",
		envVar: "ANTHROPIC_API_KEY",
	},
	{
		id: "openrouter",
		name: "OpenRouter",
		description: "Multi-provider gateway with unified API",
		placeholder: "sk-or-...",
		envVar: "OPENROUTER_API_KEY",
	},
	{
		id: "openai",
		name: "OpenAI",
		description: "GPT models",
		placeholder: "sk-...",
		envVar: "OPENAI_API_KEY",
	},
	{
		id: "zhipu",
		name: "Z.ai (GLM)",
		description: "GLM models (GLM-4, GLM-4-Flash)",
		placeholder: "...",
		envVar: "ZHIPU_API_KEY",
	},
] as const;

export function Settings() {
	const queryClient = useQueryClient();
	const [activeSection, setActiveSection] = useState<SectionId>("providers");
	const [editingProvider, setEditingProvider] = useState<string | null>(null);
	const [keyInput, setKeyInput] = useState("");
	const [message, setMessage] = useState<{
		text: string;
		type: "success" | "error";
	} | null>(null);

	const {data, isLoading} = useQuery({
		queryKey: ["providers"],
		queryFn: api.providers,
		staleTime: 5_000,
	});

	const updateMutation = useMutation({
		mutationFn: ({provider, apiKey}: {provider: string; apiKey: string}) =>
			api.updateProvider(provider, apiKey),
		onSuccess: (result) => {
			if (result.success) {
				setEditingProvider(null);
				setKeyInput("");
				setMessage({text: result.message, type: "success"});
				queryClient.invalidateQueries({queryKey: ["providers"]});
				// Agents will auto-start on the backend, refetch agent list after a short delay
				setTimeout(() => {
					queryClient.invalidateQueries({queryKey: ["agents"]});
					queryClient.invalidateQueries({queryKey: ["overview"]});
				}, 3000);
			} else {
				setMessage({text: result.message, type: "error"});
			}
		},
		onError: (error) => {
			setMessage({text: `Failed: ${error.message}`, type: "error"});
		},
	});

	const removeMutation = useMutation({
		mutationFn: (provider: string) => api.removeProvider(provider),
		onSuccess: (result) => {
			if (result.success) {
				setMessage({text: result.message, type: "success"});
				queryClient.invalidateQueries({queryKey: ["providers"]});
			} else {
				setMessage({text: result.message, type: "error"});
			}
		},
		onError: (error) => {
			setMessage({text: `Failed: ${error.message}`, type: "error"});
		},
	});

	const editingProviderData = PROVIDERS.find((p) => p.id === editingProvider);

	const handleSave = () => {
		if (!keyInput.trim() || !editingProvider) return;
		updateMutation.mutate({provider: editingProvider, apiKey: keyInput.trim()});
	};

	const handleClose = () => {
		setEditingProvider(null);
		setKeyInput("");
	};

	const isConfigured = (providerId: string): boolean => {
		if (!data) return false;
		return data.providers[providerId as keyof typeof data.providers] ?? false;
	};

	return (
		<div className="flex h-full">
			{/* Sidebar */}
			<div className="flex w-52 flex-shrink-0 flex-col border-r border-app-line/50 bg-app-darkBox/20 overflow-y-auto">
				<div className="px-3 pb-1 pt-4">
					<span className="text-tiny font-medium uppercase tracking-wider text-ink-faint">
						Settings
					</span>
				</div>
				<div className="flex flex-col gap-0.5 px-2">
					{SECTIONS.map((section) => (
						<SettingSidebarButton
							key={section.id}
							onClick={() => setActiveSection(section.id)}
							active={activeSection === section.id}
						>
							<span className="flex-1">{section.label}</span>
						</SettingSidebarButton>
					))}
				</div>
			</div>

			{/* Content */}
			<div className="flex flex-1 flex-col overflow-hidden">
				<header className="flex h-12 items-center border-b border-app-line bg-app-darkBox/50 px-6">
					<h1 className="font-plex text-sm font-medium text-ink">
						{SECTIONS.find((s) => s.id === activeSection)?.label}
					</h1>
				</header>
				<div className="flex-1 overflow-y-auto">
					<div className="mx-auto max-w-2xl px-6 py-6">
						{/* Section header */}
						<div className="mb-6">
							<h2 className="font-plex text-sm font-semibold text-ink">
								LLM Providers
							</h2>
							<p className="mt-1 text-sm text-ink-dull">
								Configure API keys for LLM providers. At least one provider is
								required for agents to function.
							</p>
						</div>

						{isLoading ? (
							<div className="flex items-center gap-2 text-ink-dull">
								<div className="h-2 w-2 animate-pulse rounded-full bg-accent" />
								Loading providers...
							</div>
						) : (
							<div className="flex flex-col gap-3">
								{PROVIDERS.map((provider) => {
									const configured = isConfigured(provider.id);

									return (
										<div
											key={provider.id}
											className="rounded-lg border border-app-line bg-app-box p-4"
										>
											<div className="flex items-center justify-between">
												<div className="flex-1">
													<span className="text-sm font-medium text-ink">
														{provider.name}
													</span>
													<p className="mt-0.5 text-sm text-ink-dull">
														{provider.description}
													</p>
												</div>
												<div className="flex gap-2">
													<Button
														onClick={() => {
															setEditingProvider(provider.id);
															setKeyInput("");
															setMessage(null);
														}}
														variant="outline"
														size="sm"
													>
														{configured ? "Update" : "Add key"}
													</Button>
													{configured && (
														<Button
															onClick={() =>
																removeMutation.mutate(provider.id)
															}
															variant="outline"
															size="sm"
															loading={removeMutation.isPending}
														>
															Remove
														</Button>
													)}
												</div>
											</div>
										</div>
									);
								})}
							</div>
						)}

						{/* Info note */}
						<div className="mt-6 rounded-md border border-app-line bg-app-darkBox/20 px-4 py-3">
							<p className="text-sm text-ink-faint">
								Keys are written to{" "}
								<code className="rounded bg-app-box px-1 py-0.5 text-tiny text-ink-dull">
									config.toml
								</code>{" "}
								in your instance directory. You can also set them via
								environment variables (
								<code className="rounded bg-app-box px-1 py-0.5 text-tiny text-ink-dull">
									ANTHROPIC_API_KEY
								</code>
								, etc.).
							</p>
						</div>
					</div>
				</div>
			</div>

			<Dialog open={!!editingProvider} onOpenChange={(open) => { if (!open) handleClose(); }}>
				<DialogContent className="max-w-md">
					<DialogHeader>
						<DialogTitle>{isConfigured(editingProvider ?? "") ? "Update" : "Add"} API Key</DialogTitle>
						<DialogDescription>
							Enter your {editingProviderData?.name} API key. It will be saved to your instance config.
						</DialogDescription>
					</DialogHeader>
					<Input
						type="password"
						value={keyInput}
						onChange={(e) => setKeyInput(e.target.value)}
						placeholder={editingProviderData?.placeholder}
						autoFocus
						onKeyDown={(e) => {
							if (e.key === "Enter") handleSave();
						}}
					/>
					{message && (
						<div
							className={`rounded-md border px-3 py-2 text-sm ${
								message.type === "success"
									? "border-green-500/20 bg-green-500/10 text-green-400"
									: "border-red-500/20 bg-red-500/10 text-red-400"
							}`}
						>
							{message.text}
						</div>
					)}
					<DialogFooter>
						<Button onClick={handleClose} variant="ghost" size="sm">
							Cancel
						</Button>
						<Button
							onClick={handleSave}
							disabled={!keyInput.trim()}
							loading={updateMutation.isPending}
							size="sm"
						>
							Save
						</Button>
					</DialogFooter>
				</DialogContent>
			</Dialog>
		</div>
	);
}
