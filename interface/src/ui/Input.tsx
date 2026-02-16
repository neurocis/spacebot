import * as React from "react";
import {cva, type VariantProps} from "class-variance-authority";
import {cx} from "./utils";
import {Search01Icon} from "@hugeicons/core-free-icons";
import {HugeiconsIcon} from "@hugeicons/react";

export const inputSizes = {
	sm: "h-8 text-sm px-3 py-1.5",
	md: "h-9 text-sm px-3",
	lg: "h-10 text-base px-3.5",
} as const;

export const inputStyles = cva(
	[
		"rounded-md border text-sm leading-4",
		"outline-none transition-colors",
		"text-ink",
	],
	{
		variants: {
			variant: {
				default: ["border-app-line bg-app-darkBox", "focus-within:border-accent/50"],
				transparent: [
					"border-transparent bg-app-box",
					"focus-within:border-app-line",
				],
			},
			error: {
				true: "border-red-500/50 focus-within:border-red-500",
			},
			size: inputSizes,
		},
		defaultVariants: {
			variant: "default",
			size: "sm",
		},
	},
);

type InputVariants = VariantProps<typeof inputStyles>;

export interface InputProps
	extends
		Omit<React.InputHTMLAttributes<HTMLInputElement>, "size">,
		InputVariants {
	icon?: React.ReactNode;
	right?: React.ReactNode;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
	({className, variant, size, error, icon, right, ...props}, ref) => {
		return (
			<div
				className={cx(
					inputStyles({variant, size, error}),
					"flex items-center gap-2",
					className,
				)}
			>
				{icon && <span className="text-ink-faint shrink-0">{icon}</span>}
				<input
					ref={ref}
					className="bg-transparent w-full outline-none placeholder:text-ink-faint"
					{...props}
				/>
				{right && <span className="shrink-0">{right}</span>}
			</div>
		);
	},
);

Input.displayName = "Input";

export interface TextAreaProps
	extends
		React.TextareaHTMLAttributes<HTMLTextAreaElement>,
		Pick<InputVariants, "variant" | "error"> {}

export const TextArea = React.forwardRef<HTMLTextAreaElement, TextAreaProps>(
	({className, variant = "default", error, ...props}, ref) => {
		return (
			<textarea
				ref={ref}
				className={cx(
					inputStyles({variant, error}),
					"w-full min-h-[80px] resize-y py-2 placeholder:text-ink-faint",
					className,
				)}
				{...props}
			/>
		);
	},
);

TextArea.displayName = "TextArea";

export const SearchInput = React.forwardRef<
	HTMLInputElement,
	Omit<InputProps, "icon">
>(({size = "sm", ...props}, ref) => (
	<Input
		ref={ref}
		icon={
			<HugeiconsIcon
				icon={Search01Icon}
				size={size === "sm" ? 14 : size === "md" ? 16 : 18}
			/>
		}
		size={size}
		{...props}
	/>
));

SearchInput.displayName = "SearchInput";

export const Label = React.forwardRef<
	HTMLLabelElement,
	React.LabelHTMLAttributes<HTMLLabelElement>
>(({className, ...props}, ref) => (
	<label
		ref={ref}
		className={cx("block text-xs font-medium text-ink-dull mb-1.5", className)}
		{...props}
	/>
));

Label.displayName = "Label";

export const PasswordInput = React.forwardRef<
	HTMLInputElement,
	Omit<InputProps, "type" | "right">
>(({...props}, ref) => {
	const [show, setShow] = React.useState(false);

	return (
		<Input
			ref={ref}
			type={show ? "text" : "password"}
			right={
				<button
					type="button"
					onClick={() => setShow(!show)}
					className="text-xs text-ink-dull hover:text-ink transition-colors"
				>
					{show ? "Hide" : "Show"}
				</button>
			}
			{...props}
		/>
	);
});

PasswordInput.displayName = "PasswordInput";
