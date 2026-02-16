"use client";

import * as React from "react";
import * as CheckboxPrimitive from "@radix-ui/react-checkbox";
import { Tick02Icon, MinusSignIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { cva, type VariantProps } from "class-variance-authority";
import { cx } from "./utils";

const checkboxStyles = cva(
  [
    "peer h-4 w-4 shrink-0 rounded border border-app-line",
    "ring-offset-app-box focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2",
    "disabled:cursor-not-allowed disabled:opacity-50",
    "data-[state=checked]:bg-accent data-[state=checked]:border-accent",
    "data-[state=indeterminate]:bg-accent data-[state=indeterminate]:border-accent",
  ]
);

const indicatorStyles = cva(
  "flex items-center justify-center text-current"
);

export interface CheckboxProps
  extends React.ComponentPropsWithoutRef<typeof CheckboxPrimitive.Root>,
    VariantProps<typeof checkboxStyles> {
  indeterminate?: boolean;
}

export const Checkbox = React.forwardRef<
  React.ElementRef<typeof CheckboxPrimitive.Root>,
  CheckboxProps
>(({ className, indeterminate, checked, ...props }, ref) => (
  <CheckboxPrimitive.Root
    ref={ref}
    className={cx(checkboxStyles(), className)}
    checked={indeterminate ? "indeterminate" : checked}
    {...props}
  >
    <CheckboxPrimitive.Indicator
      className={cx(indicatorStyles())}
    >
      {indeterminate ? (
        <HugeiconsIcon icon={MinusSignIcon} className="h-3 w-3 text-white" />
      ) : (
        <HugeiconsIcon icon={Tick02Icon} className="h-3 w-3 text-white" />
      )}
    </CheckboxPrimitive.Indicator>
  </CheckboxPrimitive.Root>
));

Checkbox.displayName = CheckboxPrimitive.Root.displayName;

export interface CheckboxLabelProps {
  children: React.ReactNode;
  disabled?: boolean;
  className?: string;
}

export const CheckboxLabel: React.FC<CheckboxLabelProps> = ({
  children,
  disabled,
  className,
}) => (
  <span
    className={cx(
      "text-sm font-medium text-ink",
      disabled && "opacity-50",
      className
    )}
  >
    {children}
  </span>
);

export interface CheckboxFieldProps {
  label: React.ReactNode;
  description?: string;
  disabled?: boolean;
  className?: string;
  checkboxProps?: Omit<CheckboxProps, "disabled">;
}

export const CheckboxField: React.FC<CheckboxFieldProps> = ({
  label,
  description,
  disabled,
  className,
  checkboxProps,
}) => (
  <label
    className={cx(
      "flex items-start gap-3 cursor-pointer",
      disabled && "cursor-not-allowed",
      className
    )}
  >
    <div className="mt-0.5">
      <Checkbox disabled={disabled} {...checkboxProps} />
    </div>
    <div className="space-y-1">
      <CheckboxLabel disabled={disabled}>{label}</CheckboxLabel>
      {description && (
        <p className={cx("text-xs text-ink-dull", disabled && "opacity-50")}>
          {description}
        </p>
      )}
    </div>
  </label>
);
