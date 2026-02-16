"use client";

import * as React from "react";
import * as RadioGroupPrimitive from "@radix-ui/react-radio-group";
import { CircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { cva, type VariantProps } from "class-variance-authority";
import { cx } from "./utils";

const radioGroupStyles = cva("grid gap-3");

const radioGroupItemStyles = cva(
  [
    "aspect-square h-4 w-4 rounded-full border border-app-line",
    "text-accent ring-offset-app-box focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2",
    "disabled:cursor-not-allowed disabled:opacity-50",
    "data-[state=checked]:border-accent",
  ]
);

const radioIndicatorStyles = cva("flex items-center justify-center");

export interface RadioGroupProps
  extends React.ComponentPropsWithoutRef<typeof RadioGroupPrimitive.Root>,
    VariantProps<typeof radioGroupStyles> {}

export const RadioGroup = React.forwardRef<
  React.ElementRef<typeof RadioGroupPrimitive.Root>,
  RadioGroupProps
>(({ className, ...props }, ref) => {
  return (
    <RadioGroupPrimitive.Root
      className={cx(radioGroupStyles(), className)}
      {...props}
      ref={ref}
    />
  );
});

RadioGroup.displayName = RadioGroupPrimitive.Root.displayName;

export interface RadioGroupItemProps
  extends React.ComponentPropsWithoutRef<typeof RadioGroupPrimitive.Item>,
    VariantProps<typeof radioGroupItemStyles> {}

export const RadioGroupItem = React.forwardRef<
  React.ElementRef<typeof RadioGroupPrimitive.Item>,
  RadioGroupItemProps
>(({ className, ...props }, ref) => {
  return (
    <RadioGroupPrimitive.Item
      ref={ref}
      className={cx(radioGroupItemStyles(), className)}
      {...props}
    >
      <RadioGroupPrimitive.Indicator className={cx(radioIndicatorStyles())}>
        <HugeiconsIcon icon={CircleIcon} className="h-2 w-2 fill-current text-current" fill="currentColor" />
      </RadioGroupPrimitive.Indicator>
    </RadioGroupPrimitive.Item>
  );
});

RadioGroupItem.displayName = RadioGroupPrimitive.Item.displayName;

export interface RadioLabelProps {
  children: React.ReactNode;
  disabled?: boolean;
  className?: string;
}

export const RadioLabel: React.FC<RadioLabelProps> = ({
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

export interface RadioGroupFieldProps {
  value: string;
  label: React.ReactNode;
  description?: string;
  disabled?: boolean;
  className?: string;
}

export const RadioGroupField: React.FC<RadioGroupFieldProps> = ({
  value,
  label,
  description,
  disabled,
  className,
}) => (
  <label
    className={cx(
      "flex items-start gap-3 cursor-pointer",
      disabled && "cursor-not-allowed",
      className
    )}
  >
    <div className="mt-0.5">
      <RadioGroupItem value={value} disabled={disabled} />
    </div>
    <div className="space-y-1">
      <RadioLabel disabled={disabled}>{label}</RadioLabel>
      {description && (
        <p className={cx("text-xs text-ink-dull", disabled && "opacity-50")}>
          {description}
        </p>
      )}
    </div>
  </label>
);
