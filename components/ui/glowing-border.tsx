"use client";

import { cn } from "@/lib/utils";

export function GlowingBorder({
  children,
  className,
  glowColor = "rgba(34, 211, 238, 0.4)",
  borderRadius = "0.75rem",
}: {
  children: React.ReactNode;
  className?: string;
  glowColor?: string;
  borderRadius?: string;
}) {
  return (
    <div
      className={cn("relative p-px", className)}
      style={{ borderRadius }}
    >
      <div
        className="absolute inset-0"
        style={{
          borderRadius,
          background: `conic-gradient(from 0deg, transparent, ${glowColor}, transparent, transparent)`,
        }}
      />
      <div
        className="relative h-full w-full bg-black/60 backdrop-blur-xl"
        style={{ borderRadius }}
      >
        {children}
      </div>
    </div>
  );
}
