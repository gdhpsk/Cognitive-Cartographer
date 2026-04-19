"use client";

import { useEffect, useState } from "react";
import { motion, stagger, useAnimate } from "motion/react";
import { cn } from "@/lib/utils";

export function TextGenerateEffect({
  text,
  className,
}: {
  text: string;
  className?: string;
}) {
  const [scope, animate] = useAnimate();
  const [hasAnimated, setHasAnimated] = useState(false);
  const words = text.split(" ");

  useEffect(() => {
    if (hasAnimated) return;
    setHasAnimated(true);
    void animate(
      "span",
      { opacity: 1, filter: "blur(0px)" },
      { duration: 0.3, delay: stagger(0.02) }
    );
  }, [animate, hasAnimated]);

  return (
    <div ref={scope} className={cn("text-sm leading-relaxed", className)}>
      {words.map((word, idx) => (
        <motion.span
          key={`${word}-${idx}`}
          className="inline-block text-white/90"
          style={{ opacity: 0, filter: "blur(4px)" }}
        >
          {word}{idx < words.length - 1 ? "\u00A0" : ""}
        </motion.span>
      ))}
    </div>
  );
}
