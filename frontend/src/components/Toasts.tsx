"use client";

import { useApp } from "@/context/AppContext";

const KIND = { info: "bg-ink", success: "bg-[#0f3d17]", error: "bg-[#6d1510]" } as const;

export function Toasts() {
  const { toasts } = useApp();
  return (
    <div aria-live="polite" className="fixed bottom-5 right-5 z-50 grid max-w-[min(420px,calc(100vw-40px))] gap-2">
      {toasts.map((t) => (
        <div key={t.id} className={`animate-feed-in rounded-md px-3.5 py-2.5 text-[13px] text-white shadow-float ${KIND[t.kind]}`}>
          {t.message}
        </div>
      ))}
    </div>
  );
}
