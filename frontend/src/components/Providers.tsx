"use client";

import type { ReactNode } from "react";

import { AppProvider } from "@/context/AppContext";
import { ReplayProvider } from "@/context/ReplayContext";
import { ThresholdDraftProvider } from "@/context/ThresholdDraftContext";

export function Providers({ children }: { children: ReactNode }) {
  return (
    <AppProvider>
      <ThresholdDraftProvider>
        <ReplayProvider>{children}</ReplayProvider>
      </ThresholdDraftProvider>
    </AppProvider>
  );
}
