import type { Metadata } from "next";

import { PerformanceScreen } from "@/components/screens/PerformanceScreen";

export const metadata: Metadata = { title: "Hiệu năng mô hình" };

export default function PerformancePage() {
  return <PerformanceScreen />;
}
