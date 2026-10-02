import type { Metadata } from "next";

import { ThresholdScreen } from "@/components/screens/ThresholdScreen";

export const metadata: Metadata = { title: "Cấu hình ngưỡng" };

export default function ThresholdPage() {
  return <ThresholdScreen />;
}
