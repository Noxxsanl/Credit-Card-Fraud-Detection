import type { Metadata } from "next";

import { ReplayScreen } from "@/components/screens/ReplayScreen";

export const metadata: Metadata = { title: "Chế độ phát lại" };

export default function ReplayPage() {
  return <ReplayScreen />;
}
