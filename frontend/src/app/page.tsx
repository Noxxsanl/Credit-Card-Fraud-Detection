import type { Metadata } from "next";

import { QueueScreen } from "@/components/screens/QueueScreen";

export const metadata: Metadata = { title: "Hàng đợi thẩm định" };

/** UI-01 là màn hình mặc định khi mở ứng dụng (docs/07 §3). */
export default function QueuePage() {
  return <QueueScreen />;
}
