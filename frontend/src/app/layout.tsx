import type { Metadata } from "next";

import { AppShell } from "@/components/AppShell";
import { Providers } from "@/components/Providers";

import "./globals.css";

/*
 * Layout gốc: khung chung và các context sống qua mọi lần chuyển trang (docs/07 §8, §9).
 * Không dùng next/font/google: phông hệ thống, không tải gì từ Internet (NFR-08).
 */

export const metadata: Metadata = {
  title: { default: "Fraud Review Console", template: "%s · Fraud Review Console" },
  description: "Bảng điều khiển thẩm định gian lận thẻ tín dụng — đồ án phát hiện gian lận",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="vi">
      <body>
        <Providers>
          <AppShell>{children}</AppShell>
        </Providers>
      </body>
    </html>
  );
}
