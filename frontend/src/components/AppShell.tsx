"use client";

/*
 * Khung chung (docs/07 §8): thanh bên 4 mục, chân thanh bên luôn hiện model_version (NFR-11),
 * thanh trên luôn hiện ngưỡng hiện hành (UI-D4) và tình trạng API/CSDL.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

import { useApp } from "@/context/AppContext";
import { apiBase } from "@/lib/api";
import { fmt } from "@/lib/format";
import { SCREENS } from "@/lib/labels";

import { SamplesDialog } from "./SamplesDialog";
import { Toasts } from "./Toasts";
import { TxDrawer } from "./TxDrawer";

const ICONS: Record<string, ReactNode> = {
  queue: <path d="M3 5h14M3 10h14M3 15h9" />,
  threshold: (
    <>
      <path d="M3 10h14" />
      <circle cx="8" cy="10" r="2.6" />
    </>
  ),
  performance: <path d="M3 17V3M3 17h14M6 13l3.5-4 3 2.5L17 5" />,
  replay: <path d="M6 4.5v11l9-5.5z" />,
};

/** "/threshold/" (trailingSlash) và "/threshold" là một. */
function normalize(pathname: string | null): string {
  if (!pathname || pathname === "/") return "/";
  return pathname.replace(/\/+$/, "");
}

export function AppShell({ children }: { children: ReactNode }) {
  const path = normalize(usePathname());
  const screen = SCREENS.find((s) => s.href === path) ?? SCREENS[0];

  return (
    <div className="grid min-h-screen grid-cols-1 md:grid-cols-[228px_minmax(0,1fr)]">
      <a href="#main" className="absolute -top-12 left-3 z-[100] rounded-md bg-surface px-3 py-2 shadow-float focus:top-3">
        Bỏ qua, tới nội dung chính
      </a>
      <Sidebar path={path} />
      <div className="flex min-w-0 flex-col">
        <Topbar title={screen.title} onThresholdScreen={path === "/threshold"} />
        <Banners />
        <main id="main" tabIndex={-1} className="px-4 pb-12 pt-5 outline-none md:px-7">
          <div className="max-w-[1320px]">{children}</div>
        </main>
      </div>
      <TxDrawer />
      <SamplesDialog />
      <Toasts />
    </div>
  );
}

function Sidebar({ path }: { path: string }) {
  const { modelVersion, trainedAt } = useApp();
  return (
    <aside className="flex flex-wrap items-center gap-x-4 gap-y-2 bg-sidebar px-4 py-2.5 text-sidebar-ink md:sticky md:top-0 md:h-screen md:flex-col md:flex-nowrap md:items-stretch md:gap-0 md:px-3 md:pb-4 md:pt-5 [&_:focus-visible]:outline-[#8fb8ef]">
      <div className="flex items-center gap-2.5 text-base font-bold leading-tight md:px-2 md:pb-5">
        <svg viewBox="0 0 32 32" aria-hidden="true" className="size-[30px] flex-none">
          <rect width="32" height="32" rx="7" fill="#1c5cab" />
          <path d="M16 7 L25 24 H7 Z" fill="none" stroke="#fff" strokeWidth="3" strokeLinejoin="round" />
        </svg>
        <span>
          Fraud
          <br />
          Console
        </span>
      </div>
      <nav aria-label="Màn hình">
        <ul className="grid grid-flow-col gap-0.5 md:grid-flow-row">
          {SCREENS.map((s) => {
            const active = s.href === path;
            return (
              <li key={s.key}>
                <Link
                  href={s.href}
                  aria-current={active ? "page" : undefined}
                  className={`flex items-center gap-2.5 rounded-md px-2.5 py-2 font-medium text-sidebar-ink no-underline hover:bg-white/5 ${
                    active ? "bg-sidebar-active shadow-[inset_3px_0_0_#6ea8f0]" : ""
                  }`}
                >
                  <svg viewBox="0 0 20 20" aria-hidden="true" className="size-[18px] flex-none fill-none stroke-current [stroke-linecap:round] [stroke-linejoin:round] [stroke-width:1.8]">
                    {ICONS[s.key]}
                  </svg>
                  {s.label}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
      <div className="ml-auto text-xs text-sidebar-muted md:ml-0 md:mt-auto md:border-t md:border-white/10 md:px-2 md:pt-3.5">
        <div className="mb-0.5 font-semibold uppercase tracking-wide">Mô hình đang phục vụ</div>
        <div className="mb-0.5 break-all font-mono text-sidebar-ink">{modelVersion ?? "—"}</div>
        <div>Huấn luyện {fmt.date(trainedAt)}</div>
      </div>
    </aside>
  );
}

function Topbar({ title, onThresholdScreen }: { title: string; onThresholdScreen: boolean }) {
  const { health, threshold } = useApp();
  const healthText =
    health.state === "ok" ? "API và CSDL hoạt động"
      : health.state === "loading" ? "Đang kết nối API…"
        : health.code === "DATABASE_UNAVAILABLE" ? "Mất kết nối CSDL"
          : health.code === "MODEL_NOT_LOADED" ? "Chưa nạp mô hình"
            : "Không kết nối được API";
  const dot = health.state === "ok" ? "bg-good" : health.state === "loading" ? "bg-axis" : "bg-critical";

  return (
    <header className="sticky top-0 z-10 flex flex-wrap items-center justify-between gap-4 border-b border-line bg-page/90 px-4 py-3.5 backdrop-blur md:px-7">
      <h1 className="text-xl font-bold">{title}</h1>
      <div className="flex flex-wrap items-center justify-end gap-3.5">
        <div role="status" className={`inline-flex items-center gap-1.5 text-[12.5px] ${health.state === "error" ? "font-semibold text-critical-text" : "text-ink-2"}`}>
          <span aria-hidden="true" className={`size-2 rounded-full ${dot}`} />
          {healthText}
        </div>
        <div className="inline-flex items-center gap-2 rounded-full border border-line-strong bg-surface py-1 pl-3 pr-1.5">
          <span className="text-[12.5px] text-ink-2">Ngưỡng</span>
          <strong className="text-[15px] tabular-nums">{fmt.tau(threshold?.current)}</strong>
          {threshold && (
            <span className="text-[11.5px] text-muted">{threshold.source === "user" ? "đã đặt" : "τ* mặc định"}</span>
          )}
          {!onThresholdScreen && (
            <Link href="/threshold" className="btn btn-sm">
              Đổi
            </Link>
          )}
        </div>
      </div>
    </header>
  );
}

function Banners() {
  const { health, checkHealth, modelChanged, scoresError, loadScores } = useApp();
  return (
    <div className="px-4 md:px-7">
      {health.state === "error" && (
        <div role="alert" className="mt-3.5 flex items-start justify-between gap-4 rounded-md border border-[#efb9b6] bg-critical-soft px-3.5 py-3 text-[#5c1410]">
          <div>
            <strong>{health.message}</strong>
            {health.code === "NETWORK" && (
              <p className="mt-1">
                Chạy API trong một cửa sổ khác: <code>.\.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000</code> — địa
                chỉ đang gọi: <code>{apiBase()}</code>
              </p>
            )}
            {health.code === "DATABASE_UNAVAILABLE" && (
              <p className="mt-1">
                PostgreSQL không trả lời: <code>docker compose up -d db</code>. Màn hình Hiệu năng và thanh trượt ngưỡng vẫn dùng được.
              </p>
            )}
            {health.code === "MODEL_NOT_LOADED" && (
              <p className="mt-1">
                Thiếu hoặc hỏng hiện vật — chạy <code>notebooks/08_export_artifacts.ipynb</code>. {health.reason}
              </p>
            )}
          </div>
          <button type="button" className="btn btn-sm" onClick={() => checkHealth()}>
            Thử lại
          </button>
        </div>
      )}
      {modelChanged && (
        <div role="alert" className="mt-3.5 flex items-start justify-between gap-4 rounded-md border border-[#f1d58b] bg-warning-soft px-3.5 py-3 text-[#4d3500]">
          <div>
            <strong>Mô hình phía API vừa đổi.</strong> Tải lại trang để mọi con số dùng mô hình mới.
          </div>
          <button type="button" className="btn btn-sm" onClick={() => window.location.reload()}>
            Tải lại
          </button>
        </div>
      )}
      {scoresError && health.state === "ok" && (
        <div role="alert" className="mt-3.5 flex items-start justify-between gap-4 rounded-md border border-[#f1d58b] bg-warning-soft px-3.5 py-3 text-[#4d3500]">
          <div>
            <strong>Chưa tải được điểm của tập kiểm thử:</strong> {scoresError}
          </div>
          <button type="button" className="btn btn-sm" onClick={() => loadScores()}>
            Thử lại
          </button>
        </div>
      )}
    </div>
  );
}
