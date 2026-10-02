"use client";

/*
 * UI-05 — Chế độ phát lại (docs/07 §7). Bộ máy phát lại nằm ở context/ReplayContext.tsx để
 * luồng vẫn chạy khi chuyển sang màn hình khác; trang này chỉ hiển thị và điều khiển.
 */

import { useApp } from "@/context/AppContext";
import { SPEEDS, useReplay, type ReplayStatus } from "@/context/ReplayContext";
import { fmt } from "@/lib/format";

import { RiskScore, Tile } from "../ui";

const STATUS_TEXT: Record<ReplayStatus, string> = {
  idle: "Sẵn sàng",
  connecting: "Đang kết nối…",
  playing: "Đang phát",
  paused: "Tạm dừng",
  reconnecting: "Đang nối lại",
  ended: "Đã hết dữ liệu",
};

const STATUS_DOT: Record<ReplayStatus, string> = {
  idle: "bg-axis",
  connecting: "bg-warning",
  playing: "bg-good shadow-[0_0_0_3px_var(--color-good-soft)]",
  paused: "bg-ink-2",
  reconnecting: "bg-warning",
  ended: "bg-accent",
};

export function ReplayScreen() {
  const { threshold: appThreshold, openTx } = useApp();
  const r = useReplay();
  const running = r.status === "playing" || r.status === "connecting" || r.status === "reconnecting";
  const hours = r.simSeconds / 3600;
  const progress = r.dayTotal ? r.processed / r.dayTotal : 0;
  const realPerHour = 3600 / r.speed;

  return (
    <div className="grid gap-4">
      <section className="card grid grid-cols-[auto_minmax(0,1fr)] items-center gap-x-7 gap-y-4 xl:grid-cols-[auto_auto_minmax(260px,1fr)]">
        <div>
          <span className="block text-xs font-semibold uppercase tracking-wide text-muted">Đồng hồ mô phỏng · ngày 2</span>
          <span className="block text-[40px] font-bold leading-tight tabular-nums">{fmt.clock(r.simSeconds)}</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {running ? (
            <button type="button" className="btn" onClick={r.pause}>❚❚ Tạm dừng</button>
          ) : (
            <button type="button" className="btn btn-primary" onClick={r.play}>
              {r.status === "paused" ? "▶ Tiếp tục" : r.status === "ended" ? "▶ Phát lại từ đầu" : "▶ Phát"}
            </button>
          )}
          <button type="button" className="btn" onClick={r.reset} disabled={r.status === "idle"}>↺ Đặt lại</button>
        </div>
        <div className="col-span-full grid grid-cols-[auto_minmax(120px,1fr)_auto] items-center gap-x-3 gap-y-1 xl:col-span-1">
          <label htmlFor="replay-speed" className="text-[13px] font-semibold">Tốc độ</label>
          <input
            id="replay-speed"
            type="range"
            min={0}
            max={SPEEDS.length - 1}
            step={1}
            value={r.speedIndex}
            onChange={(e) => r.setSpeedIndex(Number(e.target.value))}
            aria-valuetext={`nén ${fmt.int(r.speed)} lần`}
            className="accent-accent"
          />
          <output htmlFor="replay-speed" className="min-w-14 text-right font-bold tabular-nums">{fmt.int(r.speed)}×</output>
          <small className="col-start-2 col-end-4 text-xs text-muted">
            1 giờ dữ liệu ≈ {realPerHour >= 60 ? `${fmt.num(realPerHour / 60, 0)} phút` : `${fmt.num(realPerHour, 0)} giây`} thực
          </small>
        </div>
        <p role="status" className="col-span-full flex items-center gap-2 font-semibold">
          <span aria-hidden="true" className={`size-[9px] rounded-full ${STATUS_DOT[r.status]}`} />
          {STATUS_TEXT[r.status]}
          {r.message && <span className="font-normal text-muted"> — {r.message}</span>}
        </p>
      </section>

      <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        <Tile label="Đã xử lý" value={fmt.int(r.processed)} sub={r.dayTotal ? `${fmt.pct(progress)} của ${fmt.int(r.dayTotal)} giao dịch ngày 2` : "giao dịch"} />
        <Tile label="Cảnh báo" value={fmt.int(r.alerts)} sub={r.processed ? `${fmt.pct(r.alerts / r.processed, 2)} số giao dịch` : "vượt ngưỡng"} />
        <Tile label="Cảnh báo mỗi giờ mô phỏng" value={hours > 0 ? fmt.num(r.alerts / hours, 1) : "—"} sub="trên phần luồng đã phát (20% giao dịch thật)" />
        <Tile label="Ngưỡng máy chủ đang dùng" value={fmt.tau(r.threshold ?? appThreshold?.current)} sub="đọc lại mỗi 5 giây — đổi ngưỡng giữa chừng có tác dụng ngay" />
      </div>
      <div
        role="progressbar"
        aria-label="Tiến độ phát lại ngày 2"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(progress * 100)}
        className="-mt-1 h-1.5 overflow-hidden rounded-full bg-grid"
      >
        <span className="block h-full bg-accent transition-[width] duration-300 ease-linear" style={{ width: `${progress * 100}%` }} />
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        <section className="card">
          <h2 className="card-title">Dòng giao dịch</h2>
          <p className="card-sub">Mới nhất ở trên. Dưới ngưỡng trôi qua màu xám; vượt ngưỡng chuyển đỏ rồi rơi vào hàng đợi.</p>
          <ol aria-label="Giao dịch vừa phát" className="min-h-[120px] text-[13px]">
            {r.feed.map((ev) => (
              <li
                key={ev.id}
                className={`grid grid-cols-[68px_78px_minmax(0,1fr)_64px] items-center gap-2.5 rounded-md px-2 py-1 sm:grid-cols-[68px_78px_minmax(0,1fr)_64px_150px] ${
                  ev.alert ? "feed-alert bg-critical-soft font-bold shadow-[inset_3px_0_0_var(--color-critical)]" : "animate-feed-in text-muted"
                }`}
              >
                <span className="font-mono">{ev.sim_time}</span>
                <span className="font-mono">{ev.id}</span>
                <span className="text-right tabular-nums">{fmt.num(ev.amount)} EUR</span>
                <span className="text-right tabular-nums">{fmt.score(ev.risk_score)}</span>
                <span className={`hidden sm:inline ${ev.alert ? "text-critical-text" : ""}`}>{ev.alert ? "Cảnh báo → hàng đợi" : "cho qua"}</span>
              </li>
            ))}
          </ol>
          {!r.feed.length && <p className="text-muted">Bấm Phát để bắt đầu dòng giao dịch của ngày 2.</p>}
        </section>
        <section className="card">
          <h2 className="card-title">Cảnh báo vừa vào hàng đợi</h2>
          <p className="card-sub">
            Bấm để mở chi tiết. Phát lại ghi vào cơ sở dữ liệu theo mẻ 100 dòng, nên giao dịch vừa phát có thể cần vài giây mới mở được.
          </p>
          <ul className="grid max-h-[560px] gap-1 overflow-y-auto">
            {r.alertList.map((ev) => (
              <li key={ev.id}>
                <button
                  type="button"
                  onClick={(e) => openTx(ev.id, e.currentTarget)}
                  className="grid w-full animate-feed-in cursor-pointer grid-cols-[92px_80px_minmax(0,1fr)_70px] items-center gap-2.5 rounded-md border border-line bg-surface px-2.5 py-1.5 text-left text-[13px] hover:border-line-strong hover:bg-surface-2"
                >
                  <RiskScore score={ev.risk_score} />
                  <span className="font-mono">{ev.id}</span>
                  <span className="text-right tabular-nums">{fmt.num(ev.amount)} EUR</span>
                  <span className="font-mono text-muted">{ev.sim_time}</span>
                </button>
              </li>
            ))}
          </ul>
          {!r.alertList.length && <p className="text-muted">Chưa có cảnh báo.</p>}
        </section>
      </div>
    </div>
  );
}
