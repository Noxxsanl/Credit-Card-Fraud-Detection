"use client";

/*
 * Chế độ phát lại — UI-05, API-15 (Server-Sent Events).
 *
 * Để ở layout gốc chứ không trong trang /replay: chuyển sang Hàng đợi giữa chừng thì luồng vẫn
 * chạy và cảnh báo tiếp tục rơi vào hàng đợi. Máy chủ không giữ trạng thái nào ngoài con trỏ
 * của kết nối: tạm dừng là đóng EventSource, tiếp tục là mở lại với `start` bằng giây mô phỏng
 * cuối cùng đã nhận. Giây đó được phát lại, nên trình duyệt tự đếm và bỏ trùng theo mã.
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { apiBase, query } from "@/lib/api";
import type { ReplayEvent } from "@/lib/types";

import { useApp } from "./AppContext";

export const SPEEDS = [1, 10, 30, 60, 120, 300, 600, 1800, 3600] as const;
const FEED_SIZE = 30;
const ALERT_LIST_SIZE = 60;
const DAY_SECONDS = 86_400;

export type ReplayStatus = "idle" | "connecting" | "playing" | "paused" | "reconnecting" | "ended";

export interface FeedItem extends ReplayEvent {
  alert: boolean;
}

interface ReplayValue {
  status: ReplayStatus;
  speedIndex: number;
  speed: number;
  simSeconds: number;
  processed: number;
  alerts: number;
  dayTotal: number | null;
  threshold: number | null;
  feed: FeedItem[];
  alertList: FeedItem[];
  message: string;
  play: () => void;
  pause: () => void;
  reset: () => void;
  setSpeedIndex: (index: number) => void;
}

const ReplayContext = createContext<ReplayValue | null>(null);

export function useReplay(): ReplayValue {
  const value = useContext(ReplayContext);
  if (!value) throw new Error("useReplay phải nằm trong <ReplayProvider>");
  return value;
}

export function ReplayProvider({ children }: { children: ReactNode }) {
  const { bumpQueue } = useApp();
  const [status, setStatus] = useState<ReplayStatus>("idle");
  const [speedIndex, setSpeedIndexState] = useState(3); // 60 lần — 1 giờ dữ liệu ≈ 1 phút thực (07 §7)
  const [simSeconds, setSimSeconds] = useState(0);
  const [processed, setProcessed] = useState(0);
  const [alerts, setAlerts] = useState(0);
  const [dayTotal, setDayTotal] = useState<number | null>(null);
  const [threshold, setThreshold] = useState<number | null>(null);
  const [feed, setFeed] = useState<FeedItem[]>([]);
  const [alertList, setAlertList] = useState<FeedItem[]>([]);
  const [message, setMessage] = useState("");
  /** Lịch nối lại sau khi mất kết nối — effect bên dưới thực hiện sau 3 giây */
  const [retry, setRetry] = useState<{ start: number; speed: number } | null>(null);

  // Ngoài trạng thái React: kết nối, tập mã đã đếm, bộ đệm sự kiện, mốc đồng hồ
  const source = useRef<EventSource | null>(null);
  const seen = useRef(new Set<string>());
  const buffer = useRef<FeedItem[]>([]);
  const frame = useRef(0);
  const lastEventSeconds = useRef(0);
  const clock = useRef<{ base: number; sim: number; wall: number; speed: number } | null>(null);
  const statusRef = useRef<ReplayStatus>("idle");
  const processedRef = useRef(0);
  const dayTotalRef = useRef<number | null>(null);

  const setStatusBoth = useCallback((next: ReplayStatus) => {
    statusRef.current = next;
    setStatus(next);
  }, []);

  const disconnect = useCallback(() => {
    setRetry(null);
    source.current?.close();
    source.current = null;
  }, []);

  /** Gộp sự kiện theo khung hình: ở tốc độ 3.600 lần có hơn nghìn giao dịch mỗi giây. */
  const flush = useCallback(() => {
    frame.current = 0;
    const batch = buffer.current;
    buffer.current = [];
    const fresh: FeedItem[] = [];
    for (const ev of batch) {
      lastEventSeconds.current = Math.max(lastEventSeconds.current, ev.sim_seconds);
      if (seen.current.has(ev.id)) continue;
      seen.current.add(ev.id);
      fresh.push(ev);
    }
    if (!fresh.length) return;
    const newAlerts = fresh.filter((ev) => ev.alert);
    processedRef.current += fresh.length;
    setProcessed(processedRef.current);
    setAlerts((n) => n + newAlerts.length);
    setFeed((old) => [...fresh.slice(-FEED_SIZE).reverse(), ...old].slice(0, FEED_SIZE));
    if (newAlerts.length) setAlertList((old) => [...newAlerts.reverse(), ...old].slice(0, ALERT_LIST_SIZE));
  }, []);

  const connect = useCallback((start: number, speed: number) => {
    disconnect();
    setStatusBoth("connecting");
    setMessage("");
    const es = new EventSource(`${apiBase()}/replay/stream${query({ speed, start: Math.min(start, DAY_SECONDS - 1) })}`);
    source.current = es;
    const parse = (e: Event) => JSON.parse((e as MessageEvent).data);

    es.addEventListener("start", (e) => {
      const d = parse(e);
      setStatusBoth("playing");
      setThreshold(d.threshold);
      if (dayTotalRef.current === null) {
        dayTotalRef.current = d.start === 0 ? d.total : processedRef.current + d.total;
        setDayTotal(dayTotalRef.current);
      }
      clock.current = { base: d.start, sim: d.start, wall: performance.now(), speed: d.speed };
    });
    const onTransaction = (alert: boolean) => (e: Event) => {
      buffer.current.push({ ...(parse(e) as ReplayEvent), alert });
      if (!frame.current) frame.current = requestAnimationFrame(flush);
    };
    es.addEventListener("transaction", onTransaction(false));
    es.addEventListener("alert", onTransaction(true));
    es.addEventListener("stats", (e) => {
      const d = parse(e);
      setThreshold(d.threshold);
      // Đồng hồ máy chủ là chuẩn; giữa hai nhịp stats, trình duyệt tự nội suy theo tốc độ
      if (clock.current) {
        clock.current = { ...clock.current, sim: clock.current.base + (d.elapsed_sim_seconds ?? 0), wall: performance.now() };
      }
    });
    es.addEventListener("end", () => {
      flush();
      disconnect();
      setStatusBoth("ended");
      setSimSeconds(DAY_SECONDS);
      setMessage("Đã phát hết ngày 2 của tập kiểm thử.");
      bumpQueue();
    });
    es.onerror = () => {
      if (source.current !== es) return;
      // Tự nối lại với start = giây cuối đã nhận; để mặc định, EventSource phát lại từ start cũ
      disconnect();
      setStatusBoth("reconnecting");
      setMessage("Mất kết nối với API — thử nối lại sau 3 giây…");
      setRetry({ start: lastEventSeconds.current, speed });
    };
  }, [disconnect, flush, bumpQueue, setStatusBoth]);

  useEffect(() => {
    if (!retry) return;
    const timer = setTimeout(() => connect(retry.start, retry.speed), 3000);
    return () => clearTimeout(timer);
  }, [retry, connect]);

  // Đồng hồ mô phỏng: 10 lần mỗi giây là đủ mượt cho HH:MM:SS
  useEffect(() => {
    if (status !== "playing") return;
    const timer = setInterval(() => {
      const c = clock.current;
      if (c) setSimSeconds(Math.min(c.sim + ((performance.now() - c.wall) / 1000) * c.speed, DAY_SECONDS));
    }, 100);
    return () => clearInterval(timer);
  }, [status]);

  useEffect(() => () => disconnect(), [disconnect]);

  const reset = useCallback(() => {
    disconnect();
    seen.current = new Set();
    buffer.current = [];
    lastEventSeconds.current = 0;
    clock.current = null;
    processedRef.current = 0;
    dayTotalRef.current = null;
    setStatusBoth("idle");
    setSimSeconds(0);
    setProcessed(0);
    setAlerts(0);
    setDayTotal(null);
    setThreshold(null);
    setFeed([]);
    setAlertList([]);
    setMessage("");
  }, [disconnect, setStatusBoth]);

  const value = useMemo<ReplayValue>(() => ({
    status, speedIndex, speed: SPEEDS[speedIndex], simSeconds, processed, alerts, dayTotal, threshold, feed, alertList, message,
    play: () => {
      if (statusRef.current === "playing" || statusRef.current === "connecting") return;
      if (statusRef.current === "ended") reset();
      connect(statusRef.current === "ended" ? 0 : lastEventSeconds.current, SPEEDS[speedIndex]);
    },
    /** Tạm dừng là đóng kết nối (docs/05 §7, API-15) */
    pause: () => {
      disconnect();
      if (statusRef.current !== "idle" && statusRef.current !== "ended") setStatusBoth("paused");
    },
    reset,
    setSpeedIndex: (index) => {
      setSpeedIndexState(index);
      if (["playing", "connecting", "reconnecting"].includes(statusRef.current)) {
        connect(lastEventSeconds.current, SPEEDS[index]);
      }
    },
  }), [status, speedIndex, simSeconds, processed, alerts, dayTotal, threshold, feed, alertList, message, connect, disconnect, reset, setStatusBoth]);

  return <ReplayContext.Provider value={value}>{children}</ReplayContext.Provider>;
}
