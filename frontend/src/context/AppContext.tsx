"use client";

/*
 * Trạng thái toàn cục (docs/07 §9): ngưỡng hiện hành và tham số chi phí, điểm của tập kiểm thử,
 * tình trạng API, thông báo nổi, và các yêu cầu mở ngăn kéo chi tiết / thư viện mẫu.
 *
 * Nằm trong layout gốc nên sống qua mọi lần chuyển trang. Điểm của tập kiểm thử (56.746 phần tử,
 * đã sắp) nằm trong state như mọi giá trị khác: React so sánh theo tham chiếu nên kích thước mảng
 * không ảnh hưởng gì, và mảng không bao giờ bị sửa sau khi nạp.
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { api, ApiError } from "@/lib/api";
import { loadTestScores, metrics, type TestScores, type ThresholdMetrics } from "@/lib/scores";
import type { Dataset, ReviewDecision, ThresholdState } from "@/lib/types";

export interface HealthState {
  state: "loading" | "ok" | "error";
  code?: string;
  message?: string;
  reason?: string;
}

export interface Toast {
  id: number;
  message: string;
  kind: "info" | "success" | "error";
}

interface AppValue {
  health: HealthState;
  modelVersion: string | null;
  modelChanged: boolean;
  trainedAt: string | null;
  threshold: ThresholdState | null;
  thresholdError: string | null;
  scores: TestScores | null;
  scoresError: string | null;
  /** Chỉ số của ngưỡng hiện hành trên tập kiểm thử — dòng đầu UI-01, ma trận ở UI-04 */
  summary: ThresholdMetrics | null;
  toasts: Toast[];
  /** Tăng mỗi khi dữ liệu hàng đợi đổi ở nơi khác (tải CSV, nạp mẫu, hết phát lại) */
  queueVersion: number;
  /** Kết luận ghi trong phiên này — hàng đợi hiện ngay, không chờ tải lại */
  reviewed: Record<string, ReviewDecision>;
  drawerRequest: { id: string; seq: number } | null;
  samplesRequest: number;

  checkHealth: () => void;
  loadScores: () => void;
  setThreshold: (state: ThresholdState) => void;
  toast: (message: string, kind?: Toast["kind"]) => void;
  openTx: (id: string, returnFocus?: HTMLElement | null) => void;
  takeReturnFocus: () => HTMLElement | null;
  openSamples: () => void;
  bumpQueue: () => void;
  noteReview: (id: string, decision: ReviewDecision) => void;
}

const AppContext = createContext<AppValue | null>(null);

export function useApp(): AppValue {
  const value = useContext(AppContext);
  if (!value) throw new Error("useApp phải nằm trong <AppProvider>");
  return value;
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [health, setHealth] = useState<HealthState>({ state: "loading" });
  const [modelVersion, setModelVersion] = useState<string | null>(null);
  const [modelChanged, setModelChanged] = useState(false);
  const [trainedAt, setTrainedAt] = useState<string | null>(null);
  const [threshold, setThreshold] = useState<ThresholdState | null>(null);
  const [thresholdError, setThresholdError] = useState<string | null>(null);
  const [scores, setScores] = useState<TestScores | null>(null);
  const [scoresError, setScoresError] = useState<string | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [queueVersion, setQueueVersion] = useState(0);
  const [reviewed, setReviewed] = useState<Record<string, ReviewDecision>>({});
  const [drawerRequest, setDrawerRequest] = useState<AppValue["drawerRequest"]>(null);
  const [samplesRequest, setSamplesRequest] = useState(0);

  const versionRef = useRef<string | null>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const loaded = useRef({ threshold: false, scores: false, info: false });
  const seq = useRef(0);

  const noteModel = useCallback((version: string | undefined | null) => {
    if (!version) return;
    if (versionRef.current && versionRef.current !== version) setModelChanged(true);
    versionRef.current = version;
    setModelVersion(version);
  }, []);

  const loadThreshold = useCallback(() => {
    api<ThresholdState>("/threshold").then(
      (state) => {
        setThreshold(state);
        setThresholdError(null);
        loaded.current.threshold = true;
        noteModel(state.model_version);
      },
      (e: Error) => setThresholdError(e.message),
    );
  }, [noteModel]);

  const loadScores = useCallback(() => {
    Promise.all([
      api<{ test_scores: { y_true: number[]; y_score: number[] } }>("/metrics?section=test_scores"),
      api<{ dataset: Dataset }>("/metrics?section=dataset"),
    ]).then(
      ([s, d]) => {
        setScores(loadTestScores(s.test_scores.y_true, s.test_scores.y_score, d.dataset));
        setScoresError(null);
        loaded.current.scores = true;
      },
      (e: Error) => setScoresError(e.message),
    );
  }, []);

  const loadModelInfo = useCallback(() => {
    api<{ model_version: string; trained_at: string }>("/metrics?section=trained_at").then(
      (info) => {
        setTrainedAt(info.trained_at);
        loaded.current.info = true;
        noteModel(info.model_version);
      },
      () => undefined, // chân thanh bên để trống; /health báo lỗi riêng
    );
  }, [noteModel]);

  const checkHealth = useCallback(() => {
    api<{ model_version: string }>("/health").then(
      (h) => {
        setHealth({ state: "ok" });
        noteModel(h.model_version);
        if (!loaded.current.threshold) loadThreshold();
        if (!loaded.current.scores) loadScores();
        if (!loaded.current.info) loadModelInfo();
      },
      (e: ApiError) => {
        const details = (e.details ?? {}) as { reason?: string; model_version?: string };
        setHealth({ state: "error", code: e.code, message: e.message, reason: details.reason });
        noteModel(details.model_version);
      },
    );
  }, [noteModel, loadThreshold, loadScores, loadModelInfo]);

  // Thăm /health lúc mở trang rồi mỗi 15 giây (API và CSDL có thể dừng giữa chừng)
  useEffect(() => {
    checkHealth();
    const timer = setInterval(checkHealth, 15_000);
    return () => clearInterval(timer);
  }, [checkHealth]);

  const toast = useCallback((message: string, kind: Toast["kind"] = "info") => {
    const id = ++seq.current;
    setToasts((list) => [...list, { id, message, kind }]);
    setTimeout(() => setToasts((list) => list.filter((t) => t.id !== id)), kind === "error" ? 8000 : 4500);
  }, []);

  const openTx = useCallback((id: string, returnFocus?: HTMLElement | null) => {
    returnFocusRef.current = returnFocus ?? (document.activeElement as HTMLElement | null);
    setDrawerRequest({ id, seq: ++seq.current });
  }, []);

  const takeReturnFocus = useCallback(() => {
    const element = returnFocusRef.current;
    returnFocusRef.current = null;
    return element;
  }, []);

  const openSamples = useCallback(() => setSamplesRequest((n) => n + 1), []);
  const bumpQueue = useCallback(() => setQueueVersion((n) => n + 1), []);
  const noteReview = useCallback((id: string, decision: ReviewDecision) => setReviewed((map) => ({ ...map, [id]: decision })), []);

  const summary = useMemo(
    () => (scores && threshold ? metrics(scores, threshold.current, threshold.cost_fn, threshold.cost_fp) : null),
    [scores, threshold],
  );

  const value = useMemo<AppValue>(() => ({
    health, modelVersion, modelChanged, trainedAt, threshold, thresholdError, scores, scoresError,
    summary, toasts, queueVersion, reviewed, drawerRequest, samplesRequest,
    checkHealth, loadScores, setThreshold, toast, openTx, takeReturnFocus, openSamples, bumpQueue, noteReview,
  }), [health, modelVersion, modelChanged, trainedAt, threshold, thresholdError, scores, scoresError, summary, toasts,
    queueVersion, reviewed, drawerRequest, samplesRequest, checkHealth, loadScores, toast, openTx, takeReturnFocus,
    openSamples, bumpQueue, noteReview]);

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}
