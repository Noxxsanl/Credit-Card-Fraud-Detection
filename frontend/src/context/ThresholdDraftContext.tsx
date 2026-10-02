"use client";

/*
 * Bản nháp của màn hình UI-03: ngưỡng đang xem, tham số chi phí, ràng buộc, và kết quả
 * POST /threshold/optimize. Để ở layout gốc để sống qua các lần chuyển trang — sang Hàng đợi
 * rồi quay lại vẫn thấy đúng vị trí thanh trượt. Chỉ `PUT /threshold` ("Áp dụng") mới đổi
 * ngưỡng của toàn ứng dụng.
 */

import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { api, isAbort } from "@/lib/api";
import type { ConstraintType, OptimizeResponse } from "@/lib/types";

import { useApp } from "./AppContext";

interface Costs {
  fn: number;
  fp: number;
  valid: boolean;
}

interface Constraint {
  valid: boolean;
  body: { type: Exclude<ConstraintType, "none">; value: number } | null;
}

interface DraftValue {
  initialized: boolean;
  tau: number;
  setTau: (value: number) => void;
  costFnInput: string;
  setCostFnInput: (value: string) => void;
  costFpInput: string;
  setCostFpInput: (value: string) => void;
  constraintType: ConstraintType;
  setConstraintType: (value: ConstraintType) => void;
  constraintValueInput: string;
  setConstraintValueInput: (value: string) => void;
  costs: Costs;
  constraint: Constraint;
  opt: OptimizeResponse | null;
  optLoading: boolean;
  optError: string | null;
}

const DraftContext = createContext<DraftValue | null>(null);

export function useThresholdDraft(): DraftValue {
  const value = useContext(DraftContext);
  if (!value) throw new Error("useThresholdDraft phải nằm trong <ThresholdDraftProvider>");
  return value;
}

export function ThresholdDraftProvider({ children }: { children: ReactNode }) {
  const { threshold, scores } = useApp();
  const [initialized, setInitialized] = useState(false);
  const [tau, setTauState] = useState(0.5);
  const [costFnInput, setCostFnInput] = useState("");
  const [costFpInput, setCostFpInput] = useState("");
  const [constraintType, setConstraintType] = useState<ConstraintType>("none");
  const [constraintValueInput, setConstraintValueInput] = useState("");
  const [opt, setOpt] = useState<OptimizeResponse | null>(null);
  const [optLoading, setOptLoading] = useState(false);
  const [optError, setOptError] = useState<string | null>(null);

  const requestKey = useRef<string | null>(null);
  const controller = useRef<AbortController | null>(null);

  // Khởi tạo một lần, ngay khi có ngưỡng hiện hành và điểm của tập kiểm thử. Đặt state trong lúc
  // render (không trong effect) là cách React khuyên cho state dẫn xuất từ dữ liệu vừa đến.
  if (!initialized && threshold && scores) {
    setInitialized(true);
    setTauState(threshold.current);
    setCostFnInput(String(threshold.cost_fn));
    setCostFpInput(String(threshold.cost_fp));
  }

  const costs = useMemo<Costs>(() => {
    const fn = Number(costFnInput);
    const fp = Number(costFpInput);
    const valid = costFnInput !== "" && costFpInput !== "" && Number.isFinite(fn) && Number.isFinite(fp) && fn > 0 && fp >= 0;
    return { fn, fp, valid };
  }, [costFnInput, costFpInput]);

  const constraint = useMemo<Constraint>(() => {
    if (constraintType === "none") return { valid: true, body: null };
    const value = Number(constraintValueInput);
    if (constraintValueInput === "" || !Number.isFinite(value)) return { valid: false, body: null };
    if (constraintType === "min_recall") {
      const recall = value / 100; // nhập theo phần trăm
      return { valid: recall > 0 && recall <= 1, body: { type: "min_recall", value: recall } };
    }
    return { valid: value > 0, body: { type: "max_alerts_per_day", value } };
  }, [constraintType, constraintValueInput]);

  // POST /threshold/optimize, debounce 300 ms kể từ lần gõ cuối (07 §5)
  useEffect(() => {
    if (!initialized || !costs.valid || !constraint.valid) return;
    const body: Record<string, unknown> = { cost_fn: costs.fn, cost_fp: costs.fp };
    if (constraint.body) body.constraint = constraint.body;
    const key = JSON.stringify(body);
    if (key === requestKey.current) return; // đang chạy hoặc đã có đúng kết quả này
    const timer = setTimeout(async () => {
      requestKey.current = key;
      controller.current?.abort();
      const current = new AbortController();
      controller.current = current;
      setOptLoading(true);
      setOptError(null);
      try {
        setOpt(await api<OptimizeResponse>("/threshold/optimize", { method: "POST", json: body, signal: current.signal }));
      } catch (e) {
        if (!isAbort(e)) {
          setOptError((e as Error).message);
          requestKey.current = null;
        }
      } finally {
        if (controller.current === current) setOptLoading(false);
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [initialized, costs, constraint]);

  const value = useMemo<DraftValue>(() => ({
    initialized,
    tau,
    setTau: (v) => {
      if (Number.isFinite(v) && v > 0 && v < 1) setTauState(v);
    },
    costFnInput, setCostFnInput, costFpInput, setCostFpInput,
    constraintType, setConstraintType, constraintValueInput, setConstraintValueInput,
    costs, constraint, opt, optLoading, optError,
  }), [initialized, tau, costFnInput, costFpInput, constraintType, constraintValueInput, costs, constraint, opt, optLoading, optError]);

  return <DraftContext.Provider value={value}>{children}</DraftContext.Provider>;
}
