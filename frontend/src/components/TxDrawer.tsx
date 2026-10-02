"use client";

/*
 * UI-02 — Chi tiết giao dịch: ngăn kéo bên phải (docs/07 §4).
 *
 * <dialog> mở bằng showModal(): trình duyệt tự bẫy trọng tâm và bắt Esc; khi đóng, trọng tâm
 * trả về đúng dòng đã mở nó. Điểm rủi ro hiện ngay (GET /transactions/{id}); SHAP (POST /explain,
 * 60–100 ms) tải song song và có khung xương riêng. Nhãn thật chỉ lộ SAU khi người dùng đã kết
 * luận — để tự đối chiếu mà không bị mồi (FR-44).
 */

import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";

import { useApp } from "@/context/AppContext";
import { api, ApiError } from "@/lib/api";
import { fmt, logit, sigmoid } from "@/lib/format";
import { DECISION, REVIEW, SOURCE } from "@/lib/labels";
import type { Explanation, ReviewDecision, ReviewResponse, TransactionDetail } from "@/lib/types";

import { PcaNote, RiskScore, Skeleton } from "./ui";

interface WaterfallRow {
  kind: "base" | "group" | "pos" | "neg" | "rest" | "total";
  key: string;
  label: string;
  feature?: string;
  value?: number;
  shap?: number;
  from: number;
  to: number;
  left: number;
  width: number;
}

/** Thác nước ở thang log-odds: base_value + Σ SHAP = margin (docs/05 §7). */
function buildWaterfall(e: Explanation, tau: number) {
  const rows: WaterfallRow[] = [];
  let cum = e.base_value;
  const push = (row: Omit<WaterfallRow, "left" | "width">) => rows.push({ ...row, left: 0, width: 0 });
  const step = (kind: "pos" | "neg" | "rest", shap: number, label: string, feature?: string, value?: number) => {
    const from = cum;
    cum += shap;
    push({ kind, key: `${kind}-${feature ?? label}`, label, feature, value, shap, from, to: cum });
  };
  push({ kind: "base", key: "base", label: "Giá trị cơ sở E[f(x)]", from: cum, to: cum });
  if (e.top_positive.length) push({ kind: "group", key: "g-pos", label: "Yếu tố đẩy điểm rủi ro lên", from: 0, to: 0 });
  e.top_positive.forEach((c) => step("pos", c.shap, c.feature, c.feature, c.value));
  if (e.top_negative.length) push({ kind: "group", key: "g-neg", label: "Yếu tố kéo điểm rủi ro xuống", from: 0, to: 0 });
  e.top_negative.forEach((c) => step("neg", c.shap, c.feature, c.feature, c.value));
  const others = e.contributions.length - e.top_positive.length - e.top_negative.length;
  if (others > 0) step("rest", e.remaining_shap, `${others} đặc trưng còn lại`);
  push({ kind: "total", key: "total", label: "Kết quả f(x)", from: e.margin, to: e.margin });

  const tauLogit = tau > 0 && tau < 1 ? logit(tau) : null;
  const ends = rows.filter((r) => r.kind !== "group").flatMap((r) => [r.from, r.to]);
  if (tauLogit !== null) ends.push(tauLogit);
  const pad = (Math.max(...ends) - Math.min(...ends)) * 0.06 || 1;
  const lo = Math.min(...ends) - pad;
  const hi = Math.max(...ends) + pad;
  const pos = (v: number) => ((v - lo) / (hi - lo)) * 100;
  for (const r of rows) {
    if (r.kind === "group") continue;
    r.left = pos(Math.min(r.from, r.to));
    r.width = r.kind === "base" || r.kind === "total" ? 0 : Math.max(Math.abs(pos(r.to) - pos(r.from)), 0.6);
  }
  return { rows, tauPos: tauLogit === null ? null : pos(tauLogit), tauLogit, probability: sigmoid(e.margin) };
}

const BAR_COLOR = { pos: "bg-critical", neg: "bg-series-1", rest: "bg-axis" } as const;
const VERDICT_COLOR = { block: "text-critical-text", review: "text-serious-text", allow: "text-good-text" } as const;

export function TxDrawer() {
  const { drawerRequest, takeReturnFocus, noteReview, toast, threshold } = useApp();
  const dialog = useRef<HTMLDialogElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const returnFocus = useRef<HTMLElement | null>(null);
  const [handledSeq, setHandledSeq] = useState(0);
  const [attempt, setAttempt] = useState(0);
  const [id, setId] = useState<string | null>(null);
  const [tx, setTx] = useState<TransactionDetail | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [exp, setExp] = useState<Explanation | null>(null);
  const [expError, setExpError] = useState<string | null>(null);
  const [saved, setSaved] = useState<ReviewResponse | null>(null);
  const [saving, setSaving] = useState(false);
  const [note, setNote] = useState("");

  const clear = () => {
    setTx(null);
    setError(null);
    setExp(null);
    setExpError(null);
    setSaved(null);
    setNote("");
  };

  // Yêu cầu mở mới (từ UI-01 hoặc UI-05): đặt lại ngăn kéo ngay trong lúc render
  if (drawerRequest && drawerRequest.seq !== handledSeq) {
    setHandledSeq(drawerRequest.seq);
    setId(drawerRequest.id);
    clear();
  }

  // Phần việc với DOM: mở hộp thoại, chuyển trọng tâm, nhớ phần tử để trả trọng tâm khi đóng
  useEffect(() => {
    if (!drawerRequest) return;
    returnFocus.current = takeReturnFocus();
    if (dialog.current && !dialog.current.open) dialog.current.showModal();
    closeButton.current?.focus();
  }, [drawerRequest, takeReturnFocus]);

  // Hai lời gọi song song: điểm hiện ngay, SHAP theo sau
  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    api<TransactionDetail>(`/transactions/${encodeURIComponent(id)}`).then(
      (detail) => {
        if (cancelled) return;
        setTx(detail);
        setNote(detail.review?.note ?? "");
      },
      (e: ApiError) => !cancelled && setError(e),
    );
    api<Explanation>("/explain", { method: "POST", json: { transaction_id: id } }).then(
      (result) => !cancelled && setExp(result),
      (e: Error) => !cancelled && setExpError(e.message),
    );
    return () => {
      cancelled = true;
    };
  }, [id, handledSeq, attempt]);

  const tauForChart = tx?.threshold ?? threshold?.current ?? 0.5;
  const waterfall = useMemo(() => (exp ? buildWaterfall(exp, tauForChart) : null), [exp, tauForChart]);

  const decided: ReviewDecision | null = saved?.decision ?? tx?.review?.decision ?? null;
  const trueLabel = decided ? (saved ? saved.true_label : tx?.true_label ?? null) : null;
  const matches = decided && trueLabel !== null ? (decided === "confirmed_fraud") === (trueLabel === 1) : null;

  const close = () => dialog.current?.close();

  const onClose = () => {
    setId(null);
    const target = returnFocus.current;
    returnFocus.current = null;
    if (target && document.contains(target)) target.focus();
  };

  async function decide(decision: ReviewDecision) {
    if (!tx || saving) return;
    setSaving(true);
    try {
      const r = await api<ReviewResponse>("/reviews", {
        method: "POST",
        json: { transaction_id: tx.id, decision, note: note.trim() || null },
      });
      setSaved(r);
      noteReview(r.transaction_id, r.decision);
      toast(`${r.transaction_id}: đã ghi "${REVIEW[r.decision]}"`, "success");
    } catch (e) {
      toast(`Không ghi được kết luận: ${(e as Error).message}`, "error");
    } finally {
      setSaving(false);
    }
  }

  /** F xác nhận gian lận, A báo động sai (07 §4) — trừ khi đang gõ ghi chú hay giữ Ctrl/Alt. */
  function onKeyDown(event: KeyboardEvent<HTMLDialogElement>) {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if ((event.target as HTMLElement).closest("textarea, input, select")) return;
    const key = event.key.toLowerCase();
    if (key === "f") {
      event.preventDefault();
      decide("confirmed_fraud");
    } else if (key === "a") {
      event.preventDefault();
      decide("false_alarm");
    }
  }

  return (
    <dialog ref={dialog} className="drawer" aria-labelledby="drawer-title" onClose={onClose} onKeyDown={onKeyDown}>
      <header className="flex items-start justify-between gap-3 border-b border-grid px-5.5 pb-3.5 pt-4.5">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-muted">Chi tiết giao dịch</p>
          <h2 id="drawer-title" className="font-mono text-xl font-bold">{id ?? ""}</h2>
        </div>
        <button ref={closeButton} type="button" className="btn btn-ghost" onClick={close}>
          Đóng <kbd>Esc</kbd>
        </button>
      </header>

      <div className="flex-1 overflow-y-auto px-5.5 py-4.5">
        {error && (
          <div role="alert" className="rounded-[10px] border border-dashed border-line-strong p-6 text-center">
            <p className="font-bold">{error.status === 404 ? "Chưa tìm thấy giao dịch này." : "Không tải được giao dịch."}</p>
            {error.status === 404 && id?.startsWith("RP-") && (
              <p className="text-muted">Phát lại ghi vào cơ sở dữ liệu theo mẻ 100 dòng — thử lại sau vài giây.</p>
            )}
            {error.status !== 404 && <p className="text-muted">{error.message}</p>}
            <button type="button" className="btn mt-2" onClick={() => { clear(); setAttempt((n) => n + 1); }}>
              Thử lại
            </button>
          </div>
        )}

        {!tx && !error && (
          <div aria-hidden="true" className="mb-4">
            <Skeleton className="mb-2.5 h-11 w-[45%]" />
            <Skeleton />
          </div>
        )}

        {tx && (
          <>
            <div className="mb-4">
              <div className="flex flex-wrap items-center gap-5">
                <RiskScore score={tx.risk_score} large />
                <div>
                  <p>
                    {tx.decision === "allow" ? "Dưới" : "Vượt"} ngưỡng <span className="font-mono">{fmt.tau(tx.threshold)}</span> →{" "}
                    <strong className={`tracking-wide ${VERDICT_COLOR[tx.decision]}`}>{DECISION[tx.decision].toUpperCase()}</strong>
                  </p>
                  <p className="text-[12.5px] text-muted">
                    Điểm chính xác <span className="font-mono">{fmt.raw(tx.risk_score)}</span>
                  </p>
                </div>
              </div>
              <div role="img" aria-label={`Điểm rủi ro ${fmt.score(tx.risk_score)}, ngưỡng ${fmt.tau(tx.threshold)}`} className="relative mt-3.5 h-2.5 rounded-full bg-grid">
                <span
                  className={`absolute inset-y-0 left-0 rounded-full ${tx.risk_score >= 0.9 ? "bg-critical" : tx.risk_score >= 0.6 ? "bg-serious" : "bg-warning"}`}
                  style={{ width: `${tx.risk_score * 100}%` }}
                />
                <span className="absolute -inset-y-1 w-0 border-l-2 border-ink" style={{ left: `${tx.threshold * 100}%` }} />
              </div>
              <div aria-hidden="true" className="mt-1 flex justify-between text-[11.5px] text-muted">
                <span>0%</span>
                <span>ngưỡng ▲ tại {fmt.pct(tx.threshold, 2)}</span>
                <span>100%</span>
              </div>
            </div>

            {!tx.model_version_current && (
              <p className="mb-3.5 rounded-md border border-[#f1d58b] bg-warning-soft px-3.5 py-3 text-[#4d3500]">
                Điểm do mô hình <span className="font-mono">{tx.model_version}</span> chấm, khác mô hình đang phục vụ — không so trực tiếp được với ngưỡng hiện hành.
              </p>
            )}

            <dl className="mb-4.5 grid grid-cols-1 gap-x-5 gap-y-3 border-y border-grid py-3.5 sm:grid-cols-2">
              <div>
                <dt className="text-xs text-muted">Số tiền</dt>
                <dd className="font-semibold">
                  {fmt.eur(tx.amount)}
                  <small className="block font-normal text-muted">cao hơn {fmt.pct(tx.amount_percentile, 0)} giao dịch của tập kiểm thử</small>
                </dd>
              </div>
              <div>
                <dt className="text-xs text-muted">Thời điểm</dt>
                <dd className="font-semibold">{fmt.timeOffset(tx.features.Time)}</dd>
              </div>
              <div>
                <dt className="text-xs text-muted">Nguồn</dt>
                <dd className="font-semibold">{SOURCE[tx.source] + (tx.batch_id ? ` · ${tx.batch_id}` : "")}</dd>
              </div>
              <div>
                <dt className="text-xs text-muted">Mô hình chấm</dt>
                <dd className="font-mono font-semibold">{tx.model_version}</dd>
              </div>
            </dl>
          </>
        )}

        {!error && (
          <section aria-labelledby="shap-title">
            <h3 id="shap-title" className="mb-2 text-[15px] font-bold">Vì sao mô hình cho điểm này</h3>
            {!exp && !expError && (
              <div aria-hidden="true">
                {Array.from({ length: 8 }, (_, i) => <Skeleton key={i} className="my-2 h-4" />)}
              </div>
            )}
            {expError && <p role="alert" className="text-[12.5px] font-semibold text-critical-text">Không tính được SHAP: {expError}</p>}
            {exp && waterfall && (
              <>
                <table className="w-full border-collapse text-[12.5px] tabular-nums">
                  <caption className="sr-only">Đóng góp SHAP của từng đặc trưng, thang log-odds</caption>
                  <thead>
                    <tr className="text-[11.5px] font-semibold text-muted">
                      <th scope="col" className="px-1.5 pb-1.5 text-left font-semibold">Đặc trưng</th>
                      <th scope="col" className="px-1.5 pb-1.5 text-right font-semibold">Giá trị</th>
                      <th scope="col" className="px-1.5 pb-1.5 text-left font-semibold">
                        <span className="flex justify-between gap-2">
                          <span>thang log-odds</span>
                          <span aria-hidden="true" className="text-ink">│ ngưỡng</span>
                        </span>
                      </th>
                      <th scope="col" className="px-1.5 pb-1.5 text-right font-semibold">SHAP</th>
                    </tr>
                  </thead>
                  <tbody>
                    {waterfall.rows.map((row) =>
                      row.kind === "group" ? (
                        <tr key={row.key}>
                          <th scope="rowgroup" colSpan={4} className="px-1.5 pt-2.5 text-left text-xs font-bold text-ink-2">
                            {row.label}
                          </th>
                        </tr>
                      ) : (
                        <tr key={row.key} className={row.kind === "total" ? "border-t border-grid font-bold" : ""}>
                          <th scope="row" className={`whitespace-nowrap px-1.5 py-0.5 text-left ${row.feature ? "font-mono text-xs font-normal" : "font-semibold"}`}>
                            {row.feature ? fmt.feature(row.feature) : row.label}
                          </th>
                          <td className="whitespace-nowrap px-1.5 py-0.5 text-right font-mono">
                            {row.feature && row.value !== undefined ? fmt.featureValue(row.feature, row.value) : ""}
                          </td>
                          <td aria-hidden="true" className="relative h-6 w-[46%] min-w-[150px] px-1.5">
                            {waterfall.tauPos !== null && (
                              <span className="absolute inset-y-0 w-0 border-l-[1.5px] border-ink-2" style={{ left: `${waterfall.tauPos}%` }} />
                            )}
                            {row.kind === "base" || row.kind === "total" ? (
                              <span className="absolute top-1.5 -ml-[5.5px] size-[11px] rotate-45 rounded-[2px] bg-ink" style={{ left: `${row.left}%` }} />
                            ) : (
                              <span className={`absolute top-1.5 h-3 rounded-[3px] ${BAR_COLOR[row.kind]}`} style={{ left: `${row.left}%`, width: `${row.width}%` }} />
                            )}
                          </td>
                          <td className="whitespace-nowrap px-1.5 py-0.5 text-right font-mono">
                            {row.kind === "base" ? fmt.num(row.to, 2) : row.kind === "total" ? `= ${fmt.num(row.to, 2)}` : fmt.signed(row.shap, 2)}
                          </td>
                        </tr>
                      ),
                    )}
                  </tbody>
                </table>
                <p className="mt-2 text-[12.5px] text-muted">
                  Giá trị cơ sở cộng mọi đóng góp bằng f(x) = {fmt.num(exp.margin, 2)}; điểm rủi ro = 1 / (1 + e<sup>−f(x)</sup>) ={" "}
                  {fmt.score(waterfall.probability)}. Vạch dọc là ngưỡng hiện hành quy về cùng thang ({fmt.num(waterfall.tauLogit, 2)}).
                  Giá trị cơ sở cao vì tính theo trọng số huấn luyện (scale_pos_weight ≈ 600).
                </p>
                {exp.top_positive.length < 5 && (
                  <p className="mt-2 text-[12.5px] text-muted">
                    Giao dịch này chỉ có {exp.top_positive.length} đặc trưng đẩy điểm lên — mô hình kéo gần hết đặc trưng về phía an toàn.
                    Danh sách không mượn một yếu tố kéo xuống để cho đủ năm.
                  </p>
                )}
              </>
            )}
          </section>
        )}

        <PcaNote />
      </div>

      {tx && (
        <footer className="border-t border-grid bg-surface-2 px-5.5 pb-4.5 pt-3.5">
          {decided ? (
            <div role="status" className="mb-2.5 rounded-md border border-line bg-surface px-3 py-2.5">
              <p>
                Kết luận của bạn: <strong>{REVIEW[decided]}</strong>{" "}
                {!saved && tx.review && (
                  <span className="text-muted">
                    ({fmt.datetime(tx.review.reviewed_at)}, ngưỡng khi đó {fmt.tau(tx.review.threshold_used)})
                  </span>
                )}
              </p>
              {trueLabel !== null ? (
                <p>
                  Nhãn thật: <strong>{trueLabel === 1 ? "Gian lận" : "Hợp lệ"}</strong> —{" "}
                  <span className={`font-semibold ${matches ? "text-good-text" : "text-critical-text"}`}>
                    {matches ? "✓ khớp kết luận của bạn" : "✗ khác kết luận của bạn"}
                  </span>
                </p>
              ) : (
                <p className="text-muted">Giao dịch này không có nhãn thật (tệp tải lên không có cột Class).</p>
              )}
            </div>
          ) : (
            <p className="mb-1 text-[12.5px] text-muted">Nhãn thật (nếu có) chỉ hiện sau khi bạn kết luận — để tự đối chiếu mà không bị mồi.</p>
          )}
          <label htmlFor="review-note" className="text-[13px] font-semibold text-ink-2">Ghi chú (tùy chọn)</label>
          <textarea id="review-note" rows={2} maxLength={2000} value={note} onChange={(e) => setNote(e.target.value)} className="field-input mb-2.5 mt-1 resize-y" />
          <div className="flex flex-wrap gap-2.5">
            <button type="button" className="btn btn-danger" disabled={saving} aria-pressed={decided === "confirmed_fraud"} onClick={() => decide("confirmed_fraud")}>
              <kbd>F</kbd> Xác nhận gian lận
            </button>
            <button type="button" className="btn" disabled={saving} aria-pressed={decided === "false_alarm"} onClick={() => decide("false_alarm")}>
              <kbd>A</kbd> Đánh dấu báo động sai
            </button>
          </div>
        </footer>
      )}
    </dialog>
  );
}
