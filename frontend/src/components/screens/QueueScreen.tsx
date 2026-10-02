"use client";

/*
 * UI-01 — Hàng đợi thẩm định (docs/07 §3). Màn hình mặc định.
 *
 * GET /transactions: máy chủ mặc định chỉ trả giao dịch có điểm ≥ ngưỡng hiện hành, sắp theo
 * điểm giảm dần (FR-40, AC-A2). Đổi ngưỡng ở UI-03 thì danh sách tải lại, không tải lại trang.
 */

import Link from "next/link";
import { useCallback, useEffect, useRef, useState, type ChangeEvent, type KeyboardEvent, type ReactNode } from "react";

import { useApp } from "@/context/AppContext";
import { api, query } from "@/lib/api";
import { clamp, fmt } from "@/lib/format";
import { REVIEW, SOURCE } from "@/lib/labels";
import type { ReviewStatus, TransactionItem, TransactionPage, UploadResponse } from "@/lib/types";

import { DecisionBadge, ReviewStatusText, RiskScore, Skeleton } from "../ui";

const PAGE_SIZE = 25;
const STATUS_FILTERS: [ReviewStatus | "all", string][] = [
  ["all", "Tất cả"],
  ["pending", REVIEW.pending],
  ["confirmed_fraud", REVIEW.confirmed_fraud],
  ["false_alarm", REVIEW.false_alarm],
];

type Sort = "-risk_score" | "risk_score" | "-amount" | "amount";
type UploadResult = UploadResponse & { fileName: string };

/** ‹ 1 2 3 … 8 › — luôn có hai trang đầu, hai trang cuối và hai bên trang hiện tại */
function pageItems(page: number, pages: number): (number | string)[] {
  if (pages <= 7) return Array.from({ length: pages }, (_, i) => i + 1);
  const keep = [...new Set([1, 2, page - 1, page, page + 1, pages - 1, pages])]
    .filter((x) => x >= 1 && x <= pages)
    .sort((a, b) => a - b);
  const out: (number | string)[] = [];
  keep.forEach((x, i) => {
    if (i && x - keep[i - 1] > 1) out.push(`gap-${x}`);
    out.push(x);
  });
  return out;
}

export function QueueScreen() {
  const { threshold, summary, queueVersion, reviewed, openTx, openSamples, bumpQueue, toast } = useApp();
  const [items, setItems] = useState<TransactionItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [sort, setSort] = useState<Sort>("-risk_score");
  const [status, setStatus] = useState<ReviewStatus | "all">("all");
  const [batchId, setBatchId] = useState<string | null>(null);
  const [loadedKey, setLoadedKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hasAnyData, setHasAnyData] = useState<boolean | null>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [reloadTick, setReloadTick] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [uploadError, setUploadError] = useState<{ message: string; missing?: string[] } | null>(null);
  const tbody = useRef<HTMLTableSectionElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const tau = threshold?.current;
  // Mỗi tổ hợp tham số là một yêu cầu; "đang tải" = kết quả đang hiện chưa ứng với yêu cầu mới nhất
  const requestKey = JSON.stringify([sort, page, status, batchId, tau, queueVersion, reloadTick]);
  const loaded = loadedKey !== null;
  const loading = requestKey !== loadedKey;

  useEffect(() => {
    let cancelled = false;
    const params: Record<string, string | number | null> = { sort, page, page_size: PAGE_SIZE, batch_id: batchId };
    if (status !== "all") params.review_status = status;
    (async () => {
      try {
        const data = await api<TransactionPage>(`/transactions${query(params)}`);
        if (cancelled) return;
        setItems(data.items);
        setTotal(data.total);
        setActiveIndex(0);
        setError(null);
        if (data.total === 0) {
          // Phân biệt "bảng trống" (hướng dẫn ba cách nạp) với "bộ lọc không khớp"
          const any = await api<TransactionPage>(`/transactions${query({ min_score: 0, page_size: 1 })}`);
          if (!cancelled) setHasAnyData(any.total > 0);
        } else {
          setHasAnyData(true);
        }
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      } finally {
        if (!cancelled) setLoadedKey(requestKey);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [sort, page, status, batchId, tau, queueVersion, reloadTick, requestKey]);

  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const goto = (p: number) => {
    if (p >= 1 && p <= pages && p !== page) setPage(p);
  };
  const toggleSort = (key: "risk_score" | "amount") => {
    setSort((s) => (s === `-${key}` ? key : `-${key}`) as Sort);
    setPage(1);
  };
  const ariaSort = (key: string) => (sort === key ? "ascending" : sort === `-${key}` ? "descending" : "none");
  const arrow = (key: string) => (sort === key ? "▲" : sort === `-${key}` ? "▼" : "");

  const focusRow = useCallback((i: number) => {
    const row = tbody.current?.querySelector<HTMLTableRowElement>(`[data-row="${i}"]`);
    if (row) {
      setActiveIndex(i);
      row.focus();
    }
  }, []);

  /** ↑/↓ giữa các dòng, Enter mở chi tiết (07 §3). Chỉ một dòng nằm trong thứ tự Tab.
   *  Không mở bằng phím cách: ngăn kéo mở lúc nhấn, trọng tâm nhảy sang nút "Đóng",
   *  và lúc nhả phím nút đó bị bấm. */
  function onRowKey(event: KeyboardEvent<HTMLTableRowElement>, i: number) {
    const last = items.length - 1;
    const moves: Record<string, number> = { ArrowDown: i + 1, ArrowUp: i - 1, Home: 0, End: last };
    if (event.key in moves) {
      event.preventDefault();
      focusRow(clamp(moves[event.key], 0, last));
    } else if (event.key === "Enter") {
      event.preventDefault();
      openTx(items[i].id, event.currentTarget);
    }
  }

  async function upload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    setUploading(true);
    setUploadError(null);
    setUploadResult(null);
    try {
      const result = await api<UploadResponse>("/score/upload", { method: "POST", form });
      setUploadResult({ ...result, fileName: file.name });
      toast(`Đã chấm ${fmt.int(result.count)} giao dịch, ${fmt.int(result.alerts)} vượt ngưỡng`, "success");
      setPage(1);
      bumpQueue();
    } catch (e) {
      const details = (e as { details?: { missing?: string[] } }).details;
      setUploadError({ message: (e as Error).message, missing: details?.missing });
    } finally {
      setUploading(false);
    }
  }

  const pickFile = () => fileInput.current?.click();

  return (
    <div>
      {summary && (
        <p className="mb-3.5 rounded-md border border-l-[3px] border-line border-l-accent bg-surface px-3.5 py-2.5">
          Với ngưỡng này, trên {fmt.int(testSetSize(summary))} giao dịch kiểm thử: <strong>{fmt.int(summary.alerts)} cảnh báo</strong> (≈{" "}
          {fmt.int(summary.alerts_per_day)}/ngày) · ước tính bắt được <strong>{fmt.pct(summary.recall, 0)} gian lận</strong> ({summary.tp}/
          {summary.tp + summary.fn})
        </p>
      )}

      <div className="mb-3.5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-2">
          <button type="button" className="btn" onClick={pickFile} disabled={uploading}>
            {uploading ? "Đang chấm điểm…" : "Tải CSV"}
          </button>
          <input ref={fileInput} type="file" accept=".csv,text/csv" className="sr-only" tabIndex={-1} aria-hidden="true" onChange={upload} />
          <button type="button" className="btn" onClick={openSamples}>
            Chọn mẫu có sẵn
          </button>
          <Link href="/replay" className="btn">
            ▶ Phát lại ngày 2
          </Link>
        </div>
        <div role="group" aria-label="Lọc theo trạng thái thẩm định" className="segmented">
          {STATUS_FILTERS.map(([key, label]) => (
            <button key={key} type="button" aria-pressed={status === key} onClick={() => { setStatus(key); setPage(1); }}>
              {label}
            </button>
          ))}
        </div>
      </div>

      {uploading && (
        <div role="status" className="mb-3 rounded-md bg-accent-soft px-3.5 py-2.5 font-semibold text-accent-strong">
          Đang tải lên và chấm điểm tệp — 10.000 dòng mất khoảng 2 giây (AC-A1: dưới 30 giây).
        </div>
      )}
      {uploadError && (
        <div role="alert" className="card mb-4 border-[#efb9b6] bg-critical-soft">
          <div className="flex items-center justify-between gap-3">
            <strong>Không chấm được tệp: {uploadError.message}</strong>
            <button type="button" className="btn btn-ghost btn-sm" aria-label="Đóng thông báo" onClick={() => setUploadError(null)}>✕</button>
          </div>
          {uploadError.missing && (
            <p>
              Thiếu cột: <code>{uploadError.missing.join(", ")}</code>. Tệp cần đủ 30 cột Time, V1…V28, Amount; cột Class nếu có sẽ thành
              nhãn thật.
            </p>
          )}
        </div>
      )}
      {uploadResult && <UploadCard result={uploadResult} onClose={() => setUploadResult(null)} onShowBatch={(id) => { setBatchId(id); setStatus("all"); setPage(1); }} />}

      <div className="mb-2 mt-1 flex min-h-[30px] items-center gap-3">
        {loaded && !error && <p className="text-muted">{fmt.int(total)} giao dịch vượt ngưỡng khớp bộ lọc</p>}
        {batchId && (
          <span className="inline-flex items-center gap-1.5 rounded-full bg-accent-soft py-0.5 pl-2.5 pr-1 text-[12.5px]">
            Lô <span className="font-mono">{batchId}</span>
            <button type="button" aria-label="Bỏ lọc theo lô" className="rounded-full px-1.5" onClick={() => { setBatchId(null); setPage(1); }}>✕</button>
          </span>
        )}
        <button type="button" className="btn btn-ghost btn-sm ml-auto" disabled={loading} onClick={() => setReloadTick((n) => n + 1)}>
          Làm mới
        </button>
      </div>

      {!loaded && (
        <div aria-hidden="true" className="overflow-hidden rounded-[10px] border border-line bg-surface">
          {Array.from({ length: 5 }, (_, i) => (
            <div key={i} className="border-b border-grid px-3 py-3.5 last:border-0"><Skeleton /></div>
          ))}
        </div>
      )}

      {error && (
        <div role="alert" className="rounded-[10px] border border-dashed border-line-strong bg-surface p-7 text-center">
          <p><strong>Không tải được hàng đợi.</strong> {error}</p>
          <button type="button" className="btn mt-2" onClick={() => setReloadTick((n) => n + 1)}>Thử lại</button>
        </div>
      )}

      {loaded && !error && items.length > 0 && (
        <div className={`overflow-x-auto rounded-[10px] border border-line bg-surface transition-opacity ${loading ? "opacity-55" : ""}`}>
          <table className="data-table">
            <caption className="sr-only">
              Giao dịch có điểm rủi ro từ ngưỡng hiện hành trở lên. Dùng mũi tên lên, xuống để di chuyển giữa các dòng, Enter để mở chi tiết.
            </caption>
            <thead>
              <tr>
                <th scope="col" aria-sort={ariaSort("risk_score")}>
                  <button type="button" className="cursor-pointer hover:underline" onClick={() => toggleSort("risk_score")}>
                    Rủi ro <span aria-hidden="true">{arrow("risk_score")}</span>
                  </button>
                </th>
                <th scope="col">Mã GD</th>
                <th scope="col" className="num" aria-sort={ariaSort("amount")}>
                  <button type="button" className="cursor-pointer hover:underline" onClick={() => toggleSort("amount")}>
                    Số tiền (EUR) <span aria-hidden="true">{arrow("amount")}</span>
                  </button>
                </th>
                <th scope="col">Giờ</th>
                <th scope="col">Đề xuất</th>
                <th scope="col">Trạng thái</th>
                <th scope="col">Nguồn</th>
                <th scope="col"><span className="sr-only">Mở chi tiết</span></th>
              </tr>
            </thead>
            <tbody ref={tbody}>
              {items.map((item, i) => (
                <tr
                  key={item.id}
                  data-row={i}
                  tabIndex={i === activeIndex ? 0 : -1}
                  className="row-link"
                  onClick={(e) => openTx(item.id, e.currentTarget)}
                  onKeyDown={(e) => onRowKey(e, i)}
                  onFocus={() => setActiveIndex(i)}
                >
                  <td><RiskScore score={item.risk_score} /></td>
                  <td className="font-mono">{item.id}</td>
                  <td className="num">{fmt.num(item.amount)}</td>
                  <td>{fmt.hour(item.hour)}</td>
                  <td><DecisionBadge decision={item.decision} /></td>
                  <td><ReviewStatusText status={reviewed[item.id] ?? item.review_decision ?? "pending"} /></td>
                  <td className="text-muted">{SOURCE[item.source]}</td>
                  <td className="w-6 text-muted"><span aria-hidden="true">▸</span><span className="sr-only">Mở chi tiết</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {loaded && !error && pages > 1 && (
        <nav aria-label="Phân trang hàng đợi" className="mt-3 flex items-center justify-end gap-1">
          <PagerButton label="Trang trước" disabled={page === 1} onClick={() => goto(page - 1)}>‹</PagerButton>
          {pageItems(page, pages).map((p) =>
            typeof p === "number" ? (
              <PagerButton key={p} label={`Trang ${p}`} current={p === page} onClick={() => goto(p)}>{p}</PagerButton>
            ) : (
              <span key={p} aria-hidden="true" className="px-1.5 text-muted">…</span>
            ),
          )}
          <PagerButton label="Trang sau" disabled={page === pages} onClick={() => goto(page + 1)}>›</PagerButton>
        </nav>
      )}

      {loaded && !error && total === 0 && hasAnyData === false && (
        <div className="rounded-[10px] border border-dashed border-line-strong bg-surface p-7">
          <h2 className="mb-1.5 text-lg font-bold">Chưa có giao dịch nào để thẩm định</h2>
          <p className="text-muted">Hàng đợi đọc từ cơ sở dữ liệu. Nạp dữ liệu bằng một trong ba cách:</p>
          <ol className="mt-4 grid grid-cols-1 gap-3.5 lg:grid-cols-3">
            <Way n={1} title="Tải tệp CSV" action={<button type="button" className="btn" onClick={pickFile}>Chọn tệp CSV</button>}>
              Đủ 30 cột <code>Time</code>, <code>V1</code>…<code>V28</code>, <code>Amount</code>; cột <code>Class</code> nếu có sẽ thành nhãn thật. Tối đa 100 MB.
            </Way>
            <Way n={2} title="Chọn mẫu có sẵn" action={<button type="button" className="btn" onClick={openSamples}>Mở thư viện mẫu</button>}>
              200 giao dịch của tập kiểm thử, chia bốn nhóm: gian lận dễ/khó, hợp lệ dễ/khó.
            </Way>
            <Way n={3} title="Phát lại ngày 2" action={<Link href="/replay" className="btn">Mở chế độ phát lại</Link>}>
              Dòng giao dịch chảy theo thời gian mô phỏng; giao dịch vượt ngưỡng rơi vào hàng đợi.
            </Way>
          </ol>
        </div>
      )}
      {loaded && !error && total === 0 && hasAnyData && (
        <div className="rounded-[10px] border border-dashed border-line-strong bg-surface p-7 text-center">
          <p><strong>Không có giao dịch vượt ngưỡng nào khớp bộ lọc.</strong></p>
          <p className="text-muted">
            Hàng đợi chỉ gồm giao dịch có điểm rủi ro từ ngưỡng hiện hành trở lên. Đổi bộ lọc trạng thái, bỏ lọc theo lô, hoặc hạ ngưỡng ở
            màn hình Ngưỡng.
          </p>
        </div>
      )}
    </div>
  );
}

/** Số giao dịch của tập kiểm thử = tổng bốn ô của ma trận nhầm lẫn */
function testSetSize(m: { tp: number; fp: number; fn: number; tn: number }): number {
  return m.tp + m.fp + m.fn + m.tn;
}

function PagerButton({ children, label, current, disabled, onClick }: {
  children: ReactNode;
  label: string;
  current?: boolean;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      aria-current={current ? "page" : undefined}
      disabled={disabled}
      onClick={onClick}
      className={`min-h-8 min-w-8 rounded-md border px-2 ${current ? "border-ink bg-ink font-bold text-white" : "border-line bg-surface"} disabled:cursor-not-allowed disabled:opacity-40`}
    >
      {children}
    </button>
  );
}

function Way({ n, title, action, children }: { n: number; title: string; action: ReactNode; children: ReactNode }) {
  return (
    <li className="flex flex-col gap-1.5 rounded-[10px] border border-line bg-surface-2 p-4">
      <span aria-hidden="true" className="grid size-[26px] place-items-center rounded-full bg-ink text-[13px] font-bold text-white">{n}</span>
      <strong>{title}</strong>
      <span className="flex-1 text-[13px] text-ink-2">{children}</span>
      <span className="self-start">{action}</span>
    </li>
  );
}

function UploadCard({ result, onClose, onShowBatch }: { result: UploadResult; onClose: () => void; onShowBatch: (id: string) => void }) {
  const m = result.actual_metrics;
  return (
    <div role="status" className="card mb-4 px-4.5 py-3.5">
      <div className="flex items-center justify-between gap-3">
        <strong>Đã chấm tệp {result.fileName}</strong>
        <button type="button" className="btn btn-ghost btn-sm" aria-label="Đóng kết quả tải tệp" onClick={onClose}>✕</button>
      </div>
      <dl className="my-2.5 grid grid-cols-[repeat(auto-fill,minmax(170px,1fr))] gap-x-4.5 gap-y-2.5 tabular-nums">
        <Fact label="Dòng đọc được" value={fmt.int(result.rows_read)} />
        <Fact label="Đã chấm và lưu" value={fmt.int(result.count)} />
        <Fact label="Vượt ngưỡng" value={fmt.int(result.alerts)} />
        <Fact label="Dòng bị loại" value={fmt.int(result.rows_rejected)} />
        <Fact label="Thời gian phía máy chủ" value={`${fmt.num(result.elapsed_ms / 1000, 1)} giây`} />
        {m && <Fact label="Trên nhãn của tệp" value={`recall ${fmt.pct(m.recall)} · precision ${fmt.pct(m.precision)} · PR-AUC ${fmt.num(m.pr_auc, 3)}`} />}
      </dl>
      {result.rejection_reasons.length > 0 && (
        <details className="mb-2">
          <summary>Lý do loại {result.rejection_reasons.length} dòng đầu tiên</summary>
          <ul className="ml-5 list-disc text-[13px]">
            {result.rejection_reasons.slice(0, 20).map((r) => (
              <li key={r.row}><span className="font-mono">Dòng {r.row}</span>: {r.reason}</li>
            ))}
          </ul>
        </details>
      )}
      <button type="button" className="btn btn-sm" onClick={() => onShowBatch(result.batch_id)}>Chỉ xem lô này</button>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs text-muted">{label}</dt>
      <dd className="font-semibold">{value}</dd>
    </div>
  );
}
