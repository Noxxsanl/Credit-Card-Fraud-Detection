"use client";

/*
 * Thư viện giao dịch mẫu (API-14 → API-02). Mỗi mẫu một lời gọi POST /score kèm sample_id:
 * giao dịch được lưu với nhãn thật của mẫu, nên UI-02 đối chiếu được kết luận với sự thật.
 * /score/batch nhanh hơn nhưng không giữ được nhãn.
 */

import { useEffect, useRef, useState } from "react";

import { useApp } from "@/context/AppContext";
import { api } from "@/lib/api";
import { fmt } from "@/lib/format";
import { CATEGORY, DECISION } from "@/lib/labels";
import type { SampleCategory, SampleItem, SampleList, ScoreResponse } from "@/lib/types";

type Result = ScoreResponse | { error: string };
const isError = (r: Result | undefined): r is { error: string } => Boolean(r && "error" in r);

export function SamplesDialog() {
  const { samplesRequest, bumpQueue, toast } = useApp();
  const dialog = useRef<HTMLDialogElement>(null);
  const returnFocus = useRef<HTMLElement | null>(null);
  const [data, setData] = useState<SampleList | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [category, setCategory] = useState<SampleCategory>("fraud_easy");
  const [results, setResults] = useState<Record<string, Result>>({});
  const [busy, setBusy] = useState(false);

  // Mỗi yêu cầu mở mới: mở hộp thoại, nhớ phần tử để trả trọng tâm khi đóng
  useEffect(() => {
    if (samplesRequest === 0) return;
    returnFocus.current = document.activeElement as HTMLElement | null;
    if (dialog.current && !dialog.current.open) dialog.current.showModal();
  }, [samplesRequest]);

  // Tải thư viện một lần, ở lần mở đầu tiên
  useEffect(() => {
    if (samplesRequest === 0 || data || loadError) return;
    api<SampleList>("/samples").then(setData, (e: Error) => setLoadError(e.message));
  }, [samplesRequest, data, loadError]);

  const visible = data ? data.items.filter((item) => item.category === category) : [];
  const pending = visible.filter((item) => !results[item.id] || isError(results[item.id]));

  async function scoreOne(item: SampleItem): Promise<ScoreResponse | null> {
    try {
      const r = await api<ScoreResponse>("/score", {
        method: "POST",
        json: { transaction: item.features, sample_id: item.id, persist: true },
      });
      setResults((old) => ({ ...old, [item.id]: r }));
      return r;
    } catch (e) {
      setResults((old) => ({ ...old, [item.id]: { error: (e as Error).message } }));
      return null;
    }
  }

  async function scoreSingle(item: SampleItem) {
    setBusy(true);
    const r = await scoreOne(item);
    setBusy(false);
    if (r) {
      bumpQueue();
      toast(`${item.id} → ${r.transaction_id}: ${fmt.score(r.risk_score)}, ${DECISION[r.decision].toLowerCase()}`, "success");
    }
  }

  /** Nạp cả nhóm, bốn yêu cầu song song. */
  async function scoreCategory() {
    const queue = [...pending];
    if (!queue.length) return;
    setBusy(true);
    let done = 0;
    let alerts = 0;
    const worker = async () => {
      for (let item = queue.shift(); item; item = queue.shift()) {
        const r = await scoreOne(item);
        if (r) {
          done++;
          if (r.decision !== "allow") alerts++;
        }
      }
    };
    await Promise.all([worker(), worker(), worker(), worker()]);
    setBusy(false);
    bumpQueue();
    toast(`Đã nạp ${done} mẫu nhóm "${CATEGORY[category]}", ${alerts} vào hàng đợi`, "success");
  }

  const onClose = () => {
    const target = returnFocus.current;
    returnFocus.current = null;
    if (target && document.contains(target)) target.focus();
  };

  return (
    <dialog ref={dialog} className="modal" aria-labelledby="samples-title" onClose={onClose}>
      <header className="flex items-start justify-between gap-3 border-b border-grid px-5.5 pb-3.5 pt-4.5">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-muted">Nạp dữ liệu</p>
          <h2 id="samples-title" className="text-xl font-bold">Thư viện giao dịch mẫu</h2>
        </div>
        <button type="button" className="btn btn-ghost" onClick={() => dialog.current?.close()}>
          Đóng <kbd>Esc</kbd>
        </button>
      </header>
      <div className="overflow-y-auto px-5.5 pb-5.5 pt-4">
        <p className="text-muted">
          200 giao dịch lấy từ tập kiểm thử. Chấm một mẫu là ghi nó vào cơ sở dữ liệu kèm nhãn thật; mẫu vượt ngưỡng hiện hành sẽ vào
          hàng đợi.
        </p>
        {!data && !loadError && <p>Đang tải thư viện…</p>}
        {loadError && <p role="alert" className="text-[12.5px] font-semibold text-critical-text">{loadError}</p>}
        {data && (
          <>
            <div role="group" aria-label="Nhóm mẫu" className="segmented my-1.5">
              {(Object.keys(data.categories) as SampleCategory[]).map((key) => (
                <button key={key} type="button" aria-pressed={category === key} onClick={() => setCategory(key)}>
                  {CATEGORY[key]} ({data.categories[key].count})
                </button>
              ))}
            </div>
            <p className="mt-2 text-[12.5px] text-muted">Nhóm này: {data.categories[category].rule}</p>
            <div className="my-3">
              <button type="button" className="btn btn-primary" disabled={busy || !pending.length} onClick={scoreCategory}>
                {busy ? "Đang chấm…" : `Nạp ${pending.length} mẫu còn lại của nhóm này`}
              </button>
            </div>
            <div className="max-h-[420px] overflow-auto">
              <table className="data-table compact">
                <thead>
                  <tr>
                    <th scope="col">Mẫu</th>
                    <th scope="col">Mô tả</th>
                    <th scope="col" className="num">Số tiền</th>
                    <th scope="col">Kết quả</th>
                    <th scope="col"><span className="sr-only">Hành động</span></th>
                  </tr>
                </thead>
                <tbody>
                  {visible.map((item) => {
                    const r = results[item.id];
                    return (
                      <tr key={item.id}>
                        <td className="font-mono">{item.id}</td>
                        <td className="min-w-[220px] whitespace-normal!">{item.description}</td>
                        <td className="num">{fmt.num(item.amount)}</td>
                        <td>
                          {r && !isError(r) && (
                            <span>
                              <span className="font-mono">{r.transaction_id}</span> · {fmt.score(r.risk_score)} · {DECISION[r.decision]}
                            </span>
                          )}
                          {isError(r) && <span className="font-semibold text-critical-text">{r.error}</span>}
                        </td>
                        <td>
                          <button type="button" className="btn btn-sm" disabled={busy} onClick={() => scoreSingle(item)}>
                            {r && !isError(r) ? "Chấm lại" : "Chấm điểm"}
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </dialog>
  );
}
