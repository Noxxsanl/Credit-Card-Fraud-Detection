/* Mảnh giao diện nhỏ dùng ở nhiều màn hình. */

import type { ReactNode } from "react";

import { fmt } from "@/lib/format";
import { DECISION, REVIEW, riskLevel } from "@/lib/labels";
import type { Decision, ReviewStatus } from "@/lib/types";

/** Điểm rủi ro dạng phần trăm kèm hình dạng ▲ ◆ ● — không chỉ dựa vào màu (NFR-10). */
export function RiskScore({ score, large = false }: { score: number; large?: boolean }) {
  const level = riskLevel(score);
  return (
    <span className={`inline-flex items-center font-bold tabular-nums ${large ? "gap-2.5 text-[40px] leading-none" : "gap-1.5"}`}>
      <span aria-hidden="true" className={`risk-${level.key} ${large ? "text-[22px]" : "w-3 text-center text-[11px]"} leading-none`}>
        {level.shape}
      </span>
      <span>{fmt.score(score)}</span>
      <span className="sr-only">, mức {level.text}</span>
    </span>
  );
}

const DECISION_STYLE: Record<Decision, string> = {
  block: "bg-critical-soft text-critical-text",
  review: "bg-serious-soft text-serious-text",
  allow: "bg-surface-2 text-ink-2",
};

export function DecisionBadge({ decision }: { decision: Decision }) {
  return (
    <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-bold ${DECISION_STYLE[decision]}`}>
      {DECISION[decision]}
    </span>
  );
}

const STATUS_STYLE: Record<ReviewStatus, string> = {
  pending: "text-ink-2",
  confirmed_fraud: "font-bold text-critical-text",
  false_alarm: "font-bold text-good-text",
};

export function ReviewStatusText({ status }: { status: ReviewStatus }) {
  return <span className={`text-[13px] ${STATUS_STYLE[status]}`}>{REVIEW[status]}</span>;
}

export function Tile({ label, value, sub, delta, deltaClass = "" }: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  delta?: string | null;
  deltaClass?: string;
}) {
  return (
    <div className="rounded-[10px] border border-line bg-surface px-4 py-3.5 shadow-card">
      <div className="text-[13px] font-semibold text-ink-2">{label}</div>
      <div className="mt-1 text-[28px] font-bold leading-tight tabular-nums">{value}</div>
      {sub != null && <div className="mt-0.5 text-[12.5px] text-muted">{sub}</div>}
      {delta && <div className={`mt-1.5 text-[12.5px] font-semibold ${deltaClass || "text-ink-2"}`}>{delta}</div>}
    </div>
  );
}

export function Card({ title, sub, children, className = "" }: {
  title?: ReactNode;
  sub?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {title && <h2 className="card-title">{title}</h2>}
      {sub && <p className="card-sub">{sub}</p>}
      {children}
    </section>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <span aria-hidden="true" className={`skeleton ${className}`} />;
}

/** Chú thích PCA cố định, không thu gọn được (UI-D2). */
export function PcaNote() {
  return (
    <aside role="note" className="mt-4 flex items-start gap-3 rounded-md border border-[#c8daf3] bg-accent-soft px-3.5 py-3 text-[13px]">
      <span aria-hidden="true" className="grid size-[22px] flex-none place-items-center rounded-full bg-accent font-serif font-extrabold italic text-white">
        i
      </span>
      <p>
        V1–V28 là thành phần chính sau PCA. Chúng cho biết đặc trưng nào ảnh hưởng tới dự đoán, nhưng{" "}
        <strong>không diễn giải được</strong> thành nguyên nhân nghiệp vụ.
      </p>
    </aside>
  );
}
