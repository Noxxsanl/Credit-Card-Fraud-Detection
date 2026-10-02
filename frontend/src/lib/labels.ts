/* Nhãn tiếng Việt cho các giá trị mã của API và metrics.json. */

import type { Decision, ReviewStatus, SampleCategory, Source } from "./types";

export const DECISION: Record<Decision, string> = {
  allow: "Cho qua",
  review: "Cần thẩm định",
  block: "Đề xuất chặn",
};

export const REVIEW: Record<ReviewStatus, string> = {
  pending: "Chờ xử lý",
  confirmed_fraud: "Xác nhận gian lận",
  false_alarm: "Báo động sai",
};

export const SOURCE: Record<Source, string> = {
  upload: "Tải CSV",
  sample: "Mẫu",
  replay: "Phát lại",
  manual: "Nhập tay",
};

export const MODEL: Record<string, string> = {
  logistic_regression: "Hồi quy logistic",
  decision_tree: "Cây quyết định",
  random_forest: "Rừng ngẫu nhiên",
  xgboost: "XGBoost",
};

export const STRATEGY: Record<string, string> = {
  none: "Không xử lý",
  class_weight: "Trọng số lớp",
  smote: "SMOTE",
  smote_tomek: "SMOTE + Tomek",
  undersample: "Giảm mẫu lớp đa số",
};

export const SPLIT: Record<string, string> = {
  random_stratified: "Ngẫu nhiên phân tầng 80/20 (dùng trong báo cáo)",
  random_downsized_day1: "Ngẫu nhiên, tập huấn luyện thu nhỏ bằng ngày 1",
  temporal: "Theo thời gian: học ngày 1, kiểm ngày 2",
};

export const CATEGORY: Record<SampleCategory, string> = {
  fraud_easy: "Gian lận dễ",
  fraud_hard: "Gian lận khó",
  legit_easy: "Hợp lệ dễ",
  legit_hard: "Hợp lệ khó",
};

/** Ngưỡng gợi ý sẵn của UI-03 — khóa của GET /threshold → alternatives, theo thứ tự 07 §5. */
export const ALTERNATIVES = [
  { key: "default_naive", label: "Mặc định 0,5", hint: "Không có cơ sở cho dữ liệu mất cân bằng 1:578" },
  { key: "min_expected_cost", label: "Cực tiểu chi phí kỳ vọng (τ*)", hint: "Chọn trên out-of-fold với chi phí của threshold.json" },
  { key: "max_f1", label: "F1 lớn nhất", hint: "Dùng khi không ước lượng được chi phí" },
  { key: "recall_at_least_90", label: "Ràng buộc recall ≥ 90%", hint: "Ngưỡng lớn nhất còn giữ recall 90% trên out-of-fold" },
  { key: "budget_200_alerts", label: "Ngân sách 200 lượt thẩm định/ngày", hint: "Ngưỡng nhỏ nhất còn nằm trong ngân sách trên out-of-fold" },
] as const;

export type RiskLevelKey = "critical" | "serious" | "warning";

/** Màu theo mức điểm tuyệt đối (07 §3): đỏ ≥ 90%, cam 60–90%, vàng < 60% — luôn kèm hình dạng. */
export function riskLevel(score: number): { key: RiskLevelKey; shape: string; text: string } {
  if (score >= 0.9) return { key: "critical", shape: "▲", text: "rất cao (≥ 90%)" };
  if (score >= 0.6) return { key: "serious", shape: "◆", text: "cao (60–90%)" };
  return { key: "warning", shape: "●", text: "dưới 60%" };
}

/** Tên mô hình trong baseline_comparison của metrics.json → nhãn tiếng Việt */
export function modelLabel(name: string): string {
  if (MODEL[name]) return MODEL[name];
  if (name === "mô hình rỗng") return "Mô hình rỗng (luôn đoán hợp lệ)";
  const exported = name.match(/^(\w+) \+ (\w+) \((.+)\)$/);
  if (exported) {
    return `${MODEL[exported[1]] ?? exported[1]} + ${(STRATEGY[exported[2]] ?? exported[2]).toLowerCase()} (${exported[3]})`;
  }
  return name.charAt(0).toUpperCase() + name.slice(1);
}

export const SCREENS = [
  { href: "/", key: "queue", label: "Hàng đợi", title: "Hàng đợi thẩm định" },
  { href: "/threshold", key: "threshold", label: "Ngưỡng", title: "Cấu hình ngưỡng quyết định" },
  { href: "/performance", key: "performance", label: "Hiệu năng", title: "Hiệu năng mô hình" },
  { href: "/replay", key: "replay", label: "Phát lại", title: "Chế độ phát lại" },
] as const;
