/*
 * Màu cho biểu đồ SVG (Recharts cần giá trị màu cụ thể trong thuộc tính stroke/fill).
 * Cùng giá trị với các biến --color-* ở app/globals.css — sửa một nơi thì sửa cả hai.
 *
 * Năm màu phân loại theo bảng tham chiếu đã kiểm bằng validate_palette (đạt kiểm tra mù màu
 * cho các cặp kề nhau); ba màu nhạt dưới 3:1 trên nền trắng nên mọi biểu đồ nhiều đường đều
 * kèm bảng số liệu. Màu đi theo chiến lược, không theo thứ hạng.
 */

export const COLORS = {
  ink: "#0b0b0b",
  ink2: "#4a4945",
  ink3: "#8a8983",
  grid: "#e4e3dd",
  axis: "#c3c2b7",
  surface: "#ffffff",
  series1: "#2a78d6",
  series2: "#eb6834",
  series3: "#1baf7a",
  series4: "#eda100",
  series5: "#e87ba4",
} as const;

export const STRATEGY_COLORS: Record<string, string> = {
  class_weight: COLORS.series1,
  smote: COLORS.series2,
  smote_tomek: COLORS.series3,
  none: COLORS.series4,
  undersample: COLORS.series5,
};
