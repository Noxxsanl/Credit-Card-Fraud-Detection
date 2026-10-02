/*
 * Định dạng số kiểu Việt Nam — dấu phẩy thập phân, dấu chấm hàng nghìn (UI-D3).
 * Cùng quy tắc với bản web/app.js để hai bản giao diện hiện cùng một con số.
 */

const LOCALE = "vi-VN";
const cache = new Map<string, Intl.NumberFormat>();

function numberFormat(options: Intl.NumberFormatOptions): Intl.NumberFormat {
  const key = JSON.stringify(options);
  let format = cache.get(key);
  if (!format) {
    format = new Intl.NumberFormat(LOCALE, options);
    cache.set(key, format);
  }
  return format;
}

const missing = (v: number | null | undefined): v is null | undefined => v == null || !Number.isFinite(v);

const FEATURE_NOTES: Record<string, string> = { Amount: "số tiền", hour_sin: "giờ trong ngày", hour_cos: "giờ trong ngày" };

export const fmt = {
  int(v: number | null | undefined): string {
    return missing(v) ? "—" : numberFormat({ maximumFractionDigits: 0 }).format(v);
  },
  num(v: number | null | undefined, digits = 2): string {
    return missing(v) ? "—" : numberFormat({ minimumFractionDigits: digits, maximumFractionDigits: digits }).format(v);
  },
  /** Số có dấu, dùng dấu trừ thật (−): +2,31 / −0,45 */
  signed(v: number | null | undefined, digits = 2): string {
    if (missing(v)) return "—";
    const body = fmt.num(Math.abs(v), digits);
    return v > 0 ? `+${body}` : v < 0 ? `−${body}` : body;
  },
  pct(v: number | null | undefined, digits = 1): string {
    return missing(v) ? "—" : `${fmt.num(v * 100, digits)}%`;
  },
  /** Điểm rủi ro dạng phần trăm; không làm tròn 0,99999 thành "100%" hay 0,00001 thành "0%". */
  score(v: number | null | undefined): string {
    if (missing(v)) return "—";
    if (v > 0 && v < 0.0005) return "< 0,1%";
    if (v < 1 && v >= 0.9995) return "> 99,9%";
    return fmt.pct(v, 1);
  },
  /**
   * Ngưỡng: 3 chữ số thập phân khi ≥ 0,1; nhỏ hơn thì 4 chữ số có nghĩa (0,02317).
   * Sát 1 thì thêm chữ số cho tới khi thấy được nó nhỏ hơn 1: 0,99986 hiện "0,9999", không phải "1,000".
   */
  tau(v: number | null | undefined): string {
    if (missing(v)) return "—";
    if (v < 0.1) return numberFormat({ maximumSignificantDigits: 4 }).format(v);
    let digits = 3;
    while (digits < 8 && v < 1 && Number(v.toFixed(digits)) >= 1) digits++;
    return fmt.num(v, digits);
  },
  tauTick(v: number): string {
    return numberFormat({ maximumSignificantDigits: 1 }).format(v);
  },
  /** Số thực nguyên độ chính xác, chỉ đổi dấu thập phân */
  raw(v: number | null | undefined): string {
    return missing(v) ? "—" : String(v).replace(".", ",");
  },
  eur(v: number | null | undefined, digits = 2): string {
    return missing(v) ? "—" : `${fmt.num(v, digits)} EUR`;
  },
  /** Giây trong ngày → HH:MM:SS */
  clock(seconds: number | null | undefined): string {
    if (missing(seconds)) return "—";
    const s = Math.max(0, Math.floor(seconds)) % 86400;
    const pad = (n: number) => String(n).padStart(2, "0");
    return `${pad(Math.floor(s / 3600))}:${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}`;
  },
  /** Cột Time (giây kể từ giao dịch đầu tiên) → "ngày 1 · 02:14:33" */
  timeOffset(seconds: number | null | undefined): string {
    if (missing(seconds)) return "—";
    return `ngày ${Math.floor(seconds / 86400) + 1} · ${fmt.clock(seconds)}`;
  },
  hour(h: number | null | undefined): string {
    return missing(h) ? "—" : `${String(h).padStart(2, "0")}h`;
  },
  date(iso: string | null | undefined): string {
    return iso ? new Date(iso).toLocaleDateString(LOCALE) : "—";
  },
  datetime(iso: string | null | undefined): string {
    return iso ? new Date(iso).toLocaleString(LOCALE, { dateStyle: "short", timeStyle: "medium" }) : "—";
  },
  feature(name: string): string {
    return FEATURE_NOTES[name] ? `${name} (${FEATURE_NOTES[name]})` : name;
  },
  featureValue(name: string, value: number): string {
    return name === "Amount" ? `${fmt.num(value, 2)} EUR` : fmt.num(value, 2);
  },
};

/** Làm tròn lên một mốc "đẹp" cho trục: 1, 2, 2,5, 5 × 10ⁿ */
export function niceCeil(v: number): number {
  if (!(v > 0)) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(v));
  const step = [1, 2, 2.5, 5, 10].find((s) => s * magnitude >= v) ?? 10;
  return step * magnitude;
}

export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
export const logit = (p: number) => Math.log(p / (1 - p));
export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
