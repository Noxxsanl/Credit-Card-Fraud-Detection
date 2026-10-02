/*
 * Gọi API FastAPI — mô hình lỗi thống nhất của docs/05 §4.
 *
 * Địa chỉ API: biến NEXT_PUBLIC_API_BASE lúc build nếu có; không thì cùng máy với trang, cổng
 * 8000 (uvicorn khi phát triển). Trang phải mở ở http://localhost:3000 hoặc
 * http://127.0.0.1:3000 — hai origin CORS mà API cho phép (api/config.py).
 */

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public details: Record<string, unknown> | null = null,
  ) {
    super(message);
  }
}

export function apiBase(): string {
  const configured = process.env.NEXT_PUBLIC_API_BASE;
  if (configured) return configured.replace(/\/$/, "");
  if (typeof window === "undefined") return "http://localhost:8000/api/v1";
  const protocol = window.location.protocol === "https:" ? "https:" : "http:";
  return `${protocol}//${window.location.hostname || "localhost"}:8000/api/v1`;
}

interface Options {
  method?: "GET" | "POST" | "PUT";
  json?: unknown;
  form?: FormData;
  signal?: AbortSignal;
}

export async function api<T>(path: string, { method = "GET", json, form, signal }: Options = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(apiBase() + path, {
      method,
      headers: json === undefined ? undefined : { "Content-Type": "application/json" },
      body: json === undefined ? form : JSON.stringify(json),
      signal,
    });
  } catch (error) {
    if ((error as Error).name === "AbortError") throw error;
    throw new ApiError(0, "NETWORK", `Không kết nối được API tại ${apiBase()}. API đã chạy chưa?`);
  }
  const text = await response.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = null;
  }
  if (!response.ok) {
    const err = (body as { error?: { code?: string; message?: string; details?: Record<string, unknown> } } | null)?.error ?? {};
    throw new ApiError(response.status, err.code ?? `HTTP_${response.status}`,
      err.message ?? `Lỗi HTTP ${response.status}`, err.details ?? null);
  }
  return body as T;
}

export function query(params: Record<string, string | number | boolean | null | undefined>): string {
  const q = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== null && value !== undefined && value !== "") q.set(key, String(value));
  }
  const s = q.toString();
  return s ? `?${s}` : "";
}

export const isAbort = (error: unknown) => (error as Error)?.name === "AbortError";
