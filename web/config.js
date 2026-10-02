/*
 * Địa chỉ API mà giao diện gọi tới.
 *
 * Mặc định: cùng máy với trang, cổng 8000 — đúng cách chạy khi phát triển
 * (uvicorn api.main:app --port 8000, docs/10 §4.2). Trang phải được phục vụ qua
 * http://localhost:3000 hoặc http://127.0.0.1:3000: đó là hai origin CORS của API cho phép
 * (api/config.py, biến CORS_ORIGINS). Mở thẳng tệp index.html (file://) sẽ bị chặn.
 *
 * Khi đóng gói (giai đoạn 9) chỉ cần thay tệp này, không phải sửa app.js.
 */
window.FRAUD_CONSOLE_CONFIG = {
  apiBase: `${location.protocol === "https:" ? "https:" : "http:"}//${location.hostname || "localhost"}:8000/api/v1`,
};
