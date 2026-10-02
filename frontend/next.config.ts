import type { NextConfig } from "next";

/*
 * Giao diện là ứng dụng một trang thuần phía máy khách: mọi dữ liệu lấy từ API FastAPI lúc chạy.
 * Xuất tĩnh (`output: "export"`) cho ra thư mục `out/` phục vụ được bằng bất kỳ máy chủ tệp tĩnh
 * nào — cùng kiến trúc với docs/03 §6.3, không cần tiến trình Node lúc chạy.
 *
 * `trailingSlash`: `/threshold` được ghi thành `out/threshold/index.html`, nên
 * `python -m http.server` cũng phục vụ đúng mà không cần luật viết lại địa chỉ.
 */
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
