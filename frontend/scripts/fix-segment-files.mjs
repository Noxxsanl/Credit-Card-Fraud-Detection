/*
 * Chạy sau `next build` (npm run build → postbuild).
 *
 * Lỗi của Next.js 16.3 khi xuất tĩnh trên Windows: tệp tải trước của từng đoạn route phải có
 * tên phẳng, ví dụ `out/threshold/__next.threshold.__PAGE__.txt` — đúng tên mà trình duyệt xin.
 * Nhưng next/dist/export/index.js dựng tên bằng `segmentPath.replace(/\//g, ".")` trên một đường
 * dẫn mang dấu `\` của Windows, nên tệp bị ghi vào thư mục con
 * `out/threshold/__next.threshold/__PAGE__.txt`. Hậu quả: mỗi lần tải trang, console báo 404 cho
 * các liên kết được tải trước (điều hướng vẫn chạy, nhưng docs/08 §5 đòi console sạch).
 *
 * Kịch bản này dời các tệp đó về đúng tên phẳng. Build trên Linux (Docker) không sinh thư mục
 * như vậy nên kịch bản không làm gì; chạy lại nhiều lần cũng không sao.
 */

import { readdir, rename, rm, stat } from "node:fs/promises";
import path from "node:path";

const OUT = path.resolve(import.meta.dirname, "..", "out");

async function files(dir) {
  const out = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...(await files(full)));
    else out.push(full);
  }
  return out;
}

async function fix(dir) {
  let moved = 0;
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const full = path.join(dir, entry.name);
    if (entry.name.startsWith("__next.")) {
      for (const file of await files(full)) {
        const flat = [entry.name, ...path.relative(full, file).split(path.sep)].join(".");
        await rename(file, path.join(dir, flat));
        moved++;
      }
      await rm(full, { recursive: true, force: true });
    } else {
      moved += await fix(full);
    }
  }
  return moved;
}

try {
  await stat(OUT);
} catch {
  console.log("fix-segment-files: chưa có thư mục out/ — bỏ qua");
  process.exit(0);
}
const moved = await fix(OUT);
console.log(`fix-segment-files: đã làm phẳng ${moved} tệp tải trước`);
