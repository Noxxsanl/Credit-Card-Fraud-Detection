# frontend/ — Fraud Review Console (Next.js)

Bản Next.js của giao diện 4 màn hình (docs/07). Cùng chức năng và cùng số với bản
HTML + Alpine.js trong `../web/` — bản đó giữ lại làm phương án dự phòng.

| | |
|---|---|
| Khung | Next.js 16 (App Router, Turbopack), React 19, TypeScript |
| Kiểu dáng | Tailwind CSS 4 — token màu ở `src/app/globals.css` |
| Biểu đồ | Recharts 3 — màu ở `src/lib/colors.ts` |
| Chạy | Xuất tĩnh (`output: "export"`) → thư mục `out/`, không cần Node lúc chạy |

## Chạy

Cần API ở cổng 8000 (`uvicorn api.main:app --port 8000`, xem `../docs/lenh-chay-giai-doan-7.md`)
và Node.js 20.9 trở lên.

```bash
npm install          # lần đầu
npm run dev          # phát triển: http://localhost:3000, tải lại khi sửa mã
npm run build        # xuất tĩnh ra out/ (kèm bước postbuild, xem dưới)
npm run serve        # phục vụ out/ ở cổng 3000 bằng python -m http.server
npm run lint         # ESLint, gồm các luật của React Compiler
npm run typecheck    # tsc --noEmit
```

Cổng phải là **3000**: API chỉ cho phép hai origin `http://localhost:3000` và
`http://127.0.0.1:3000` (CORS, `api/config.py`). Vì vậy không chạy cùng lúc với bản `web/`.
Địa chỉ API mặc định là cùng máy, cổng 8000; đặt `NEXT_PUBLIC_API_BASE` lúc build để đổi.

## Cấu trúc

```
src/
├── app/                      # route: / (hàng đợi), /threshold, /performance, /replay
├── components/
│   ├── AppShell.tsx          # thanh bên, thanh trên luôn hiện ngưỡng (UI-D4), dải báo lỗi
│   ├── TxDrawer.tsx          # UI-02 — ngăn kéo chi tiết, thác nước SHAP
│   ├── SamplesDialog.tsx     # thư viện 200 giao dịch mẫu
│   ├── charts/               # Recharts: đường cong chi phí, PR/ROC
│   └── screens/              # UI-01, UI-03, UI-04, UI-05
├── context/
│   ├── AppContext.tsx        # trạng thái toàn cục (docs/07 §9)
│   ├── ThresholdDraftContext.tsx  # bản nháp của UI-03, sống qua các lần chuyển trang
│   └── ReplayContext.tsx     # bộ máy phát lại SSE, chạy tiếp khi sang màn hình khác
└── lib/
    ├── threshold.mjs         # UI-D1 — TC-12 chạy chính tệp này bằng Node
    ├── scores.ts, api.ts, format.ts, labels.ts, types.ts, colors.ts
```

`src/lib/threshold.mjs` là JavaScript thuần (không TypeScript) để
`tests/test_threshold_parity.py` chạy thẳng bằng Node mà không cần biên dịch; kiểu nằm ở
`threshold.d.mts`. Sửa tệp này thì phải chạy lại TC-12:

```bash
cd .. && .venv/Scripts/python.exe -m pytest tests/test_threshold_parity.py
```

## Bước postbuild

`scripts/fix-segment-files.mjs` sửa một lỗi của Next.js 16.3 khi xuất tĩnh **trên Windows**:
tệp tải trước của từng route bị ghi vào thư mục con `__next.threshold/__PAGE__.txt` thay vì tên
phẳng `__next.threshold.__PAGE__.txt` mà trình duyệt xin, nên console báo 404 mỗi lần tải
trang. Build trên Linux (Docker) không bị, kịch bản khi đó không làm gì.

`AGENTS.md` và `CLAUDE.md` do `create-next-app` sinh ra (và `next dev` tự thêm lại): chúng nhắc
đọc tài liệu Next.js đi kèm trong `node_modules/next/dist/docs/` trước khi sửa mã.
