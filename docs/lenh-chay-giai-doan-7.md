# Lệnh chạy giai đoạn 7 — API (T-40…T-50)

Sổ lệnh để **dựng và chạy lại** tầng API: PostgreSQL trong Docker, lược đồ bằng Alembic, máy chủ
FastAPI và bộ kiểm thử tích hợp. Mọi lệnh chạy từ thư mục gốc của repo, gọi thẳng trình thông
dịch trong `.venv`. Hợp đồng API nằm ở [05](05-thiet-ke-api.md) (§7 ghi những điểm bản thi hành cụ
thể hơn hợp đồng ban đầu); lược đồ ở [06 §2](06-thiet-ke-luu-tru.md); điều kiện nghiệm thu ở
[TASKS.md](../TASKS.md).

> **Tiền đề:**
>
> - Hiện vật của notebook 08 trong `models/` (gồm `oof_scores.npz`, thêm ở giai đoạn này) và
>   `data/sample_pool.json`, `data/test_set.parquet`. Thiếu thì API vẫn chạy nhưng `/health` trả 503.
> - Docker Desktop **đang chạy**.
> - `.env` sao từ `.env.example`. Máy đã có PostgreSQL chiếm cổng 5432 thì đặt `POSTGRES_HOST_PORT=5433`
>   và sửa hai URL tương ứng ([10 §2.3](10-van-hanh-tai-lap.md)).

## 1. Trình tự

### 1.1 PowerShell

```powershell
# 0. Cấu hình (một lần)
Copy-Item .env.example .env          # rồi chỉnh cổng nếu 5432 đã bị chiếm

# 1. PostgreSQL 16 — T-40. Lần đầu tải image khoảng 30 giây
docker compose up -d db
docker compose exec db pg_isready -U fraud -d fraud          # phải ra "accepting connections"

# 2. Lược đồ — T-42
.\.venv\Scripts\python.exe -m alembic upgrade head

# 3. Kiểm thử: tự tạo rồi xoá cơ sở dữ liệu fraud_test (~3 phút)
.\.venv\Scripts\python.exe -m pytest tests\test_db.py tests\test_scoring.py tests\test_api.py

# 4. Máy chủ — tài liệu tự sinh ở http://localhost:8000/docs
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000
```

### 1.2 Git Bash

```bash
cp .env.example .env
docker compose up -d db
docker compose exec db pg_isready -U fraud -d fraud
./.venv/Scripts/python.exe -m alembic upgrade head
./.venv/Scripts/python.exe -m pytest tests/test_db.py tests/test_scoring.py tests/test_api.py
./.venv/Scripts/python.exe -m uvicorn api.main:app --reload --port 8000
```

### 1.3 Thử nhanh bằng curl

```bash
curl localhost:8000/api/v1/health
curl "localhost:8000/api/v1/samples?category=fraud_hard"
curl "localhost:8000/api/v1/threshold/preview?value=0.5"
curl -X POST localhost:8000/api/v1/threshold/optimize -H "content-type: application/json" \
     -d '{"constraint": {"type": "max_alerts_per_day", "value": 200}}'
curl -F "file=@giao-dich.csv" localhost:8000/api/v1/score/upload
curl -N "localhost:8000/api/v1/replay/stream?speed=60"      # Ctrl+C để dừng
```

### 1.4 Đặt lại dữ liệu demo

```bash
docker compose exec db psql -U fraud -d fraud \
  -c "TRUNCATE transactions, reviews RESTART IDENTITY CASCADE; DELETE FROM settings WHERE key = 'threshold';"
```

Lệnh này giữ `cost_fn`, `cost_fp`, `replay_speed`. Xoá khóa `threshold` là quay về τ\* của
`threshold.json`.

## 2. Tệp của giai đoạn này

| Tệp | Nội dung |
|---|---|
| `docker-compose.yml` | dịch vụ `db`: `postgres:16-alpine`, volume `pgdata`, healthcheck `pg_isready`, cổng `${POSTGRES_HOST_PORT:-5432}` |
| `.env.example` | mẫu biến môi trường, commit được |
| `alembic.ini` | ở gốc repo, **chỉ ASCII** |
| `api/migrations/versions/0001_initial_schema.py` | ba bảng, chỉ mục, `CHECK`, dãy `transaction_id_seq` |
| `api/migrations/versions/0002_seed_settings.py` | `cost_fn`, `cost_fp`, `replay_speed` |
| `api/config.py`, `db.py`, `models_orm.py` | cấu hình, engine (`pool_pre_ping`, `connect_timeout=3`), bảng ORM |
| `api/loader.py` | nạp và kiểm hiện vật; chấm lại 1/50 tập kiểm thử để bắt lệch phiên bản thư viện |
| `api/errors.py` | mô hình lỗi chung; `MISSING_FEATURES` kèm tên cột thiếu |
| `api/schemas.py` | hợp đồng Pydantic; `TransactionInput` dựng từ `RAW_REQUIRED_COLUMNS` |
| `api/services/*.py`, `api/routes/*.py` | dịch vụ và route mỏng cho 15 endpoint |
| `tests/conftest.py` | cơ sở dữ liệu `fraud_test` dùng một lần, máy chủ thật trên nó |
| `tests/test_db.py`, `test_scoring.py`, `test_api.py` | 18 + 20 + 62 ca |
| `models/oof_scores.npz` | điểm out-of-fold cho `/threshold/optimize` — notebook 08 xuất thêm |

## 3. Số đối chiếu (máy 12 lõi, 2026-09-28, có tải nền khoảng 50% CPU)

| Mốc | Giá trị | Yêu cầu |
|---|---|---|
| `pg_isready` | `accepting connections`, healthy sau khoảng 10 giây | T-40 |
| `upgrade head → downgrade base → upgrade head` | lược đồ cuối trùng lược đồ đầu | T-42, TC-47 |
| Khởi động tới `/health` 200 | khoảng 5 giây | NFR-04 |
| `/score` một giao dịch, 1.000 lần | p95 45,6 ms phía máy chủ | NFR-01 < 50 ms |
| `/score/upload` 10.000 dòng | 2,0 giây | T-46, AC-A1 < 30 giây |
| `COPY` 10.000 dòng | 0,53 giây; `INSERT` từng dòng 119 dòng/giây | TC-54 < 3 giây |
| `/threshold/preview` | phần tính 0,12 ms, khoảng 15 ms qua HTTP | T-47 khoảng 10 ms |
| `/threshold/optimize` chi phí mặc định | 0,023172983899712563 = τ\* của `threshold.json` | ML-08 |
| `/threshold/optimize`, tối đa 200 cảnh báo/ngày | 0,9657, `constraint_binding = true` | T-47 |
| `/explain` | 5 dương + 3 âm ở mọi mẫu vượt ngưỡng được kiểm; 114/115 giao dịch vượt ngưỡng của tập kiểm thử có đủ 5 dương; SHAP khớp explainer tới từng bit | T-48, AC-A4 |
| Phát lại liên tục | 185 giây, 1.532 giao dịch, 0 lỗi | T-50, AC-A6 |

## 4. Kiểm tra AC-A6 — phát lại 3 phút trên máy chủ thật

`TestClient` không stream thật, nên AC-A6 kiểm bằng uvicorn và `httpx`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000      # cửa sổ 1
```

```python
# cửa sổ 2 — giữ kết nối 185 giây, đếm sự kiện và lỗi
import httpx, json, time
counts, started = {}, time.monotonic()
with httpx.stream("GET", "http://localhost:8000/api/v1/replay/stream", params={"speed": 60},
                  timeout=httpx.Timeout(10, read=30)) as r:
    name = None
    for line in r.iter_lines():
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            json.loads(line[6:])
            counts[name] = counts.get(name, 0) + 1
        if time.monotonic() - started > 185:
            break
print(counts)   # lần chạy 2026-09-28: start 1, transaction 1518, alert 14, stats 158
```

Sau đó `SELECT count(*) FROM transactions WHERE source = 'replay'` phải bằng số giao dịch đã phát
(mẻ cuối được ghi cả khi client ngắt giữa chừng), và log uvicorn không có dòng `ERROR`.

## 5. Những cái bẫy đã gặp

**Cổng 5432 đã bị chiếm.** Máy phát triển có sẵn PostgreSQL 17. Cách xử lý là đặt
`POSTGRES_HOST_PORT=5433` trong `.env`, không sửa `docker-compose.yml`, để máy khác vẫn dùng
5432 mặc định.

**`alembic.ini` có chữ tiếng Việt thì Alembic sập** (`UnicodeDecodeError … cp1252`). Alembic đọc tệp
bằng bảng mã mặc định của hệ điều hành. Chú thích trong tệp này viết không dấu. Alembic 1.20 cũng
đổi tên khóa `version_path_separator` thành `path_separator`.

**"Đúng 5 yếu tố dương" không phải lúc nào cũng có.** 77% giao dịch của tập kiểm thử có ít hơn 5
đặc trưng SHAP dương: với giao dịch hợp lệ điểm thấp, mô hình kéo gần hết đặc trưng về phía an
toàn. `/explain` trả **tối đa** 5 và 3, không mượn đóng góp âm để cho đủ. Trong hàng đợi, 114/115
giao dịch có đủ 5.

**12 luồng XGBoost cho một giao dịch làm chậm gấp đôi.** Mô hình huấn luyện với `n_jobs=-1`. Chấm
một dòng bằng 12 luồng tốn chi phí đồng bộ và tranh CPU: p95 của `/score` là 87 ms. Chấm bằng
1 luồng cho **cùng từng bit** trên 56.746 giao dịch (chỉ huấn luyện mới phụ thuộc số luồng), và p95
còn 45,6 ms. `api/loader.py` đặt `n_jobs=1` sau khi nạp.

**Biên 3τ là số thực máy.** `3 × 0,05 = 0,15000000000000002`, nên điểm 0,15 thuộc dải `high`. Trình
duyệt tính theo cùng chuẩn IEEE 754 nên hai bên vẫn khớp; ca kiểm thử dùng `3 * 0.05` làm biên.

**`TestClient` đọc hết thân phản hồi rồi mới trả.** Ca phát lại trong pytest phát giờ cuối của ngày
2 ở tốc độ 3.600. Ca 3 phút phải chạy trên uvicorn thật (§4).

**Trạng thái rò giữa các ca kiểm thử.** Một ca đổi `cost_fn` qua `PUT /threshold` làm đỏ ca xem
trước ngưỡng chạy sau nó. Fixture `clean_db` nay đưa cả bảng `settings` về trạng thái sau migration.
