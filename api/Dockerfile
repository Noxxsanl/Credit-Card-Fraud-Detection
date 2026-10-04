# Ảnh của dịch vụ api — FastAPI + mô hình (docs/03 §6.3).
#
# Build từ GỐC repo, vì API dùng chung src/ với notebook (ranh giới chống lệch train/serve):
#     docker compose build api            # docker-compose.yml đặt context: . và dockerfile: api/Dockerfile
#     docker build -f api/Dockerfile .    # tương đương, không qua Compose
# .dockerignore ở gốc chỉ cho api/, src/, alembic.ini (và hai thư mục giao diện) vào ngữ cảnh build.
#
# Hiện vật models/ và data/ KHÔNG nằm trong ảnh: Compose gắn chúng dạng chỉ đọc, nên chạy lại
# notebook 08 chỉ cần `docker compose restart api`, không phải build lại.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # shap biên dịch hàm bằng numba và muốn ghi bộ đệm cạnh mã nguồn — thư mục chỉ đọc với
    # người dùng không phải root, nên chỉ sang /tmp
    NUMBA_CACHE_DIR=/tmp/numba-cache \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Lớp thư viện tách riêng: sửa mã nguồn không phải cài lại
COPY api/requirements.txt api/requirements-nodeps.txt api/
# compileall: ảnh python chính thức xóa mọi .pyc của thư viện chuẩn cho nhẹ, còn
# PYTHONDONTWRITEBYTECODE thì cấm ghi lại — nên không biên dịch sẵn ở đây, mỗi lần khởi động
# container lại biên dịch từ mã nguồn, import chậm thêm vài giây (NFR-04)
RUN pip install -r api/requirements.txt \
 && pip install --no-deps -r api/requirements-nodeps.txt \
 && python -m compileall -q -j 0 /usr/local/lib/python3.12

COPY alembic.ini ./
COPY src/ src/
COPY api/ api/
RUN python -m compileall -q src api

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
CMD ["python", "-m", "api.entrypoint"]
