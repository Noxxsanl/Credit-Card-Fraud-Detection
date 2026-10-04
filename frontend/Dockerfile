# Ảnh của dịch vụ web — bản Next.js (mặc định của docker-compose.yml).
#
# Hai tầng: Node build ra thư mục tĩnh out/, rồi chỉ nginx phục vụ — ảnh chạy không có Node.
# Build từ GỐC repo (cần deploy/nginx.conf):
#     docker compose build web
#     docker build -f frontend/Dockerfile .
#
# Gọi API qua đường dẫn tương đối /api/v1: nginx chuyển tiếp sang dịch vụ api (deploy/nginx.conf).
# Build trên Linux nên bước làm phẳng tên tệp của scripts/fix-segment-files.mjs không có việc gì làm.

FROM node:22-alpine AS build
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
ENV NEXT_PUBLIC_API_BASE=/api/v1
RUN npm run build

FROM nginx:stable-alpine
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/out /usr/share/nginx/html
EXPOSE 80
