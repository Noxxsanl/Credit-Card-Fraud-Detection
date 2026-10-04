# Ảnh của dịch vụ web — bản HTML + Alpine.js (dự phòng). Không có bước build nào cần mạng.
#
# Chọn bản này thay cho bản Next.js:
#     WEB_UI=web docker compose up --build            # Git Bash
#     $env:WEB_UI="web"; docker compose up --build    # PowerShell
# (hoặc đặt WEB_UI=web trong .env). Build từ GỐC repo vì cần deploy/nginx.conf.

FROM nginx:stable-alpine
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY web/ /usr/share/nginx/html/

# Thay web/config.js: gọi API qua nginx của chính container này (deploy/nginx.conf)
RUN printf '%s\n' \
    '/* Sinh trong web/Dockerfile: API qua nginx cung origin, xem deploy/nginx.conf */' \
    'window.FRAUD_CONSOLE_CONFIG = { apiBase: "/api/v1" };' \
    > /usr/share/nginx/html/config.js

EXPOSE 80
