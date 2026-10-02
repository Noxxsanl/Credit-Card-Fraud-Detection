# Thư viện bên thứ ba của giao diện

Hai tệp dưới đây được **chép thẳng vào repo** thay vì nạp từ CDN: giao diện phải chạy được
khi không có mạng (NFR-08) — buổi bảo vệ không được phụ thuộc Wi-Fi của phòng. Không có
`node_modules`, không có bước đóng gói (docs/03 §6.2).

| Tệp | Thư viện | Phiên bản | Giấy phép | Nguồn |
|---|---|---|---|---|
| `alpine.min.js` | Alpine.js | 3.17.4 | MIT | `https://cdn.jsdelivr.net/npm/alpinejs@3.17.4/dist/cdn.min.js` |
| `chart.umd.min.js` | Chart.js | 4.5.1 | MIT | `https://cdn.jsdelivr.net/npm/chart.js@4.5.1/dist/chart.umd.min.js` |

SHA-256 để kiểm tệp chưa bị sửa:

```
232519394c6c8fdba6f362b1d9da16106db513cdbf899011f00daab4051df31c  alpine.min.js
48444a82d4edcb5bec0f1965faacdde18d9c17db3063d042abada2f705c9f54a  chart.umd.min.js
```

Nâng phiên bản: tải tệp mới cùng đường dẫn trên (đổi số phiên bản), cập nhật bảng và mã băm,
rồi mở cả bốn màn hình kiểm không có lỗi trong console. Không sửa tay nội dung hai tệp này.
