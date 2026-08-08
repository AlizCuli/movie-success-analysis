# Cấu trúc và chính sách dữ liệu

## Cấu trúc thư mục

- `raw/`: dữ liệu gốc được thu thập trực tiếp và không chỉnh sửa.
- `external/`: dữ liệu phụ trợ từ nguồn bên ngoài.
- `interim/`: dữ liệu trung gian sau một hoặc nhiều bước xử lý.
- `processed/`: dữ liệu đã được xử lý để phục vụ EDA và mô hình.

## Chính sách phân phối

Các tệp dữ liệu trong bốn thư mục trên được giữ trên máy cá nhân và **không
được phân phối qua repository công khai**. GitHub chỉ lưu `data/README.md` và
các tệp `.gitkeep` để bảo toàn cấu trúc thư mục.

Quy định áp dụng:

- Dữ liệu TMDb phải được người dùng tự thu thập bằng các script trong `src/` và
  API Read Access Token của chính mình.
- Token và tệp `.env` chỉ được lưu trên máy cá nhân, không được commit vào Git.
- Các artifact IMDb của pipeline khảo sát cũ không được phân phối lại.
- Không tạo dữ liệu giả để thay thế các tệp không được công khai.

Việc không phân phối dữ liệu không ảnh hưởng đến các tệp đang có trên máy. Khi
clone repository sang máy khác, người dùng cần chạy lại pipeline thu thập và
tiền xử lý trước khi tái lập EDA hoặc huấn luyện mô hình.

Nguồn dữ liệu và yêu cầu ghi nhận được trình bày tại
`docs/data_sources.md`.
