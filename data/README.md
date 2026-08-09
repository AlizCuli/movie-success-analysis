# Dữ liệu cục bộ

Repository không phân phối dữ liệu TMDb gốc hoặc dữ liệu dẫn xuất. Việc tái tạo
dữ liệu yêu cầu một TMDb API Read Access Token hợp lệ.

## Cấu trúc

- `raw/`: phản hồi JSON/JSONL từ TMDb và điểm kiểm tra thu thập.
- `external/`: giữ chỗ cho nguồn phụ trợ; không được dùng trong phạm vi cuối.
- `interim/`: CSV cấu trúc hóa trực tiếp từ dữ liệu TMDb.
- `processed/`: dữ liệu làm sạch phục vụ EDA và mô hình hóa.

## Các file chính do quy trình tạo ra

```text
data/raw/tmdb_movies_2000_2025.json
data/raw/tmdb_movie_enrichment.jsonl
data/interim/tmdb_movies_2000_2025.csv
data/processed/movies_cleaned.csv
data/processed/movies_modeling.csv
```

Mọi nội dung trong các thư mục trên, ngoại trừ `.gitkeep`, đều bị `.gitignore`
loại khỏi Git. Token không được lưu trong `data/` và chỉ được khai báo trong
file `.env` cục bộ.
