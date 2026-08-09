# Dữ liệu local

Repository không phân phối dữ liệu TMDb gốc hoặc dẫn xuất. Người dùng tự tái tạo
dữ liệu bằng TMDb API Read Access Token của mình.

## Cấu trúc

- `raw/`: JSON/JSONL phản hồi từ TMDb và checkpoint thu thập.
- `external/`: giữ chỗ cho nguồn phụ trợ; không được dùng trong phạm vi cuối.
- `interim/`: CSV cấu trúc hóa trực tiếp từ dữ liệu TMDb.
- `processed/`: dữ liệu làm sạch cho EDA và modeling.

## File chính được pipeline tạo

```text
data/raw/tmdb_movies_2000_2025.json
data/raw/tmdb_movie_enrichment.jsonl
data/interim/tmdb_movies_2000_2025.csv
data/processed/movies_cleaned.csv
data/processed/movies_modeling.csv
```

Mọi nội dung trong các thư mục trên, ngoại trừ `.gitkeep`, đều bị `.gitignore`
loại khỏi Git. Không đặt token trong `data/`; token chỉ nằm trong `.env` local.
