# Lược đồ đầu vào cho mô hình đóng gói

`src/predict_xgboost.py` nhận CSV metadata đã được cấu trúc hóa, không nhận trực
tiếp JSON từ TMDb. Mỗi dòng là một phim.

## Trường bắt buộc

### Từ dữ liệu phim đã tiền xử lý

`log_budget`, `runtime`, `release_year`, `release_month`, `genre_count`,
`primary_genre`, `original_language`, `production_country_count`,
`primary_country`, `production_company_count`, `genres`,
`production_countries`, `production_companies`, `release_date`.

### Từ enrichment TMDb

`is_collection`, `collection_id`, `primary_company_id`, `company_count`,
`spoken_language_count`, `cast_count`, `crew_count`, `certification`,
`theatrical_country_count`, `release_event_count`, `overview_word_count`,
`has_tagline`, `keyword_count`.

`tmdb_id` là tùy chọn và chỉ được chép sang file kết quả để nhận diện phim.
CSV không được chứa biến mục tiêu hoặc các biến hậu phát hành dưới vai trò đặc
trưng dự báo.

## Giá trị thiếu

Giá trị thiếu được giữ dưới dạng ô trống hoặc NaN. Bộ điền khuyết biến số và bộ
mã hóa biến phân loại đã được khớp trong gói mô hình; không sử dụng thống kê của
dữ liệu cần dự đoán để tự điền giá trị.

## Giới hạn của đặc trưng lịch sử franchise

Gói mô hình chứa kho lịch sử tham chiếu của ảnh chụp dữ liệu huấn luyện. Khi suy
luận, bốn đặc trưng franchise chỉ sử dụng các phim tham chiếu có ngày phát hành
sớm hơn phim đầu vào. Kho lịch sử không tự cập nhật bằng các dòng khác trong
file đầu vào. Việc bổ sung phim mới vào kho lịch sử yêu cầu tái huấn luyện theo
đúng giao thức theo thời điểm.
