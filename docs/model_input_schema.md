# Schema đầu vào cho model đóng gói

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
CSV không được chứa target hoặc các biến hậu phát hành để làm predictor.

## Giá trị thiếu

Giữ giá trị thiếu ở dạng ô trống/NaN. Numeric imputer và categorical encoder đã
được fit trong bundle; không tự điền bằng thống kê của dữ liệu cần dự đoán.

## Giới hạn franchise history

Bundle chứa reference history của snapshot huấn luyện. Khi dự đoán, bốn feature
franchise chỉ dùng phim trong reference có ngày phát hành sớm hơn phim đầu vào.
Bundle không tự cập nhật bằng các dòng khác trong file input. Muốn cập nhật kho
history bằng phim mới, cần tái huấn luyện theo đúng protocol point-in-time.

