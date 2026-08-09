# Quyết định tiền xử lý

## Phạm vi

Đầu vào duy nhất là CSV cấu trúc hóa từ TMDb:
`data/interim/tmdb_movies_2000_2025.csv`. File nguồn không bị chỉnh sửa.

## Giá trị thiếu và không hợp lệ

- Chuỗi rỗng và `\N` được chuyển thành giá trị thiếu.
- `budget <= 0`, `revenue <= 0` và `runtime <= 0` được xem là chưa công bố hoặc
  không hợp lệ và chuyển thành thiếu.
- Điểm đánh giá ngoài khoảng 0–10 được chuyển thành giá trị thiếu để bảo đảm
  tính hợp lệ của dữ liệu thô; trường này không tham gia EDA chính hoặc mô hình
  cuối.
- Không điền trung bình/trung vị trực tiếp vào CSV đã xử lý.
- Với các đặc trưng dự báo còn thiếu, bộ điền khuyết nằm trong quy trình mô hình
  và chỉ học từ phân hoạch huấn luyện của từng vòng kiểm định.

`movies_cleaned.csv` giữ toàn bộ 2.597 phim. `movies_modeling.csv` chỉ giữ các
phim có `budget`, `revenue`, `runtime`, `release_date` và biến mục tiêu hợp lệ;
các đặc trưng khác có thể còn thiếu và được xử lý trong quy trình mô hình.

## Đặc trưng phục vụ EDA

- Thời gian: `release_year`, `release_month`, `release_quarter`.
- Tài chính mô tả: `profit`, `revenue_to_budget`, `roi`, `log_budget`,
  `log_revenue`.
- Danh mục: `genre_count`, `primary_genre`, `production_country_count`,
  `primary_country`, `production_company_count`.
- Cờ chất lượng: `budget_available`, `revenue_available`, `budget_outlier`,
  `revenue_outlier`.

Các trường tài chính hậu phát hành chỉ phục vụ tạo nhãn hoặc mô tả dữ liệu,
không được sử dụng làm đặc trưng dự báo.

## Ngoại lai

`budget_outlier` và `revenue_outlier` dùng quy tắc IQR: ngoài khoảng
`[Q1 - 1,5 × IQR; Q3 + 1,5 × IQR]`. Ngoại lai chỉ được đánh dấu, không xóa,
vì có thể là các phim có quy mô thực sự khác biệt.

## Biến mục tiêu

Với phim có budget và revenue hợp lệ:

```text
is_successful = 1 nếu revenue >= 2 × budget
is_successful = 0 nếu revenue < 2 × budget
```

Đây là tiêu chuẩn thành công tài chính được xác lập trong phạm vi nghiên cứu.

## Chống rò rỉ biến mục tiêu

Các cột `revenue`, `log_revenue`, `profit`, `roi`, `revenue_to_budget` không
được sử dụng làm đặc trưng dự báo. `is_successful` chỉ đóng vai trò biến mục
tiêu. `popularity`, điểm đánh giá và lượt bình chọn cũng bị loại vì không bảo
đảm có sẵn trước thời điểm phát hành.
