# Quyết định tiền xử lý dữ liệu phim

## Phạm vi

Nguồn dữ liệu chính thức của báo cáo và mô hình cuối chỉ là TMDb Official API.
Các trường IMDb của pipeline khảo sát cũ không được dùng trong EDA chính, tập
biến dự báo hoặc báo cáo cuối.

## Giá trị không hợp lệ và dữ liệu thiếu

- `budget`, `revenue` và `runtime` nhỏ hơn hoặc bằng 0 được chuyển thành giá trị
  thiếu trong dữ liệu dẫn xuất. Đối với ngân sách và doanh thu, giá trị 0 thường
  phản ánh việc chưa công bố hơn là giá trị thực bằng 0.
- Chuỗi rỗng và ký hiệu `\N` được xem là giá trị thiếu.
- Không điền trung bình hoặc trung vị trực tiếp vào bộ dữ liệu đã xử lý. Việc
  điền thiếu dùng cho mô hình được thực hiện trong pipeline và chỉ học từ phần
  dữ liệu huấn luyện của từng fold.
- Không loại một phim chỉ vì thiếu biến không bắt buộc đối với mô hình.

## Ngoại lai

Hai cờ `budget_outlier` và `revenue_outlier` được xác định bằng quy tắc IQR:
nhỏ hơn `Q1 - 1,5 × IQR` hoặc lớn hơn `Q3 + 1,5 × IQR`. Ngoại lai chỉ được đánh
dấu, không bị xóa hoặc thay thế, vì chúng có thể đại diện cho phim có quy mô tài
chính thực sự khác biệt.

## Đặc trưng dẫn xuất phục vụ phân tích

- Thời gian: `release_year`, `release_month`, `release_quarter`.
- Tài chính: `profit`, `revenue_to_budget`, `roi`, `log_budget`, `log_revenue`.
- Phản hồi TMDb: `log_popularity`, `log_vote_count`.
- Danh mục: `genre_count`, `primary_genre`, `production_country_count`,
  `primary_country`, `production_company_count`.
- Cờ dữ liệu: `budget_available`, `revenue_available`, `budget_outlier`,
  `revenue_outlier`.

Các đặc trưng hậu phát hành trong danh sách trên chỉ phục vụ mô tả hoặc tạo
nhãn; chúng không được dùng làm đầu vào của mô hình dự đoán trước phát hành.

## Định nghĩa biến mục tiêu

Với phim có ngân sách và doanh thu hợp lệ:

```text
is_successful = 1 nếu revenue_to_budget >= 2
is_successful = 0 nếu revenue_to_budget < 2
```

Không tính nhãn khi thiếu ngân sách hoặc doanh thu. Đây là định nghĩa vận hành
đơn giản, chưa phản ánh chi phí marketing, phần doanh thu giữ lại của rạp, lạm
phát hoặc các nguồn thu ngoài phòng vé.

## Kiểm soát rò rỉ mục tiêu

Trong mô hình phân loại, `revenue`, `log_revenue`, `profit`, `roi` và
`revenue_to_budget` không được dùng làm biến đầu vào vì trực tiếp chứa hoặc
được suy ra từ kết quả tài chính sau phát hành. `is_successful` chỉ được dùng
làm biến mục tiêu. Các biến `popularity`, điểm và số lượt bình chọn cũng bị
loại vì không bảo đảm có sẵn trước phát hành.
