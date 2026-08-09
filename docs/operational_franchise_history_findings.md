# Lịch sử franchise theo thời điểm

## Giả thuyết

Các phần phim đã phát hành trước trong cùng collection có thể cung cấp tín hiệu
về mức độ quen thuộc của thương hiệu và lịch sử quy mô dự án. Thí nghiệm chỉ bổ
sung đặc trưng lịch sử vào tập đặc trưng A+B, không thay đổi mẫu, biến mục tiêu
hoặc phân hoạch vòng ngoài.

## Bốn đặc trưng lịch sử

- `collection_prior_movie_count`;
- `collection_prior_success_rate` có làm trơn;
- `collection_prior_mean_log_budget`;
- `collection_years_since_previous`.

Bộ kiến tạo lịch sử được khớp riêng trong từng phân hoạch huấn luyện. Chỉ phim
có ngày phát hành sớm hơn phim cần dự đoán được sử dụng; phân hoạch xác thực và
kiểm thử không cập nhật kho lịch sử. `revenue` của phim mục tiêu không được sử
dụng làm đặc trưng dự báo.

## Kết quả ngoài mẫu gộp

| Cấu hình | Macro-F1 | F1 lớp 0 | Recall lớp 0 | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| A+B cố định | 0,710977 | 0,597194 | **0,624738** | 0,716988 |
| A+B + franchise history | **0,719483** | **0,605128** | 0,618449 | **0,722398** |

Franchise history tăng Macro-F1 0,008505 và F1 lớp 0 0,007934, trong khi recall
lớp 0 giảm nhẹ 0,006289. Cải thiện là nhỏ nhưng xuất hiện ở metric chính và
giải như bằng chứng nhân quả về tác động của franchise.
balanced accuracy; vì vậy cấu hình này được chọn làm mốc tham chiếu. Chênh lệch
không được diễn giải như bằng chứng nhân quả về tác động của franchise.
giải như bằng chứng nhân quả về tác động của franchise.

Các kết quả công khai là bảng tổng hợp trong `reports/tables/`; dự đoán và phân
hoạch vòng ngoài cấp từng phim chỉ được lưu cục bộ.
