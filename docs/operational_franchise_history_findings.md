# Lịch sử franchise theo thời điểm

## Giả thuyết

Các phần phim đã phát hành trước trong cùng collection có thể cung cấp tín hiệu
về mức độ quen thuộc của thương hiệu và lịch sử quy mô dự án. Thí nghiệm chỉ bổ
sung history vào feature A+B hiện hành, không thay đổi mẫu, target hoặc outer
split.

## Bốn feature lịch sử

- `collection_prior_movie_count`;
- `collection_prior_success_rate` có smoothing;
- `collection_prior_mean_log_budget`;
- `collection_years_since_previous`.

History builder được fit riêng trong training partition. Chỉ phim có ngày phát
hành sớm hơn phim cần dự đoán được dùng; validation/test không cập nhật kho lịch
sử. Revenue của phim mục tiêu không bao giờ là predictor.

## Kết quả pooled outer-OOF

| Cấu hình | Macro-F1 | F1 lớp 0 | Recall lớp 0 | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| A+B cố định | 0,710977 | 0,597194 | **0,624738** | 0,716988 |
| A+B + franchise history | **0,719483** | **0,605128** | 0,618449 | **0,722398** |

Franchise history tăng Macro-F1 0,008505 và F1 lớp 0 0,007934, trong khi recall
lớp 0 giảm nhẹ 0,006289. Cải thiện là nhỏ nhưng xuất hiện ở metric chính và
balanced accuracy; vì vậy cấu hình này được chọn làm benchmark, không được diễn
giải như bằng chứng nhân quả về tác động của franchise.

Các kết quả công khai là bảng tổng hợp trong `reports/tables/`; dự đoán và outer
fold assignment cấp từng phim chỉ lưu local.
