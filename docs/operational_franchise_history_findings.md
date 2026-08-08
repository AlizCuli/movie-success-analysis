# Thí nghiệm franchise history theo thời điểm

> **Trạng thái:** Đây là benchmark XGBoost chính thức của project, với
> Macro-F1 outer OOF **0,719483**.

## Mục tiêu và phạm vi

Thí nghiệm này phát triển riêng nhánh `pre_release_operational`; không thay
thế hay ghi đè các mô hình strict pre-release. Mục tiêu là bổ sung tín hiệu về
lịch sử của franchise/collection mà một phim có thể thuộc về.

Mỗi đặc trưng chỉ dùng các phim có ngày phát hành **sớm hơn** phim đang dự
đoán. Các phim cùng ngày không được dùng làm lịch sử của nhau. Trong từng
outer/inner fold, history builder chỉ được fit bằng phần training của fold đó.

## Đặc trưng mới

- `collection_prior_movie_count`: số phim trước đó trong collection.
- `collection_prior_success_rate`: tỷ lệ thành công trước đó đã smoothing.
- `collection_prior_mean_log_budget`: trung bình `log_budget` của các phim
  trước đó trong collection.
- `collection_years_since_previous`: số năm từ phim collection trước đó.

Trong 1.646 phim modeling, có 709 phim có collection; 518 phim thuộc
collection có ít nhất hai phim trong mẫu. Không đưa `revenue`, `profit`,
`roi`, `revenue_to_budget`, target, rating, vote, popularity hay IMDb vào
predictor.

## Giao thức đánh giá

- Giữ nguyên 1.646 ID, target và outer split 5 fold (seed 42) của baseline.
- Dùng inner stratified CV 4 fold (seed 43) để chọn số vòng XGBoost và
  threshold; outer validation không tham gia tuning.
- Dùng đúng pipeline imputer, encoder và class weight của baseline A+B.

## Kết quả pooled outer OOF

| Cấu hình | Macro-F1 | F1 lớp 0 | Recall lớp 0 | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| A+B cố định | 0.710977 | 0.597194 | 0.624738 | 0.716988 |
| A+B + franchise history | **0.719483** | **0.605128** | 0.618449 | **0.722398** |

Franchise history tăng Macro-F1 **0.008505** so với đối chứng A+B cố định và
cao hơn mốc pre-release-like lịch sử 0.7144 khoảng 0.0051. Recall lớp 0 giảm
nhẹ so với A+B cố định, nhưng F1 lớp 0 và balanced accuracy đều tăng; vì vậy
đây là một cải thiện cân bằng hơn, không chỉ do thay đổi threshold.

Các artifact tái lập nằm ở `reports/tables/operational_franchise_*.csv`.
Mô hình, artifact và kết quả cũ vẫn được giữ nguyên.
