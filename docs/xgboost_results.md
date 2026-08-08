# Kết quả XGBoost pre-release operational

## Benchmark

Benchmark chính thức được đánh giá bằng nested Stratified CV 5 outer fold × 4
inner fold trên 1.646 phim. Pooled outer-OOF:

| Chỉ số | Giá trị |
| --- | ---: |
| Macro-F1 | **0,719483** |
| F1 lớp 0 | 0,605128 |
| Recall lớp 0 | 0,618449 |
| Balanced accuracy | 0,722398 |

Tập kiểm định outer không tham gia chọn đặc trưng, tiền xử lý, ngưỡng phân loại
hoặc số vòng lặp. Các dự đoán dùng để tính metric nằm tại
`reports/tables/operational_franchise_oof_predictions.csv`.

## Tệp kết quả phục vụ báo cáo

- `xgboost_fold_metrics.csv`: độ ổn định giữa năm outer fold.
- `xgboost_confusion_matrix.csv`: số dự đoán đúng/sai từng lớp.
- `xgboost_error_analysis.csv`: false positive, false negative và khoảng cách
  tới ngưỡng `revenue_to_budget = 2`.
- `xgboost_feature_importance.csv`: độ quan trọng của đặc trưng từ mô hình cuối;
  chỉ dùng để diễn giải mô hình, không phải độ quan trọng ở từng outer fold.
- Hình `09`–`12` trong `reports/figures/` minh họa các kết quả trên.

## Mô hình đóng gói

Mô hình trong `models/` được huấn luyện trên toàn bộ 1.646 phim sau khi cấu hình
đã được khóa. Ngưỡng và số vòng cuối được chọn bằng OOF 4 fold trên toàn bộ dữ
liệu huấn luyện, không tinh chỉnh lại siêu tham số. Vì mô hình đã thấy toàn bộ
dữ liệu, điểm trên tập huấn luyện không được xem là hiệu năng tổng quát hóa.
Hiệu năng chính thức vẫn là Macro-F1 outer-OOF `0,719483`.

## Giới hạn

- Lớp không thành công chỉ chiếm 28,98%, khiến F1 lớp 0 thấp hơn lớp 1.
- Nhãn gần ngưỡng `revenue_to_budget = 2` có tính mơ hồ cao.
- Metadata TMDb là snapshot hiện tại, không phải archive point-in-time.
- Mẫu tối đa 100 phim phổ biến mỗi năm không đại diện cho toàn bộ thị trường.
