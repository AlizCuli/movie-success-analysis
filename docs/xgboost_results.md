# Kết quả XGBoost pre-release operational

## Benchmark ngoài mẫu

Benchmark được đánh giá bằng nested Stratified CV 5 outer × 4 inner fold trên
1.646 phim. Kết quả pooled outer-OOF:

| Chỉ số | Giá trị |
| --- | ---: |
| Macro-F1 | **0,719483** |
| F1 lớp 0 | 0,605128 |
| F1 lớp 1 | 0,833837 |
| Recall lớp 0 | 0,618449 |
| Recall lớp 1 | 0,826347 |
| Balanced accuracy | 0,722398 |
| Accuracy | 0,766100 |

Outer folds không tham gia chọn feature, preprocessing, số vòng boosting hoặc
threshold. Dự đoán cấp từng phim được giữ local; repository chỉ công khai bảng
tổng hợp.

## Artifact báo cáo

- `xgboost_fold_metrics.csv`: độ ổn định giữa năm outer fold.
- `xgboost_confusion_matrix.csv`: ma trận nhầm lẫn tổng hợp.
- `xgboost_pooled_metrics.csv`: metric pooled outer-OOF.
- `xgboost_feature_importance.csv`: importance của model fit cuối, dùng để diễn
  giải model chứ không thay thế permutation/SHAP ngoài mẫu.
- Ba hình cùng tên trong `reports/figures/`: confusion matrix, Macro-F1 theo
  fold và top feature importance.

## Model đóng gói

Model trong `models/` được fit trên toàn bộ 1.646 phim sau khi cấu hình đã khóa:
144 boosting rounds và threshold 0,51. Vì model này đã thấy toàn bộ tập dữ liệu,
không sử dụng điểm fit của nó để báo cáo khả năng tổng quát hóa; benchmark chính
thức vẫn là kết quả nested outer-OOF ở trên.

## Diễn giải và giới hạn

- Model nhận diện lớp thành công tốt hơn lớp không thành công; F1 lớp 0 còn là
  giới hạn chính.
- Franchise history cải thiện Macro-F1 so với A+B cố định từ 0,710977 lên
  0,719483, nhưng chênh lệch nhỏ và cần được diễn giải thận trọng.
- Kết quả chịu ảnh hưởng bởi chiến lược lấy tối đa 100 phim phổ biến mỗi năm,
  dữ liệu tài chính thiếu và tính chất snapshot của TMDb.
- Macro-F1 0,719483 cho thấy metadata trước phát hành chứa tín hiệu dự báo có
  ích, nhưng chưa đủ cho quyết định tài chính có rủi ro cao nếu dùng độc lập.
