# Báo cáo tổng hợp

Repository công khai chỉ theo dõi các kết quả tổng hợp, không công khai bản ghi
hoặc dự đoán cấp từng phim.

## Hình

- `dataset_overview.png`: tỷ lệ thiếu ở các trường cốt lõi và phân bố biến mục
  tiêu.
- `pre_release_feature_associations.png`: tương quan Spearman của đặc trưng số
  và tỷ lệ thành công theo thể loại/trạng thái collection.
- `xgboost_confusion_matrix.png`: ma trận nhầm lẫn gộp từ dự đoán ngoài mẫu.
- `xgboost_fold_macro_f1.png`: Macro-F1 của năm vòng kiểm định ngoài.
- `xgboost_feature_importance.png`: mức độ quan trọng của đặc trưng trong mô
  hình cuối.

## Bảng

- Nhóm `dataset_*`, `core_missingness`, `pre_release_spearman`,
  `success_by_*`, `yearly_*`: kết quả EDA tổng hợp.
- Nhóm `operational_ab_fixed_*`: cấu hình đối chứng XGBoost A+B.
- Nhóm `operational_franchise_*`: cấu hình tham chiếu A+B kết hợp lịch sử
  franchise.
- Nhóm `xgboost_*`: chỉ số, ma trận nhầm lẫn, mức độ quan trọng của đặc trưng
  và thông tin đóng gói.

Các file `*_oof_predictions.csv`, `*_outer_fold_assignments.csv` và dữ liệu
phân tích lỗi từng phim được tạo cục bộ khi đánh giá nhưng bị `.gitignore` loại
khỏi repository công khai.
