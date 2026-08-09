# Báo cáo tổng hợp

Repository public chỉ theo dõi artifact tổng hợp, không công khai bản ghi hoặc
dự đoán cấp từng phim.

## Figures

- `dataset_overview.png`: tỷ lệ thiếu ở trường cốt lõi và phân bố target.
- `pre_release_feature_associations.png`: tương quan Spearman của predictor số
  và tỷ lệ thành công theo genre/collection.
- `xgboost_confusion_matrix.png`: ma trận nhầm lẫn pooled outer-OOF.
- `xgboost_fold_macro_f1.png`: Macro-F1 của năm outer fold.
- `xgboost_feature_importance.png`: importance của model fit cuối.

## Tables

- Nhóm `dataset_*`, `core_missingness`, `pre_release_spearman`,
  `success_by_*`, `yearly_*`: kết quả EDA tổng hợp.
- Nhóm `operational_ab_fixed_*`: control XGBoost A+B.
- Nhóm `operational_franchise_*`: benchmark A+B + franchise history.
- Nhóm `xgboost_*`: metric, confusion matrix, importance và thông tin đóng gói.

Các file `*_oof_predictions.csv`, `*_outer_fold_assignments.csv` và dữ liệu
phân tích lỗi từng phim được tạo local khi đánh giá nhưng bị `.gitignore` loại
khỏi repository public.
