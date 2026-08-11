# Báo cáo và hình minh họa

Thư mục này chứa các bảng tổng hợp và chín hình được chọn cho báo cáo cuối
cùng. Các tệp PNG được lưu sẵn để người dùng clone repository có thể xem ngay.

## Tái tạo chín hình

Sau khi có các dữ liệu cấp phim ở máy cục bộ, chạy từ thư mục gốc:

```powershell
& ".\.venv\Scripts\python.exe" src\generate_report_figures.py
```

Lệnh trên gọi lần lượt EDA TMDb, bản đồ metadata đầu vào, phân tích độ rộng
phát hành rạp và các hình đánh giá XGBoost. Chương trình kiểm tra schema tổng
hợp và xác nhận đủ chín tệp đầu ra trước khi kết thúc.

Các đầu vào cấp phim cần có tại máy cục bộ:

- `data/processed/movies_cleaned.csv`
- `data/processed/movies_modeling.csv`
- `data/raw/tmdb_movie_enrichment.jsonl`

Repository công khai không phân phối các tệp dữ liệu cấp phim; vì vậy một bản
clone mới có thể xem PNG đã lưu nhưng cần tái tạo dữ liệu theo hướng dẫn thu
thập và tiền xử lý trước khi chạy lại lệnh trên. Không cần gọi lại API chỉ để
xem các hình đã có.

## Bộ hình hiện hành

- `dataset_overview.png`: quy mô tập dữ liệu, thiếu dữ liệu và phân bố nhãn.
- `pre_release_spearman_heatmap.png`: tương quan Spearman của các biến số dùng
  trong phân tích mô tả.
- `success_by_genre_collection.png`: tỷ lệ `is_successful` theo thể loại và
  trạng thái collection.
- `tmdb_raw_feature_map.png`: mười lăm trường metadata TMDb trước feature
  engineering.
- `Tỷ lệ thành công tài chính theo số quốc gia phát hành rạp..png`: tỷ lệ thành
  công theo bốn nhóm độ rộng phát hành rạp.
- `ti le.png`: tương tác giữa độ rộng phát hành rạp và collection.
- `xgboost_performance_summary.png`: các chỉ số outer-OOF chính của XGBoost.
- `xgboost_confusion_matrix.png`: ma trận nhầm lẫn outer-OOF.
- `xgboost_fold_macro_f1.png`: Macro-F1 theo năm outer fold.

Các bảng `reports/tables/` là artifact tổng hợp phục vụ tái tạo những hình
đánh giá mô hình. Dữ liệu dòng-level, token và môi trường ảo luôn bị loại khỏi
Git theo chính sách của project.
