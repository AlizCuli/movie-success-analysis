# Khung báo cáo học thuật

## 1. Giới thiệu

- Trình bày trực tiếp bài toán dự đoán thành công tài chính trước phát hành.
- Nêu câu hỏi nghiên cứu và cách mô hình hóa thành bài toán phân loại nhị phân.
- Giới thiệu định nghĩa vận hành: phim thành công khi `revenue >= 2 × budget`.

Nguồn nội dung: `README.md` và `PROJECT_STATUS.md`.

## 2. Dữ liệu

- Nguồn chính thức duy nhất: TMDb Official API.
- Quy mô: 2.597 phim giai đoạn 2000–2025; 1.646 phim đủ điều kiện đánh giá.
- Trình bày chiến lược lấy mẫu, các nhóm biến, dữ liệu thiếu, ngoại lai và phân
  bố nhãn.
- Nêu rõ dữ liệu không được phân phối qua repository công khai.

Nguồn nội dung: `docs/data_sources.md`, `docs/preprocessing_decisions.md` và
`reports/tables/tmdb_enrichment_audit.csv`.

## 3. Phương pháp

- Mô tả các nhóm đặc trưng A+B và lịch sử franchise theo thời điểm.
- Trình bày XGBoost và nested Stratified Cross-Validation 5×4.
- Giải thích cách chọn ngưỡng trong inner OOF và quy tắc chống rò rỉ dữ liệu.
- Liệt kê các biến hậu phát hành bị loại khỏi tập đầu vào.

Nguồn nội dung: `docs/pre_release_operational_scope.md` và
`docs/operational_franchise_history_findings.md`.

## 4. Phân tích dữ liệu

- Phân tích mức độ thiếu dữ liệu và phân phối ngân sách/doanh thu.
- Trình bày quan hệ mô tả giữa ngân sách, doanh thu, thể loại và thời điểm phát
  hành, đồng thời tránh diễn giải quan hệ tương quan như quan hệ nhân quả.
- Phân tích mất cân bằng lớp và các quan sát gần ngưỡng tạo nhãn.
- Không đưa rating, vote hoặc popularity vào nội dung phân tích chính của báo
  cáo cuối.

Nguồn nội dung: `notebooks/01_eda.ipynb`, `docs/eda_findings.md`, các bảng EDA
và các hình chọn lọc `01`–`07`.

## 5. Kết quả, thảo luận và kết luận

- Báo cáo Macro-F1 `0,719483`, metric từng lớp và balanced accuracy.
- Trình bày confusion matrix, độ ổn định giữa các outer fold, phân tích lỗi và
  feature importance.
- Thảo luận mức độ dự đoán, hiệu năng thấp hơn ở lớp không thành công và các
  giới hạn do cách lấy mẫu, định nghĩa nhãn và snapshot TMDb.
- Kết luận trong phạm vi dữ liệu hiện có; đề xuất dữ liệu point-in-time tốt hơn
  và temporal holdout cho nghiên cứu tiếp theo.

Nguồn nội dung: `docs/xgboost_results.md`, các bảng benchmark và hình `09`–`12`.
