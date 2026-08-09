# Khung báo cáo học thuật tối đa 5 trang

Định dạng dự kiến: NeurIPS, hai cột; phần nội dung không quá 5 trang, không tính
tài liệu tham khảo nếu quy định môn học cho phép.

## 1. Giới thiệu — khoảng 0,6 trang

- Bối cảnh trực tiếp: thành công chỉ được xác nhận sau phát hành, trong khi một
  số siêu dữ liệu đã có trước thời điểm đó.
- Câu hỏi nghiên cứu và biến mục tiêu `revenue >= 2 × budget`.
- Đóng góp: quy trình chỉ sử dụng TMDb, EDA tập trung vào thông tin trước phát
  hành và mốc tham chiếu XGBoost được đánh giá theo giao thức chống rò rỉ.

## 2. Dữ liệu — khoảng 0,9 trang

- TMDb Official API; 2.597 phim từ 2000–2025, tối đa 100 phim phổ biến/năm.
- Điều kiện tạo tập dữ liệu mô hình hóa gồm 1.646 phim và phân bố hai lớp.
- Dữ liệu thiếu, quy tắc xử lý số 0 và giới hạn của chiến lược lấy mẫu.
- Hình `dataset_overview.png`, rộng một cột hoặc toàn chiều ngang khoảng
  0,30–0,35 trang.

## 3. Phương pháp — khoảng 1,2 trang

- Nhóm siêu dữ liệu A+B và bốn đặc trưng lịch sử franchise theo thời điểm.
- XGBoost, tiền xử lý trong quy trình và ngưỡng phân loại.
- Kiểm định chéo phân tầng lồng nhau 5×4, trong đó vòng ngoài chỉ phục vụ đánh
  giá; các quy tắc chống rò rỉ biến mục tiêu.
- Sơ đồ quy trình nhỏ, rộng một cột khoảng 0,20 trang nếu còn không gian.

## 4. Phân tích — khoảng 1,0 trang

- Mức độ liên hệ riêng lẻ giữa đặc trưng số và biến mục tiêu.
- Tỷ lệ thành công theo `primary_genre` và `is_collection`.
- Hình `pre_release_feature_associations.png`, rộng hai cột khoảng
  0,45–0,55 trang.
- Nhấn mạnh tương quan không đồng nghĩa nhân quả và số lượng nhóm không đều.

## 5. Kết quả và thảo luận — khoảng 1,3 trang

- Bảng Macro-F1, F1/recall từng lớp, balanced accuracy.
- So sánh A+B cố định với A+B + franchise history.
- Hình ma trận nhầm lẫn khoảng 0,20 trang; hình mức độ quan trọng của đặc trưng
  khoảng 0,25 trang nếu đủ chỗ.
- Trả lời câu hỏi nghiên cứu, phạm vi áp dụng, hạn chế và hướng phát triển.

Nguồn số liệu chính: `docs/eda_findings.md`, `docs/xgboost_results.md`, các bảng
tổng hợp trong `reports/tables/` và manifest trong `models/`.
