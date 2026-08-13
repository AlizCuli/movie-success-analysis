# Trạng thái dự án

## Phạm vi nghiên cứu cuối cùng

- Đề tài: dự đoán khả năng thành công tài chính của phim trước phát hành.
- Nguồn dữ liệu: TMDb Official API.
- Biến mục tiêu: `revenue >= 2 × budget`.
- Mô hình: XGBoost trong phạm vi trước phát hành theo quy ước vận hành.
- Phương pháp đánh giá: kiểm định chéo phân tầng lồng nhau, gồm 5 vòng ngoài và
  4 vòng trong.
- Chỉ số chính: Macro-F1.

## Dữ liệu

- 2.597 phim TMDb giai đoạn 2000–2025.
- 1.646 phim đáp ứng điều kiện của tập dữ liệu mô hình hóa.
- Lớp 0: 477 phim; lớp 1: 1.169 phim.
- Dữ liệu gốc, trung gian và đã xử lý chỉ được lưu cục bộ và bị Git bỏ qua.
- Repository công khai không chứa dữ liệu hoặc dự đoán cấp từng phim.

## Mốc tham chiếu được bảo toàn

| Chỉ số | Giá trị |
| --- | ---: |
| Macro-F1 trên dự đoán ngoài mẫu gộp | **0.719483** |
| F1 lớp 0 | 0.605128 |
| Recall lớp 0 | 0.618449 |
| Balanced accuracy | 0.722398 |
| Accuracy | 0.766100 |

Mô hình cuối đã được đóng gói trong `models/` với 144 cây, ngưỡng phân loại
0,51, 51 đặc trưng sau bước kiến tạo và 160 đặc trưng sau tiền xử lý.

## Thành phần hoàn thiện

- Quy trình thu thập, tiền xử lý, làm giàu dữ liệu, EDA, đánh giá và huấn luyện.
- Notebook EDA chỉ sử dụng TMDb và không nhúng kết quả đầu ra dung lượng lớn.
- Ba hình EDA chính và ba hình đánh giá XGBoost.
- Gói mô hình, mô hình XGBoost nguyên bản và manifest chứa checksum.
- README trình bày cấu trúc dự án, quy trình và hướng dẫn tái lập.
- Bộ kiểm tra giao ước không phụ thuộc dữ liệu cục bộ.

## Giới hạn phải nêu trong báo cáo

- Việc lấy tối đa 100 phim phổ biến mỗi năm có thể tạo sai lệch chọn mẫu.
- TMDb hiện tại không phải kho lưu trữ theo thời điểm của mọi trường siêu dữ
  liệu.
- Nhãn là tiêu chuẩn phân loại được xác lập trong phạm vi nghiên cứu.
- Lớp không thành công chiếm 28,98% và có F1 thấp hơn lớp thành công.

## Trạng thái hiện tại

Dự án đã chốt cấu hình tham chiếu ở Macro-F1 `0,719483` và chuyển sang giai
đoạn hoàn thiện báo cáo. Mọi phát triển mô hình mới phải giữ nguyên giao thức
đánh giá và được ghi nhận trong `docs/tuning_registry.md`.
