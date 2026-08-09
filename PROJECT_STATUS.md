# Trạng thái dự án

## Phạm vi cuối

- Đề tài: dự đoán khả năng thành công tài chính của phim trước phát hành.
- Nguồn dữ liệu: TMDb Official API.
- Target: `revenue >= 2 × budget`.
- Mô hình: XGBoost pre-release operational.
- Đánh giá: nested Stratified CV 5 outer × 4 inner fold.
- Metric chính: Macro-F1.

## Dữ liệu

- 2.597 phim TMDb giai đoạn 2000–2025.
- 1.646 phim đủ điều kiện modeling.
- Lớp 0: 477 phim; lớp 1: 1.169 phim.
- Dữ liệu raw/interim/processed chỉ lưu local và bị Git bỏ qua.
- Repository public không chứa artifact dự đoán cấp từng phim.

## Benchmark được bảo toàn

| Chỉ số | Giá trị |
| --- | ---: |
| Macro-F1 outer-OOF | **0.719483** |
| F1 lớp 0 | 0.605128 |
| Recall lớp 0 | 0.618449 |
| Balanced accuracy | 0.722398 |
| Accuracy | 0.766100 |

Model cuối đã được đóng gói trong `models/` với 144 cây, threshold 0,51, 51
feature sau feature builder và 160 feature sau tiền xử lý.

## Thành phần hoàn thiện

- Pipeline thu thập, tiền xử lý, enrichment, EDA, đánh giá và huấn luyện.
- Notebook EDA TMDb-only không nhúng output dung lượng lớn.
- Hai hình EDA chính và ba hình đánh giá XGBoost.
- Model bundle, native model và manifest checksum.
- README có cấu trúc project, workflow và hướng dẫn chạy từ máy mới.
- Test contract không cần dữ liệu local.

## Giới hạn phải nêu trong báo cáo

- Mẫu tối đa 100 phim phổ biến mỗi năm tạo selection bias.
- TMDb hiện tại không phải archive point-in-time của mọi metadata.
- Nhãn là quy ước của project và không bao gồm mọi cấu phần chi phí/lợi nhuận.
- Lớp không thành công chiếm 28,98% và có F1 thấp hơn lớp thành công.

## Trạng thái hiện tại

Project đã chốt ở benchmark `0.719483` và chuyển sang giai đoạn hoàn thiện báo
cáo. Mọi phát triển model mới phải giữ nguyên protocol và được ghi trong
`docs/tuning_registry.md`.
