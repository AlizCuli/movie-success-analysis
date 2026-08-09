# Kết quả XGBoost trong phạm vi trước phát hành

## Kết quả tham chiếu ngoài mẫu

Mốc tham chiếu được đánh giá bằng kiểm định chéo phân tầng lồng nhau, gồm 5 vòng
ngoài và 4 vòng trong trên 1.646 phim. Kết quả ngoài mẫu gộp như sau:

| Chỉ số | Giá trị |
| --- | ---: |
| Macro-F1 | **0,719483** |
| F1 lớp 0 | 0,605128 |
| F1 lớp 1 | 0,833837 |
| Recall lớp 0 | 0,618449 |
| Recall lớp 1 | 0,826347 |
| Balanced accuracy | 0,722398 |
| Accuracy | 0,766100 |

Các vòng ngoài không tham gia lựa chọn đặc trưng, tiền xử lý, số vòng boosting
hoặc ngưỡng phân loại. Dự đoán cấp từng phim được lưu cục bộ; repository chỉ
công khai các bảng tổng hợp.

## Kết quả phục vụ báo cáo

- `xgboost_fold_metrics.csv`: độ ổn định giữa năm vòng kiểm định ngoài.
- `xgboost_confusion_matrix.csv`: ma trận nhầm lẫn tổng hợp.
- `xgboost_pooled_metrics.csv`: các chỉ số ngoài mẫu gộp.
- `xgboost_feature_importance.csv`: mức độ quan trọng của đặc trưng trong mô
  hình cuối; kết quả này không thay thế phân tích permutation hoặc SHAP ngoài
  mẫu.
- Ba hình tương ứng trong `reports/figures/`: ma trận nhầm lẫn, Macro-F1 theo
  vòng kiểm định và các đặc trưng có mức độ quan trọng cao nhất.

## Mô hình đóng gói

Mô hình trong `models/` được khớp trên toàn bộ 1.646 phim sau khi khóa cấu hình,
với 144 vòng boosting và ngưỡng phân loại 0,51. Do mô hình này đã quan sát toàn
bộ tập dữ liệu, kết quả khớp của nó không được sử dụng để báo cáo khả năng tổng
quát hóa; căn cứ chính thức vẫn là kết quả ngoài mẫu gộp nêu trên.

## Diễn giải và giới hạn

- Mô hình nhận diện lớp thành công tốt hơn lớp không thành công; F1 lớp 0 còn là
  giới hạn chính.
- Franchise history cải thiện Macro-F1 so với A+B cố định từ 0,710977 lên
  0,719483, nhưng chênh lệch nhỏ và cần được diễn giải thận trọng.
- Kết quả chịu ảnh hưởng bởi chiến lược lấy tối đa 100 phim phổ biến mỗi năm,
  dữ liệu tài chính thiếu và tính chất ảnh chụp của TMDb.
- Macro-F1 0,719483 cho thấy metadata trước phát hành chứa tín hiệu dự báo có
  ích, nhưng chưa đủ cho quyết định tài chính có rủi ro cao nếu dùng độc lập.
