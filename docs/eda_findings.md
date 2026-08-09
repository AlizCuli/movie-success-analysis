# Phát hiện EDA TMDb-only

## Chất lượng và phạm vi mẫu

- Bộ dữ liệu làm sạch gồm 2.597 phim; tập dữ liệu mô hình hóa có 1.646 phim với
  `budget`, `revenue`, `runtime`, `release_date` và biến mục tiêu hợp lệ.
- Budget thiếu 855 phim (32,92%), revenue thiếu 839 phim (32,31%) và runtime
  thiếu 72 phim (2,77%) trong tập làm sạch.
- Tập dữ liệu mô hình hóa gồm 477 phim không thành công (28,98%) và 1.169 phim
  thành công (71,02%), cho thấy mất cân bằng lớp ở mức vừa phải.

## Mối liên hệ mô tả

- `budget` và `revenue` đều có phân phối lệch phải; trung vị thấp hơn trung
  bình. Vì vậy, phép biến đổi `log1p` được sử dụng khi mô tả quy mô tài chính và
  biểu diễn `budget` trong mô hình.
- Trong các đặc trưng trước phát hành, từng biến riêng lẻ chỉ có tương quan
  Spearman yếu đến vừa với biến mục tiêu. Kết quả này cho thấy sự cần thiết của
  việc kết hợp nhiều nhóm siêu dữ liệu thay vì dựa vào một chỉ báo đơn lẻ.
- Tỷ lệ thành công thay đổi theo `primary_genre` và trạng thái collection, nhưng
  số phim giữa các nhóm không đồng đều. Biểu đồ nhóm luôn kèm `n` để tránh diễn
  giải quá mức ở nhóm nhỏ.
- Tỷ lệ thành công theo năm biến động trong mẫu. Mỗi năm chỉ có tối đa 100 phim
  phổ biến nên xu hướng không đại diện cho toàn bộ thị trường điện ảnh.

Hai hình chính được tạo bởi `src/eda_movies.py`:

- `reports/figures/dataset_overview.png`;
- `reports/figures/pre_release_feature_associations.png`.

Các bảng tương ứng nằm trong `reports/tables/dataset_summary.csv`,
`core_missingness.csv`, `pre_release_spearman.csv`,
`success_by_primary_genre_collection.csv` và `yearly_success_summary.csv`.

## Giới hạn diễn giải

- Các hệ số tương quan và tỷ lệ nhóm chỉ mô tả mối liên hệ, không chứng minh
  quan hệ nhân quả.
- Giá trị tiền là danh nghĩa, chưa điều chỉnh lạm phát.
- Ảnh chụp TMDb hiện tại không chứng minh tuyệt đối thời điểm công bố của từng
  trường siêu dữ liệu.
- `revenue` chỉ được sử dụng để tạo biến mục tiêu và mô tả kết quả, không được
  sử dụng làm đặc trưng dự báo.
