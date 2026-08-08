# Dự đoán khả năng thành công tài chính của phim trước phát hành

## Thông tin đề tài

- **Môn học:** Phân tích Dữ liệu
- **Trường:** Trường Đại học Mở Thành phố Hồ Chí Minh
- **Sinh viên thực hiện:**
  - Phan Tấn Phúc — 2351060029
  - Phạm Nguyễn Bảo Vi — 2351060041

## Mục tiêu nghiên cứu

Nghiên cứu đánh giá khả năng dự đoán thành công tài chính của phim từ những
thông tin có thể biết trước khi phát hành. Bài toán được mô hình hóa dưới dạng
phân loại nhị phân và sử dụng XGBoost làm mô hình chính.

Một phim được xem là thành công theo định nghĩa vận hành sau:

```text
is_successful = 1 nếu revenue >= 2 × budget
is_successful = 0 nếu revenue < 2 × budget
```

Ngân sách (`budget`) là thông tin đầu vào có thể biết trước phát hành. Doanh thu
(`revenue`) chỉ được dùng để tạo nhãn và không được sử dụng làm biến dự báo.

## Dữ liệu nghiên cứu

Nguồn dữ liệu chính thức duy nhất của báo cáo và mô hình cuối là **TMDB
Official API**. Bộ dữ liệu ban đầu gồm 2.597 phim phát hành trong giai đoạn
2000–2025, được lấy tối đa 100 phim phổ biến cho mỗi năm. Sau khi áp dụng các
điều kiện về ngân sách, doanh thu, thời lượng và ngày phát hành, tập dùng để
đánh giá mô hình còn 1.646 phim.

Các nhóm thông tin đầu vào chính gồm:

- ngân sách, thời lượng và thời điểm phát hành;
- thể loại đa nhãn, ngôn ngữ và quốc gia sản xuất;
- công ty sản xuất, collection, chứng nhận phân loại và metadata phát hành;
- lịch sử chuỗi phim (franchise) theo thời điểm, chỉ sử dụng các phim đã phát
  hành trước phim cần dự đoán.

Mô hình không sử dụng `revenue`, `popularity`, điểm hoặc số lượt bình chọn,
`profit`, ROI hay các tín hiệu hậu phát hành khác của chính bộ phim.

## Phương pháp đánh giá

Mô hình được đánh giá bằng nested Stratified Cross-Validation gồm 5 outer fold
và 4 inner fold. Các bước điền giá trị thiếu, mã hóa, xây dựng đặc trưng lịch
sử, chọn số vòng lặp và chọn ngưỡng phân loại đều được thực hiện bên trong phần
dữ liệu huấn luyện tương ứng. Outer fold chỉ được dùng để ước lượng hiệu năng
cuối, qua đó hạn chế rò rỉ dữ liệu trong quá trình lựa chọn mô hình.

## Kết quả chính

Benchmark chính thức là **XGBoost pre-release operational kết hợp lịch sử
franchise theo thời điểm**. Kết quả gộp từ dự đoán outer-fold ngoài mẫu trên
1.646 phim như sau:

| Chỉ số | Giá trị |
| --- | ---: |
| Macro-F1 | **0,719483** |
| F1 lớp không thành công | 0,605128 |
| Recall lớp không thành công | 0,618449 |
| Balanced accuracy | 0,722398 |

Đây là mốc đối chứng chính thức cho các thí nghiệm tiếp theo. Phạm vi được gọi
là *pre-release operational* vì metadata được lấy từ snapshot TMDb hiện tại,
không phải kho lưu trữ lịch sử chứng minh tuyệt đối thời điểm từng trường được
công bố.

## Nguồn và ghi nhận dữ liệu

Nguồn dữ liệu: [The Movie Database (TMDB)](https://www.themoviedb.org/).

> This product uses the TMDB API but is not endorsed or certified by TMDB.

Các artifact IMDb của giai đoạn khảo sát ban đầu không thuộc tập biến dự báo,
không tham gia benchmark và không được sử dụng trong báo cáo cuối.

### Chính sách phân phối dữ liệu

Các tệp dữ liệu gốc, trung gian và đã xử lý không được phân phối qua repository
công khai. Repository chỉ giữ cấu trúc thư mục `data/`, mã nguồn, tài liệu, kết
quả tổng hợp và gói mô hình. Người dùng cần tự tái tạo dữ liệu bằng các script
trong `src/` và API Read Access Token TMDb của chính mình. Chi tiết được trình
bày trong `data/README.md` và `docs/data_sources.md`.

## Thành phần chính của repository

1. `src/collect_tmdb_full.py`: thu thập dữ liệu phim từ TMDb.
2. `src/preprocess_movies.py`: làm sạch và cấu trúc hóa dữ liệu.
3. `src/collect_tmdb_enrichment.py`: thu thập metadata bổ sung.
4. `src/eda_movies.py`: tái tạo phân tích khám phá, bảng và biểu đồ.
5. `src/reproduce_operational_ab_baseline.py`: tái lập mô hình đối chứng A+B.
6. `src/evaluate_operational_franchise.py`: đánh giá benchmark XGBoost.
7. `src/train_final_xgboost.py`: huấn luyện và đóng gói mô hình cuối.
8. `src/predict_xgboost.py`: nạp gói mô hình và thực hiện dự đoán.

Notebook EDA nằm tại `notebooks/01_eda.ipynb`. Khung báo cáo nằm tại
`docs/report_outline.md`. Bối cảnh bàn giao và lịch sử thí nghiệm được ghi tại
`GPT_CONTEXT.md` và `docs/tuning_registry.md`.
