# Trạng thái dự án

## Phạm vi chính thức

- Bài toán: dự đoán khả năng thành công tài chính của phim trước phát hành.
- Mô hình duy nhất: XGBoost.
- Nhãn: `revenue >= 2 × budget`.
- Metric chính: Macro-F1.
- Đánh giá: nested Stratified CV 5 outer fold × 4 inner fold.

## Dữ liệu

- Nguồn dữ liệu chính thức duy nhất của báo cáo và mô hình cuối là TMDb
  Official API.
- Đã thu thập 2.597 phim TMDb trong giai đoạn 2000–2025.
- Tập dữ liệu dùng cho mô hình có 1.646 phim: 477 phim không thành công và
  1.169 phim thành công.
- Không có `tmdb_id` trùng hoặc năm phát hành ngoài phạm vi.
- Dữ liệu gốc không bị chỉnh sửa trong chặng rút gọn repository.
- Các tệp dữ liệu được giữ local và không được phân phối qua repository public;
  GitHub chỉ lưu cấu trúc thư mục, `data/README.md` và các tệp `.gitkeep`.

## Benchmark được chọn

**XGBoost pre-release operational với franchise history point-in-time** là mốc
chính thức:

- Macro-F1 outer OOF: **0,719483**.
- F1 lớp 0: **0,605128**.
- Recall lớp 0: **0,618449**.
- Balanced accuracy: **0,722398**.
- Số phim đánh giá: **1.646**.

Mô hình dùng nhóm đặc trưng A+B và bốn đặc trưng lịch sử franchise:

- `collection_prior_movie_count`;
- `collection_prior_success_rate`;
- `collection_prior_mean_log_budget`;
- `collection_years_since_previous`.

Lịch sử chỉ sử dụng phim phát hành sớm hơn phim đang dự đoán và chỉ được xây
dựng từ phân vùng huấn luyện của từng fold.

## Kiểm soát leakage

- Không dùng doanh thu làm biến dự báo; doanh thu chỉ được dùng để tạo nhãn.
- Không dùng popularity, vote/rating, profit, ROI hoặc `revenue_to_budget`.
- Bộ từ vựng thể loại, bước điền thiếu, bộ mã hóa, ngưỡng phân loại và lịch sử
  franchise đều được học bên trong dữ liệu huấn luyện/inner CV.
- Outer validation chỉ dùng để đánh giá sau khi khóa cấu hình.

## Diễn giải phạm vi

Benchmark là **pre-release operational**: các metadata được xem là có thể biết
trước phát hành, nhưng TMDb hiện tại không cung cấp snapshot lịch sử để chứng
minh giá trị từng trường tại đúng thời điểm đó. Kết quả hậu phát hành 0,7597
không còn được xem là ứng viên cho bài toán chính.

## Trạng thái hiện tại

Repository đã được rút gọn quanh pipeline dữ liệu và benchmark XGBoost
0,719483. Các mô hình và artifact thí nghiệm không thuộc phạm vi cuối đã được
xóa theo xác nhận của người dùng.

Các thành phần cần thiết cho báo cáo đã được khôi phục hoặc xây dựng lại:

- Notebook EDA, 8 bảng và 8 biểu đồ từ dữ liệu hiện tại.
- Khung báo cáo gồm giới thiệu, dữ liệu, phương pháp, phân tích và kết quả.
- 4 bảng và 4 biểu đồ đánh giá riêng cho benchmark XGBoost.
- Phân tích ma trận nhầm lẫn, độ ổn định giữa các outer fold, lỗi dự đoán và độ
  quan trọng của đặc trưng.

Mô hình XGBoost cuối đã được huấn luyện trên toàn bộ 1.646 phim theo cấu hình
benchmark đã khóa, không tinh chỉnh lại siêu tham số:

- Số vòng cuối: 144.
- Ngưỡng phân loại được chọn bằng inner OOF 4 fold: 0,51.
- 51 đặc trưng thô và 160 đặc trưng sau tiền xử lý.
- Gói mô hình, mô hình native và manifest nằm trong `models/`.
- Gói mô hình đã được nạp lại và kiểm thử dự đoán thành công; benchmark chính thức
  vẫn là outer-OOF Macro-F1 0,719483.

## Thí nghiệm Operational Budget Context V1

- Đã tạo bảy đặc trưng ngân sách point-in-time: phân vị, chênh lệch so với trung
  vị, số quan sát lịch sử và cờ sẵn có cho toàn kỳ/ba năm.
- Đặc trưng chỉ dùng ngân sách của phim trong phân vùng huấn luyện có ngày phát
  hành sớm hơn; không dùng nhãn và dữ liệu kiểm định không cập nhật trạng thái.
- Sáu kiểm thử đơn vị về giới hạn thời gian, cô lập trạng thái, độc lập với
  nhãn, xử lý thiếu, loại bản ghi cùng ngày và biến dự báo bị cấm đều đạt.
- Tỷ lệ quan sát có ít nhất một bản ghi lịch sử đạt 99,39–100% giữa các fold.
- Inner screening Macro-F1 delta trung bình: **+0,001261**; thắng 3/5 partition;
  recall lớp 0 delta trung bình: **+0,002566**.
- Gate yêu cầu delta ít nhất +0,003 nên thí nghiệm bị loại trước outer evaluation.
- Không có outer score mới và benchmark **0,719483** vẫn giữ nguyên.
- Không nên lặp lại đúng representation global/3-year của Budget Context V1.

## Thí nghiệm Entity History Enrichment V1

- Đã tạo kho lịch sử phụ trợ TMDb riêng cho đạo diễn, công ty sản xuất và năm
  diễn viên có thứ tự billing cao nhất; phim lịch sử không được thêm thành quan
  sát có nhãn của tập 1.646 phim.
- Snapshot `tmdb-2026-08-05-v1` gồm 134.254 phim, 8.022 director edges, 98.788
  production-company edges, 162.144 cast edges và 6.808 thực thể.
- Ba movie-detail request trả HTTP 404 không thể phục hồi; không có target ID
  leakage, quan hệ trùng hoặc ngày phát hành ngoài phạm vi.
- Distributor bị chặn vì chưa có nguồn provenance và ghép ID đủ tin cậy;
  `production_companies` không được dùng thay thế.
- Mười một kiểm thử đơn vị về schema và rò rỉ dữ liệu theo thời điểm đều đạt.
- Inner screening so với `A+B + franchise history`:
  - cast: delta Macro-F1 `-0,001900`, thắng 2/5 partition;
  - director: delta `-0,002840`, thắng 1/5 partition;
  - production company: delta `-0,002957`, thắng 1/5 partition.
- Cả ba block không đạt gate `+0,003`; không tạo tổ hợp và không chạy
  outer-validation.
- Benchmark chính thức **0,719483** và mô hình hiện hành được giữ nguyên.
