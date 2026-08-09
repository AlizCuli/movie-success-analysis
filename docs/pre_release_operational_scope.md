# Phạm vi pre-release operational

## Quy ước phạm vi

Project sử dụng metadata TMDb có ý nghĩa trước phát hành: budget, runtime, lịch
phát hành, genre, ngôn ngữ, quốc gia, công ty sản xuất, collection,
certification, quy mô cast/crew, thông tin phát hành, overview/tagline và
keyword. Đây là phạm vi **pre-release operational** vì snapshot TMDb được tải ở
hiện tại, không phải archive chứng minh thời điểm công bố của từng trường.

Revenue chỉ tạo target. Popularity, rating, vote, profit, ROI và mọi biến dẫn
xuất từ revenue bị cấm tuyệt đối trong predictor.

## Feature set chính thức

- **Metadata cơ bản:** `log_budget`, runtime, năm/tháng/mùa phát hành, genre đa
  nhãn, ngôn ngữ, quốc gia và công ty sản xuất.
- **Metadata enrichment:** collection, số công ty/ngôn ngữ, số cast/crew,
  certification, số quốc gia/sự kiện chiếu rạp, độ dài overview, tagline và
  keyword.
- **Franchise history point-in-time:** số phim trước đó, tỷ lệ thành công đã
  smoothing, mean log-budget lịch sử và số năm từ phần phim gần nhất.

## Protocol chống leakage

1. Giữ cố định 1.646 phim, target và outer StratifiedKFold 5 fold, seed 42.
2. Inner StratifiedKFold 4 fold, seed 43 dùng để chọn số vòng và threshold.
3. Imputer, encoder, genre vocabulary và history builder chỉ fit trên training
   partition tương ứng.
4. Franchise history chỉ dùng phim có `release_date` sớm hơn phim truy vấn; phim
   cùng ngày không làm history cho nhau.
5. Outer validation chỉ đánh giá cuối, không quay lại lựa chọn feature hay tham
   số.

## Benchmark

XGBoost A+B + franchise history đạt pooled outer-OOF Macro-F1 **0,719483**, F1
lớp 0 **0,605128**, recall lớp 0 **0,618449** và balanced accuracy **0,722398**.
