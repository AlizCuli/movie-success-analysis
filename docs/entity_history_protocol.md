# Protocol feature lịch sử thực thể V1

## Control và phạm vi

- Control giữ nguyên: `A+B + franchise history`.
- Tập cố định: 1.646 phim, không đổi target hoặc outer fold assignment.
- Benchmark bảo toàn: Macro-F1 outer OOF `0.719483`.
- XGBoost và preprocessing của control không được retune khi screening block.

## Quy tắc point-in-time

Với phim cần dự đoán tại ngày `t`:

1. Chỉ phim có `historical_release_date < t` được tính vào lịch sử.
2. Revenue, nhãn thành công và recent financial form chỉ được dùng khi phim lịch sử đã
   phát hành ít nhất 365 ngày trước `t`.
3. `tmdb_id` của chính phim cần dự đoán luôn bị loại.
4. Kho external loại toàn bộ 1.646 target IDs. Các phim target thuộc training partition
   được thêm vào state riêng của fold và vẫn phải thỏa điều kiện thời gian.
5. Validation và outer-test không cập nhật state.
6. Smoothing prior cũng chỉ được tính từ phim đủ điều kiện trước `t`.

Maturity lag 365 ngày, smoothing strength 10, recent window 3 phim và half-life 3 năm
được khóa trong `config/entity_history_v1.json` trước khi đánh giá outer.

## Tổng hợp nhiều thực thể

- Director: dùng tất cả credit có `job == Director`; tổng hợp mean/max và min recency.
- Production company: dùng tất cả TMDb company ID với trọng số bằng nhau.
- Cast: cố định top 5 theo billing order; mean dùng trọng số `1 / (order + 1)`.
- Luôn có số thực thể, số thực thể có history, coverage rate và missing fallback.

## Inner gate

Từng block director, production company và cast được so với control bằng cùng 5 outer
training partitions, mỗi partition dùng inner Stratified CV 4 fold. Distributor không
được chạy khi nguồn còn bị chặn.

Block chỉ đạt khi:

- mean delta Macro-F1 inner ít nhất `+0.003`;
- thắng control ở ít nhất 3/5 outer-training partitions;
- mean delta recall lớp 0 không thấp hơn `-0.01`;
- audit không phát hiện leakage hoặc ID trùng.

Chỉ các block đạt mới được đưa vào ablation tổ hợp. Tổ hợp tốt nhất được khóa hoàn toàn
từ inner result trước khi outer-validation được chạy đúng một lần. Outer result không
được dùng để quay lại chọn group, feature, cast K, smoothing, lag hoặc threshold.

## Feature bị cấm

Không dùng predictor của chính phim gồm revenue, popularity, vote/rating TMDb,
rating/vote IMDb, profit, ROI hoặc revenue-to-budget. Bảy feature Operational Budget
Context V1 đã bị bác bỏ cũng bị blacklist trong code và không được đưa lại.

## Giới hạn provenance

TMDb là snapshot hiện tại, không phải archive cho biết chính xác thời điểm từng trường
được công bố. Cắt theo release date và maturity lag giảm leakage nhưng không chứng minh
strict as-of availability tuyệt đối; kết quả phải tiếp tục được gọi là pre-release
operational point-in-time history.
