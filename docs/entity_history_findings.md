# Kết quả Entity History Enrichment V1

## Mục tiêu

Thí nghiệm bổ sung lịch sử point-in-time cho đạo diễn, production company và
top-billed cast vào control `A+B + franchise history`. Tập đánh giá chính vẫn
gồm đúng 1.646 phim, outer splits và benchmark Macro-F1 `0.719483` không thay
đổi. Các phim lịch sử chỉ là dữ liệu phụ trợ, không được thêm thành quan sát có
nhãn của tập modeling.

Nhánh distributor bị chặn vì chưa có nguồn nào đồng thời chứng minh được vai
trò nhà phân phối thực sự, có ID ổn định và ghép được với phim đủ tin cậy.
`production_companies` của TMDb không được dùng thay cho distributor.

## Kho dữ liệu lịch sử

Snapshot `tmdb-2026-08-05-v1` được tạo ngày 2026-08-06, gồm:

- 134.254 phim lịch sử duy nhất;
- 8.022 quan hệ đạo diễn-phim;
- 98.788 quan hệ production company-phim;
- 162.144 quan hệ top-5 cast-phim;
- 6.808 thực thể duy nhất.

Ba movie-detail request trả HTTP 404 không thể phục hồi (`1741345`, `1742666`,
`1742674`). Không có target movie nào trong 1.646 phim xuất hiện trong kho lịch
sử, không có khóa quan hệ trùng và không có ngày phát hành ngoài phạm vi đã
khóa.

## Coverage

Coverage trung bình trên năm outer-training partition:

| Nhóm | Ghép được thực thể | Unseen | Có >=1 history | Có >=5 history | Có financial history đã mature |
| --- | ---: | ---: | ---: | ---: | ---: |
| Director | 100,00% | 5,83% | 94,17% | 57,53% | 75,58% |
| Production company | 100,00% | 0,55% | 99,45% | 97,27% | 97,33% |
| Top-5 cast | 100,00% | 0,18% | 99,82% | 99,27% | 98,30% |

Đây là coverage rộng hơn rõ rệt so với block H cũ. Tuy vậy, coverage cao không
đồng nghĩa representation tạo thêm tín hiệu dự báo hữu ích.

## Chống leakage

- State được tạo từ dữ liệu lịch sử ngoài target và target rows thuộc training
  partition; validation/test không cập nhật history.
- Chỉ phim có `release_date` nhỏ hơn phim cần dự đoán mới được tính vào history.
- Feature dùng revenue hoặc nhãn thành công áp dụng maturity lag cố định 365 ngày.
- Target movie bị loại bằng `tmdb_id`, kể cả khi có mặt trong snapshot nguồn.
- Thực thể được ghép bằng ID TMDb, không ghép bằng tên.
- Top cast cố định năm vị trí, giữ `billing_order`; cast được tổng hợp với trọng
  số giảm theo thứ tự billing. Đạo diễn và company dùng trọng số đều.
- Không predictor nào dùng popularity, vote/rating, revenue của phim mục tiêu,
  profit, ROI hoặc `revenue_to_budget`.

Mười một unit tests về schema, ID, thứ tự cast, date cutoff, maturity lag,
state isolation, target exclusion và predictor bị cấm đều đạt.

## Inner gate

Mỗi block được so sánh độc lập với control bằng 4-fold inner OOF trên từng một
trong năm outer-training partition. XGBoost và outer splits giữ nguyên. Gate
được khóa trước:

- mean delta Macro-F1 tối thiểu `+0.003`;
- thắng control ở ít nhất 3/5 partition;
- mean delta recall lớp 0 không thấp hơn `-0.01`;
- không leakage và kết quả phải tái lập được.

| Block | Inner Macro-F1 trung bình | Delta | Số partition thắng | Delta recall lớp 0 | Kết luận |
| --- | ---: | ---: | ---: | ---: | --- |
| Top-5 cast | 0,707323 | -0,001900 | 2/5 | -0,025714 | Loại |
| Director | 0,706383 | -0,002840 | 1/5 | +0,035113 | Loại |
| Production company | 0,706266 | -0,002957 | 1/5 | -0,012120 | Loại |

Không block nào đạt gate. Vì vậy không có block hợp lệ để kết hợp, không thực
hiện ablation tổ hợp và **không chạy outer-validation**. Không có outer score,
confusion matrix hay model mới từ thí nghiệm này. Benchmark chính thức vẫn là
Macro-F1 `0.719483`.

## Kết luận và giới hạn

Lịch sử thực thể V1 đã giải quyết nút thắt coverage của block H cũ nhưng cách
tổng hợp mean/max hiện tại không cải thiện ổn định. Không chạy lại nguyên trạng
18 feature mỗi nhóm với smoothing 10, recent-3, half-life 3 năm và maturity lag
365 ngày. Một lần thử sau chỉ hợp lệ khi thay đổi rõ giả thuyết hoặc
representation, ví dụ giảm nhiễu bằng lựa chọn vai trò cụ thể, time-decay được
khóa từ inner CV, hoặc biểu diễn company/cast theo phân vị thay cho mean/max.

