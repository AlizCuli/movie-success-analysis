# Phạm vi pre-release operational

## Quyết định phạm vi

Từ thời điểm này, project chấp nhận các metadata TMDb đã thu thập như collection, hãng sản xuất, cast/crew count, ngôn ngữ, certification, release-event, overview/tagline và keyword là thông tin **pre-release operational**. Đây là quy ước vận hành cho thí nghiệm, không phải bằng chứng rằng từng trường có snapshot lịch sử trước ngày phát hành.

Phiên bản `strict_pre_release` vẫn được giữ nguyên như mốc bảo thủ chống leakage. Không được ghi đè hoặc trộn kết quả giữa hai phạm vi.

## Benchmark chính thức

Benchmark chính thức hiện tại là `XGBoost operational A+B + franchise history`
với pooled outer OOF Macro-F1 **0,719483**, F1 lớp 0 **0,605128**, recall lớp 0
**0,618449** và balanced accuracy **0,722398**.

Mốc này dùng 1.646 phim, target `is_successful`, outer StratifiedKFold 5 fold (`shuffle=True`, seed 42) và inner StratifiedKFold 4 fold (seed 43). Target được giữ cố định: `success = 1` khi `revenue >= 2 * budget`.

## Feature được chấp nhận

- Family A: log budget, runtime, lịch phát hành, genre, ngôn ngữ, quốc gia và thông tin hãng sản xuất cơ bản.
- Family B: collection, metadata hãng/quốc gia/ngôn ngữ, cast/crew count, certification, release-event, nội dung overview/tagline và keyword.

Rating, vote, popularity, IMDb rating/vote, revenue, profit, ROI, `revenue_to_budget`, target và các biến dẫn xuất từ doanh thu vẫn bị cấm tuyệt đối.

## Quy tắc phát triển tiếp theo

1. Dùng đúng 1.646 ID, target và outer fold assignment của mốc lịch sử.
2. Chỉ dùng XGBoost; không chạy thêm model khác.
3. Fit encoder, imputer, vocabulary, feature selection, class weight và threshold bên trong training/inner folds.
4. Không dùng revenue hoặc tín hiệu audience hậu phát hành.
5. Thử từng family hoặc cách biểu diễn mới một cách độc lập; chỉ chọn nếu cải thiện inner CV nhất quán trước khi báo outer OOF.
6. Báo cáo riêng `pre_release_operational`, không gọi kết quả đó là strict pre-release.

## Kết quả franchise history point-in-time

Thí nghiệm riêng A+B cộng bốn đặc trưng lịch sử collection theo thời điểm đã
đạt Macro-F1 outer OOF **0,719483** trên đúng 1.646 phim và cùng outer split.
Kết quả cao hơn A+B cố định 0,008505 và cao hơn mốc lịch sử 0,7144 khoảng
0,0051. History chỉ dùng các phim phát hành trước ngày của phim đang dự đoán,
được fit riêng trong training của từng fold và không dùng biến doanh thu hay
tín hiệu hậu phát hành. Chi tiết tại
`docs/operational_franchise_history_findings.md`.
