# Tuning Registry

Mục đích của registry là ngăn lặp lại thí nghiệm đã được đánh giá. Benchmark
khóa hiện tại: XGBoost pre-release operational A+B + franchise history,
Macro-F1 outer-OOF **0,719483** trên 1.646 phim.

## Protocol bắt buộc

- Target: `revenue >= 2 × budget`.
- Outer StratifiedKFold 5 fold, seed 42; inner 4 fold, seed 43.
- Macro-F1 là metric chính.
- Mọi preprocessing, feature state, tuning và threshold phải nằm trong
  training/inner CV.
- Không chọn cấu hình bằng outer result.
- Cấm revenue, popularity, rating, vote, profit, ROI và biến hậu phát hành.

## Các mốc đã đạt

| Mốc | Scope | Macro-F1 | Kết luận |
| --- | --- | ---: | --- |
| Extended XGBoost cũ | Có tín hiệu hậu phát hành | 0,7597 | Không hợp lệ cho đề tài cuối |
| Pre-release-like XGBoost | Phạm vi cũ chưa audit đầy đủ | 0,7144 | Mốc lịch sử, không phải benchmark hiện hành |
| Strict conservative | Chỉ trường có bằng chứng chặt | 0,6125 | Hợp lệ bảo thủ nhưng thiếu tín hiệu |
| Strict feature engineering | Strict + biểu diễn mới | 0,6706 | Cải thiện nhưng thấp hơn operational |
| Operational A+B cố định | Pre-release operational | 0,710977 | Control chính thức |
| Operational Budget Context V2 | Pre-release operational | 0,7087 | Không vượt control |
| A+B + franchise history | Pre-release operational | **0,719483** | Benchmark chính thức |

## Hướng đã thử và bị loại

### Tín hiệu hậu phát hành

Popularity, rating và vote từng giúp điểm cao hơn, nhưng không có sẵn tin cậy
trước phát hành. Không đưa trở lại dù có thể tăng metric.

### Hyperparameter và threshold

Đã thử grid/Optuna cho learning rate, depth, child weight, subsampling,
regularization, class weight, số cây và threshold. Tuning rộng không tạo cải
thiện ổn định so với cấu hình A+B đã khóa. Không lặp lại search space cũ nếu
không có feature/data mới.

### Biểu diễn nội dung và metadata

Đã thử TF-IDF/SVD overview, keyword identity, interaction budget/genre, cách mã
hóa category và các nhóm metadata mở rộng. Kết quả inner CV không cải thiện ổn
định hoặc tăng độ phức tạp nhiều hơn giá trị thu được.

### Operational Budget Context V1

Các biến bối cảnh budget chỉ tăng inner Macro-F1 trung bình khoảng **+0,001261**,
thấp hơn gate +0,003 nên bị loại. Bảy feature của block này không thuộc
benchmark và không nên đưa lại nguyên trạng.

### Entity history V1

History point-in-time từ snapshot hiện có không vượt inner gate:

| Block | Delta inner Macro-F1 so với control |
| --- | ---: |
| Top-billed cast | -0,001900 |
| Director | -0,002840 |
| Production company | -0,002957 |

Distributor bị chặn vì không có nguồn provenance và khóa ID đáng tin cậy;
`production_companies` không được coi là distributor. Chỉ thử lại khi có kho
lịch sử rộng hơn, coverage tốt hơn và giả thuyết mới rõ ràng.

### Mô hình ngoài phạm vi

k-NN, Logistic Regression, Random Forest và CatBoost từng được khảo sát nhưng
đã bị loại khi đề tài chốt chỉ dùng XGBoost. Không tái triển khai trong pipeline
cuối.

## Điều kiện mở thí nghiệm mới

Một thí nghiệm mới phải ghi rõ: giả thuyết khác biệt, dữ liệu/feature mới,
protocol chống leakage, inner gate, chi phí tính toán và artifact riêng. Không
ghi đè model hoặc bảng benchmark hiện hành. Kết quả outer chỉ được chạy một lần
sau khi feature set đã khóa hoàn toàn bằng inner validation.
