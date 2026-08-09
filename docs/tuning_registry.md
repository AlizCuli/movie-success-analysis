# Sổ đăng ký thí nghiệm tối ưu

Tài liệu này ghi nhận các thí nghiệm đã được đánh giá nhằm tránh lặp lại những
hướng không hiệu quả. Mốc tham chiếu hiện hành là XGBoost A+B kết hợp lịch sử
franchise trong phạm vi trước phát hành theo quy ước vận hành, đạt Macro-F1
ngoài mẫu gộp **0,719483** trên 1.646 phim.

## Giao thức bắt buộc

- Biến mục tiêu: `revenue >= 2 × budget`.
- `StratifiedKFold` gồm 5 vòng ngoài với seed 42 và 4 vòng trong với seed 43.
- Macro-F1 là chỉ số chính.
- Mọi bước tiền xử lý, trạng thái đặc trưng, tối ưu tham số và lựa chọn ngưỡng
  phải diễn ra trong dữ liệu huấn luyện hoặc vòng kiểm định trong.
- Kết quả vòng ngoài không được sử dụng để lựa chọn cấu hình.
- Cấm revenue, popularity, rating, vote, profit, ROI và biến hậu phát hành.

## Các mốc đã đạt

| Mốc | Phạm vi | Macro-F1 | Kết luận |
| --- | --- | ---: | --- |
| Extended XGBoost cũ | Có tín hiệu hậu phát hành | 0,7597 | Không hợp lệ cho đề tài cuối |
| XGBoost gần với phạm vi trước phát hành | Phạm vi cũ chưa được kiểm định đầy đủ | 0,7144 | Mốc lịch sử, không phải mốc tham chiếu hiện hành |
| Phạm vi bảo thủ nghiêm ngặt | Chỉ các trường có bằng chứng chặt | 0,6125 | Hợp lệ nhưng thiếu tín hiệu |
| Kiến tạo đặc trưng nghiêm ngặt | Phạm vi nghiêm ngặt với biểu diễn mới | 0,6706 | Cải thiện nhưng thấp hơn phạm vi vận hành |
| A+B vận hành cố định | Trước phát hành theo quy ước vận hành | 0,710977 | Cấu hình đối chứng chính thức |
| Bối cảnh ngân sách vận hành V2 | Trước phát hành theo quy ước vận hành | 0,7087 | Không vượt cấu hình đối chứng |
| A+B kết hợp lịch sử franchise | Trước phát hành theo quy ước vận hành | **0,719483** | Mốc tham chiếu chính thức |

## Hướng đã thử và bị loại

### Tín hiệu hậu phát hành

Popularity, rating và vote từng giúp điểm cao hơn, nhưng không có sẵn tin cậy
trước phát hành. Các trường này không được đưa trở lại dù có thể làm tăng chỉ
số đánh giá.

### Hyperparameter và threshold

Grid search và Optuna đã được sử dụng để khảo sát learning rate, độ sâu, child
weight, subsampling, regularization, trọng số lớp, số cây và ngưỡng phân loại.
Không gian tìm kiếm rộng không tạo cải thiện ổn định so với cấu hình A+B đã
khóa. Không lặp lại không gian tìm kiếm cũ nếu không có đặc trưng hoặc dữ liệu
mới.

### Biểu diễn nội dung và metadata

Đã thử TF-IDF/SVD overview, keyword identity, interaction budget/genre, cách mã
hóa biến phân loại và các nhóm siêu dữ liệu mở rộng. Kết quả vòng kiểm định
trong không cải thiện ổn
định hoặc tăng độ phức tạp nhiều hơn giá trị thu được.

### Operational Budget Context V1

Các biến bối cảnh ngân sách chỉ tăng Macro-F1 vòng trong trung bình khoảng
**+0,001261**,
thấp hơn tiêu chí +0,003 nên bị loại. Bảy đặc trưng của nhóm này không thuộc
mốc tham chiếu và không nên được đưa lại nguyên trạng.

### Entity history V1

Đặc trưng lịch sử theo thời điểm từ ảnh chụp dữ liệu hiện có không vượt tiêu chí
sàng lọc vòng trong:

| Nhóm đặc trưng | Chênh lệch Macro-F1 vòng trong so với đối chứng |
| --- | ---: |
| Top-billed cast | -0,001900 |
| Director | -0,002840 |
| Production company | -0,002957 |

Nhánh nhà phân phối bị chặn vì không có nguồn gốc dữ liệu và khóa định danh đủ
tin cậy; `production_companies` không được xem là nhà phân phối. Hướng này chỉ
nên được đánh giá lại khi có kho lịch sử rộng hơn, độ bao phủ tốt hơn và giả
thuyết mới rõ ràng.

### Mô hình ngoài phạm vi

k-NN, Logistic Regression, Random Forest và CatBoost từng được khảo sát nhưng
đã bị loại khi đề tài chốt chỉ sử dụng XGBoost. Không tái triển khai trong quy trình
cuối.

## Điều kiện mở thí nghiệm mới

Một thí nghiệm mới phải ghi rõ giả thuyết khác biệt, dữ liệu hoặc đặc trưng mới,
giao thức chống rò rỉ, tiêu chí sàng lọc vòng trong, chi phí tính toán và kết
quả riêng. Không ghi đè mô hình hoặc bảng tham chiếu hiện hành. Đánh giá vòng
ngoài chỉ được thực hiện một lần sau khi tập đặc trưng đã được khóa hoàn toàn
bằng kiểm định vòng trong.
