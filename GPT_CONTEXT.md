# Hồ sơ bàn giao cho GPT

> Đọc file này trước khi thảo luận hoặc phát triển mô hình. Chi tiết tất cả thí
> nghiệm đã chạy nằm tại `docs/tuning_registry.md`.

## 1. Bài toán đã chốt

- Mục tiêu: dự đoán **khả năng thành công tài chính trước khi phim phát hành**.
- Mô hình được giữ lại để phát triển: **XGBoost**.
- Nhãn vận hành:

  ```text
  is_successful = 1 nếu revenue >= 2 * budget
  is_successful = 0 nếu revenue < 2 * budget
  ```

- `revenue` chỉ được dùng để tạo nhãn, không bao giờ là predictor.
- Metric chính: **Macro-F1**. Luôn báo thêm F1/recall từng lớp, balanced accuracy
  và confusion matrix.
- Phạm vi chính thức hiện nay là **pre-release operational**: chấp nhận metadata
  TMDb vốn có thể biết trước lịch chiếu, nhưng snapshot được tải ở hiện tại nên
  không chứng minh được tuyệt đối rằng mọi giá trị chưa từng bị cập nhật sau khi
  phim phát hành.

## 2. Dữ liệu hiện có

- Thu thập 2.597 phim TMDb, tối đa 100 phim phổ biến cho từng năm 2000–2025.
- Nguồn dữ liệu chính thức duy nhất của báo cáo và mô hình cuối là TMDb Official
  API. Các cột/artifact IMDb từ giai đoạn khảo sát cũ không thuộc phạm vi cuối.
- Sau tiền xử lý có 1.646 phim đủ budget, revenue, runtime, release date và nhãn.
- Phân bố nhãn: 477 phim lớp 0 (28,98%) và 1.169 phim lớp 1 (71,02%).
- Dữ liệu enrichment TMDb hiện có cho toàn bộ 1.646 phim modeling. Ba phim trong
  tập 2.597 nhận HTTP 404 nhưng không thuộc tập modeling.
- Dữ liệu gốc và CSV trong `data/` không được sửa trực tiếp hoặc đưa vào Git.

## 3. Ranh giới chống leakage

### Predictor bị cấm

Không dùng trực tiếp hoặc gián tiếp:

- `revenue`, `log_revenue`, `profit`, `roi`, `revenue_to_budget`;
- `is_successful` hoặc cờ được suy ra từ target;
- `popularity`, `vote_average`, `vote_count`, `log_vote_count` của TMDb;
- bất kỳ tín hiệu phản hồi khán giả hoặc kết quả kinh doanh sau phát hành nào.

### Quy tắc đánh giá

- Nested `StratifiedKFold`: outer 5 fold, shuffle, seed 42; inner 4 fold,
  shuffle, seed 43.
- Outer fold chỉ dùng một lần để ước lượng cuối; không chọn feature, tham số,
  số cây hay threshold dựa trên outer-validation.
- Imputer, encoder, vocabulary, feature lịch sử, class weight, chọn feature,
  early stopping và threshold phải được fit/chọn trong training/inner CV.
- Khi so sánh với benchmark, giữ nguyên 1.646 ID, target và outer-fold assignment.
- Không gọi kết quả `pre_release_operational` là strict point-in-time.

## 4. Benchmark chính thức hiện tại

Mô hình: **XGBoost operational A+B + franchise history point-in-time**.

| Chỉ số pooled outer OOF | Giá trị |
| --- | ---: |
| Macro-F1 | **0,719483** |
| F1 lớp 0 | 0,605128 |
| Recall lớp 0 | 0,618449 |
| Balanced accuracy | 0,722398 |
| Accuracy | 0,766100 |

Đây là mốc phải được bảo toàn và dùng làm đối chứng cho mọi phát triển mới.
Direct ablation A+B cố định đạt Macro-F1 0,710977; bốn feature franchise tăng
0,008505. Mốc cũ 0,7144 là kết quả lịch sử pre-release-like. Kết quả 0,7597 cao
hơn nhưng dùng popularity/rating/vote sau phát hành nên **không hợp lệ** với bài
toán hiện tại.

## 5. Feature của benchmark 0,719483

Artifact nhận 51 cột thô và tạo 160 feature sau biến đổi:

- Numeric cơ bản: `log_budget`, runtime, năm phát hành, số genre, số quốc gia và
  số hãng sản xuất.
- Genre multi-hot; cờ tiếng Anh và sản xuất tại Mỹ.
- Categorical: tháng/mùa/thập niên phát hành, genre chính, ngôn ngữ, quốc gia,
  hãng chính, collection, certification và ID hãng chính.
- Metadata A+B: collection, số hãng/ngôn ngữ/cast/crew, số quốc gia chiếu rạp,
  số release event, độ dài overview, có tagline và số keyword.
- Franchise history point-in-time: số phim collection đã phát hành trước đó,
  tỷ lệ thành công lịch sử đã smoothing, mean `log_budget` trước đó và số năm từ
  phim trước.

Không dùng nội dung văn bản trực tiếp; overview/tagline chỉ góp các cờ hoặc số
lượng. Feature history chỉ sử dụng phim có ngày phát hành trước phim đang xét và
được dựng lại trong từng training fold.

### Cấu hình XGBoost đã khóa

```text
learning_rate=0.064, max_depth=4, min_child_weight=12
subsample=0.68, colsample_bytree=0.88, gamma=1.6
reg_alpha=0.14, reg_lambda=0.93, class_weight_0=1.5
OneHotEncoder(min_frequency=10)
```

Trong nested CV, số cây và threshold được chọn bằng inner OOF ở từng outer fold.
Model đóng gói cuối dùng 144 cây và threshold 0,51, được fit trên toàn bộ 1.646
phim. Con số 0,719483 là hiệu năng outer OOF của benchmark, không phải điểm test
riêng của artifact fit toàn bộ dữ liệu.

## 6. Các mốc đã đạt

| Chặng | Kết quả chính | Đánh giá hiện tại |
| --- | --- | --- |
| Thu thập và làm sạch | 2.597 phim, 1.646 phim modeling | Nền dữ liệu hiện hành |
| EDA | 8 bảng, 8 biểu đồ, notebook chạy được | Giữ cho báo cáo |
| Linear Regression | Pre-release test R² 0,410; extended 0,594 | Lịch sử môn học, không còn model chính |
| k-NN ban đầu | Pre-release test Macro-F1 0,540; extended 0,600 | Không đạt, extended hậu phát hành |
| k-NN tối ưu nested CV | Tốt nhất 0,654; bản compact ổn định 0,651 | Không còn trong phạm vi model |
| Logistic Regression | Threshold tuned 0,686 | Đối chứng lịch sử |
| Random Forest | Threshold 0,5 đạt 0,687 | Đối chứng lịch sử |
| XGBoost enriched hậu phát hành | **0,7597** | Cao nhất tuyệt đối nhưng không hợp lệ pre-release |
| XGBoost pre-release-like cũ | 0,7144 | Mốc lịch sử |
| Strict conservative XGBoost | 0,6125 | Mốc chống leakage bảo thủ |
| Strict feature engineering | 0,6706 | Lịch sử; source/kết quả đã bị dọn, chưa tái lập hiện tại |
| Operational A+B cố định | 0,7110 | Direct ablation hiện hành |
| Operational v2 G/H | 0,7087 | Không cải thiện, đã bác bỏ |
| A+B + franchise history | **0,7195** | Benchmark chính thức hiện tại |

## 7. Kết luận từ toàn bộ tuning

- Thêm popularity, vote và rating tạo mức tăng lớn nhất nhưng làm sai câu hỏi
  nghiên cứu; không được dùng lại để nâng điểm.
- Tuning rộng hyperparameter nhiều lần chỉ đem lại mức tăng nhỏ và thiếu ổn định.
  Tín hiệu mới có chất lượng quan trọng hơn tăng số Optuna trial.
- Threshold tuning có lợi cho Logistic Regression nhưng không ổn định với k-NN;
  với XGBoost phải tiếp tục chọn hoàn toàn bằng inner OOF.
- Feature metadata A+B là block pre-release operational mạnh nhất. Full text
  TF-IDF/keyword, identity one-hot thưa và history người/hãng ở coverage hiện tại
  không cải thiện ổn định.
- Mở rộng G (multi-hot/ratio/completeness) và H (history người/hãng/team) đã được
  thử; H không được chọn ở outer fold nào và toàn bộ v2 thấp hơn mốc 0,7144.
- Franchise history là tín hiệu mới duy nhất gần đây tạo cải thiện outer OOF rõ
  và nhất quán so với direct A+B.
- Lớp 0 ít hơn và các phim có `revenue_to_budget` gần 2 gây nhiễu nhãn. Không xóa
  các phim này chỉ để tăng điểm.
- Dữ liệu lấy theo popularity và tối đa 100 phim/năm tạo selection bias; hiệu năng
  chưa chứng minh khả năng tổng quát cho toàn bộ thị trường phim.

## 8. Không lặp lại nếu giả thuyết không thay đổi

- Không chạy lại cùng grid k-NN, scaler, oversampling, numeric/categorical weight
  hoặc threshold sweep đã ghi trong tuning registry.
- Không so sánh lại Logistic Regression/Random Forest/CatBoost khi phạm vi đã chốt
  chỉ dùng XGBoost.
- Không chạy lại 40–50 Optuna trial/fold trên compact/interaction/rich cũ.
- Không chạy lại operational v2 với đúng block G/H và coverage hiện tại.
- Không chạy lại TF-IDF overview/tagline + keyword multi-hot hoặc raw identity
  one-hot nếu không có biểu diễn/dữ liệu/coverage mới.
- Không tạo “cải thiện” bằng hậu phát hành, đổi target, đổi outer split, nhìn outer
  để chọn cấu hình, hay loại mẫu khó.
- Không dùng lại kết quả XGBoost 0,7188 từ benchmark cũ như bằng chứng sạch: audit
  phát hiện outer-validation từng được truyền làm `eval_set` cho early stopping.

## 9. Hướng thật sự còn mở

Ưu tiên theo giá trị kỳ vọng:

1. Mở rộng dữ liệu point-in-time cho franchise, director, writer, cast và hãng;
   dùng lịch sử ngoài 1.646 phim để giảm sparsity nhưng vẫn cắt theo ngày.
2. Giảm selection bias bằng cách thu thập mẫu phim đa dạng hơn thay vì chỉ top
   popularity; tạo benchmark phiên bản mới và không trộn với mốc 0,7195.
3. Feature kinh tế theo thời gian: budget điều chỉnh lạm phát, percentile budget
   theo năm/thị trường, được fit bằng dữ liệu quá khứ trong fold.
4. Feature cạnh tranh phát hành từ lịch công bố trước ngày chiếu: số phim cùng
   genre/ngôn ngữ/quốc gia trong cửa sổ thời gian; cần chứng minh timing.
5. Biểu diễn thực thể có regularization và coverage tốt hơn (out-of-fold frequency
   hoặc smoothed target encoding). Không dùng raw ID one-hot như thí nghiệm cũ.
6. Đánh giá temporal holdout hoặc rolling-origin sau khi khóa model để đo khả năng
   dự báo phim tương lai; không dùng kết quả này quay lại tuning cùng vòng.
7. Phân tích độ bất định/độ nhạy của nhãn gần ngưỡng 2; giữ nguyên target chính và
   chỉ báo cáo sensitivity analysis riêng.

## 10. Artifact và tài liệu nguồn sự thật

- Bundle: `models/xgboost_pre_release_operational_bundle.joblib`.
- Native model: `models/xgboost_pre_release_operational_model.json`.
- Manifest/checksum/feature list: `models/xgboost_pre_release_operational_manifest.json`.
- Code tái lập A+B: `src/reproduce_operational_ab_baseline.py`.
- Code đánh giá franchise: `src/evaluate_operational_franchise.py`.
- Code fit artifact: `src/train_final_xgboost.py`.
- Kết quả chính: `reports/tables/operational_franchise_metrics.csv` và
  `reports/tables/operational_franchise_oof_predictions.csv`.
- Diễn giải hiện hành: `docs/pre_release_operational_scope.md`,
  `docs/operational_franchise_history_findings.md`, `docs/xgboost_results.md`.
- Lịch sử tuning đầy đủ và quy tắc tránh lặp: `docs/tuning_registry.md`.

Các thử nghiệm cũ đã bị dọn khỏi nhánh hiện tại vẫn còn trong Git, chủ yếu ở commit
`76c6041` và các commit experiment trước `131d40f`. Không khôi phục chúng vào
pipeline chính nếu không có mục đích kiểm chứng cụ thể.
