# Sổ đăng ký thí nghiệm và tuning

## Mục đích

File này ngăn việc GPT hoặc người phát triển lặp lại các thí nghiệm đã chạy. Mỗi
thử nghiệm mới phải trả lời: tín hiệu mới là gì, khác thí nghiệm cũ ở đâu, được
chọn bằng inner CV thế nào và có giữ nguyên outer evaluation hay không.

Các metric dưới đây không phải lúc nào cũng so sánh trực tiếp được. Kết quả từ
một train/test split, nested CV 5×5 và nested CV 5×4 được ghi rõ giao thức.

## 1. Linear Regression dự đoán doanh thu

Giao thức: split 80/20 seed 42; target `log1p(revenue)`; pipeline fit trên train;
5-fold CV bổ sung trên 1.316 train rows.

| Biến thể | Predictor | Test MAE | Test RMSE | Test R² |
| --- | ---: | ---: | ---: | ---: |
| Median baseline | 0 | 199,69 triệu USD | 339,25 triệu USD | -0,122 |
| Simple | 1 (`log_budget`) | 141,54 triệu USD | 258,69 triệu USD | 0,347 |
| Pre-release | 10 | 143,79 triệu USD | 245,96 triệu USD | 0,410 |
| Extended | 15 | 113,38 triệu USD | 204,08 triệu USD | 0,594 |

Extended thêm popularity/vote/rating nên không hợp lệ cho dự báo pre-release.
Linear Regression đã hoàn thành vai trò học thuật; không tiếp tục tuning vì phạm
vi cuối đã chốt XGBoost classification.

## 2. k-NN ban đầu

Giao thức: một split 80/20 stratified seed 42; GridSearchCV 5-fold chỉ trên train.

- Pre-release 10 predictor: `log_budget`, runtime, release year/month, genre count,
  production country/company count, primary genre/language/country.
- Extended thêm 5 tín hiệu hậu phát hành: log popularity, TMDb rating/log votes,
  IMDb rating/log votes.
- Grid: `k=3..31`, weights uniform/distance, metric Euclidean/Manhattan.
- Best pre-release: `k=9`, uniform, Euclidean; CV 0,5407; test Macro-F1 0,5402.
- Best extended: `k=5`, distance, Manhattan; CV 0,6266; test 0,5995;
  recall lớp 0 chỉ 0,3125.
- Most-frequent baseline test Macro-F1 0,4149; default pre-release 0,5691.

Kết luận: grid tuning không cứu được representation và mất cân bằng; extended sai
phạm vi hiện tại. Không chạy lại grid này.

## 3. Audit và tối ưu k-NN

Giao thức: nested CV outer 5 seed 42, inner 5 seed 43; 80 random configurations
mỗi outer fold; imputer/scaler/encoder/RandomOverSampler nằm trong pipeline;
threshold chọn bằng inner OOF.

Đã thử:

- original 15 predictor và compact 14 predictor;
- StandardScaler, RobustScaler, MinMaxScaler trong audit training-only;
- k, uniform/distance, Euclidean/Manhattan;
- `min_frequency` 5/10/20/30;
- trọng số riêng numeric/categorical;
- RandomOverSampler strategy 0,75 hoặc 1,00;
- threshold 0,20–0,80;
- primary genre so với multi-hot genres.

| Cấu hình | Macro-F1 outer OOF | F1 lớp 0 | Recall lớp 0 | SD outer |
| --- | ---: | ---: | ---: | ---: |
| Fixed original, threshold 0,50 | 0,6099 | 0,4016 | 0,3229 | 0,0209 |
| Optimized original, threshold 0,50 | **0,6540** | 0,5256 | 0,5702 | 0,0324 |
| Optimized original, tuned threshold | 0,6518 | 0,5240 | 0,5723 | 0,0334 |
| Optimized compact, threshold 0,50 | 0,6508 | 0,5168 | 0,5493 | **0,0103** |
| Optimized compact, tuned threshold | 0,6479 | 0,5174 | 0,5618 | 0,0171 |

Compact được chọn vì ổn định dù thấp hơn original 0,0032. Threshold tuning không
generalize; RandomOverSampler được chọn 5/5 fold; tăng nhẹ trọng số numeric thường
có ích. Không lặp lại scaler/oversampling/weight/threshold sweep này cho k-NN.

## 4. Logistic Regression và Random Forest

Giao thức: cùng nested outer 5/inner 5 và compact 14 predictor hậu phát hành.

| Model | Cấu hình báo cáo | Macro-F1 outer OOF |
| --- | --- | ---: |
| k-NN compact | threshold 0,50 | 0,6508 |
| Logistic Regression | tuned threshold | 0,6855 |
| Random Forest | threshold 0,50 | **0,6870** |

Logistic threshold tuning có ích; Random Forest threshold 0,50 tốt hơn tuned.
Bootstrap cho thấy Logistic/Random Forest hơn k-NN, nhưng RF và Logistic không
khác rõ. Các model này là đối chứng lịch sử, không còn thuộc phạm vi phát triển.

## 5. Boosted classification benchmark

Giao thức dự kiến: nested outer 5/inner 4; 40 Optuna trials cho mỗi model và mỗi
outer fold; threshold 0,20–0,80; ensemble weight bước 0,25.

Feature set đã thử:

- compact: 10 pre-release + popularity/audience rating/total vote;
- interaction: thêm votes-per-budget, popularity-per-budget, rating engagement,
  vote-source gap và release decade;
- rich: thêm genre multi-hot, primary company, English/US flags, release season.

| Model | Macro-F1 outer OOF | Ghi chú |
| --- | ---: | --- |
| CatBoost default | 0,7052 | Hậu phát hành |
| CatBoost tuned | 0,7003 | Tuning làm giảm kết quả |
| XGBoost default | 0,7139 | Hậu phát hành |
| XGBoost tuned | 0,7188 | Hậu phát hành; quy trình có lỗi early stopping |
| Soft-voting ensemble | 0,7107 | Không cải thiện |

Audit sau đó phát hiện outer-validation đã được dùng làm `eval_set` cho early
stopping ở benchmark cũ (`de1e537`). Vì vậy 0,7188 không phải bằng chứng sạch.
Không lặp lại 40–50 trial/fold trên cùng feature space, không dùng ensemble này,
và không dùng điểm này làm baseline pre-release.

Search space XGBoost rộng đã quét ở chặng này:

```text
n_estimators: 200..1800, step 100
learning_rate: 0.01..0.20 (log)
max_depth: 2..9; min_child_weight: 1..20
subsample: 0.55..1.00; colsample_bytree: 0.55..1.00
gamma: 0..6; reg_alpha: 1e-4..10 (log); reg_lambda: 0.1..30 (log)
class_weight_0: 1.0, 1.25, 1.5, 2.0, 2.5, 3.0
```

## 6. Enrichment và feature-family ablation

Giao thức: outer 5/inner 4; ablation dùng cấu hình cố định; XGBoost và CatBoost
tối đa 25 Optuna trials/outer fold, có pruning và inner-only early stopping.
Đã hoàn thành khoảng 114–116 trial/model, không mở rộng thêm 15 trial vì 5 trial
cuối không cải thiện đủ ổn định.

Các family:

- A: feature nền;
- B: collection, company/country/language, cast/crew count, certification,
  release-event, overview/tagline/keyword count;
- C: ID director/writer/producer/composer/top cast;
- D: TF-IDF overview/tagline và keyword multi-hot;
- E: history point-in-time director/company/cast/team;
- F: tổ hợp được chọn.

Kết quả quan trọng:

- XGBoost hậu phát hành A+B: Macro-F1 **0,7597**, F1 lớp 0 0,6597, recall lớp 0
  0,6646. Đây là điểm cao nhất tuyệt đối nhưng dùng votes/ratings/popularity.
- CatBoost hậu phát hành: 0,7525; ensemble: 0,7489.
- XGBoost pre-release-like lịch sử: 0,7144 ± 0,0263, F1 lớp 0 0,5973,
  recall lớp 0 0,6080.
- Text TF-IDF + keywords: 0,5772, giảm mạnh.
- Identity/history coverage thưa; khoảng 14% category unseen ở pre-release có ID.
- Learning curve hậu phát hành tăng từ 0,6600 ở 25% lên 0,7046 ở 100%; phần tăng
  từ 75% lên 100% còn 0,0086, gợi ý thêm dữ liệu/tín hiệu tốt hơn thêm trial.
- 207 quan sát bị cả bốn model phân loại sai; lỗi tập trung gần ngưỡng target.

Không lặp lại raw identity one-hot, cùng TF-IDF hay cùng history E nếu coverage và
cách biểu diễn không đổi.

Search space XGBoost thu hẹp đã quét ở chặng enrichment:

```text
n_estimators: 400..1200, step 100
learning_rate: 0.03..0.10 (log)
max_depth: 2..6; min_child_weight: 3..16
subsample: 0.60..0.90; colsample_bytree: 0.75..1.00
gamma: 0.5..3.0; reg_alpha: 1e-3..1.0; reg_lambda: 0.3..5.0
class_weight_0: 1.25, 1.5, 2.0
min_frequency: 5, 10, 20; history smoothing: 5, 10, 20
threshold: 0.20..0.80, step 0.01
```

## 7. Strict pre-release conservative

11 predictor bảo thủ: budget log, runtime, năm/tháng/quý, genre count + full genre,
ngôn ngữ, số quốc gia + quốc gia chính, số hãng. Không dùng enrichment thực thể.

Giao thức: outer 5/inner 4; tối đa 20 Optuna trial/study; class-0 weight 1–4;
threshold 0,20–0,80; không oversampling.

- Threshold 0,50: Macro-F1 0,6053.
- Inner-tuned threshold: **0,6125**; F1 lớp 0 0,4569; recall lớp 0 0,4717;
  balanced accuracy 0,6148.

Đây là mốc chống leakage bảo thủ, không phải baseline operational chính.

Search space strict đã quét: 100–800 cây, learning rate 0,01–0,20, depth 2–6,
min child weight 1–15, subsample/colsample 0,60–1,00, gamma 0–5, alpha
`1e-4`–10, lambda `1e-3`–30, class-0 weight 1–4 và threshold 0,20–0,80.

## 8. Strict feature engineering mở rộng

Đã thử metadata, point-in-time history, frequency encoding và interactions trong
nested 5×4, vẫn loại toàn bộ votes/ratings/popularity và biến doanh thu.

- Macro-F1 pooled outer OOF: **0,6706**.
- Outer mean ± SD: 0,6698 ± 0,0108.
- F1 lớp 0: 0,5333; F1 lớp 1: 0,8079.
- Metadata là block mạnh nhất; metadata + frequency có inner mean 0,6719.
- History/interactions chỉ được giữ ở fold có cải thiện inner ổn định.

Source và bảng của thí nghiệm này từng chưa commit rồi bị dọn khi rút gọn project;
vì vậy đây là mốc lịch sử, hiện chưa tái lập trực tiếp. Không tuyên bố nó là kết
quả hiện hành nếu chưa khôi phục đúng code và artifacts.

## 9. Pre-release operational A+B

Phạm vi operational chấp nhận metadata TMDb hiện có nhưng không khẳng định strict
historical snapshot.

- A+B cố định, cùng 1.646 ID và outer folds: Macro-F1 **0,710977**.
- Cấu hình khóa: learning rate 0,064; depth 4; min child weight 12; subsample
  0,68; colsample 0,88; gamma 1,6; alpha 0,14; lambda 0,93; class-0 weight 1,5;
  one-hot `min_frequency=10`.

Đây là direct ablation để đo đóng góp feature mới, không nên retune lại mỗi lần.

## 10. Operational v2: block G/H

Mỗi study tối đa 20 Optuna trial; so sánh A+B, A+B+G và A+B+G+H.

- G: multi-hot country/company/keyword, log/count ratio cho cast/crew/keyword/
  release/theatrical và completeness flags.
- H: lịch sử point-in-time director/company/cast/team.
- G được chọn ở outer folds 1, 3, 5; A+B ở folds 2, 4; H không được chọn fold nào.
- Macro-F1 cuối: **0,7087**, F1 lớp 0 0,5864, recall lớp 0 0,5870, thấp hơn mốc
  0,7144 khoảng 0,0057.

Kết luận: bác bỏ phiên bản này. Không chạy lại đúng block G/H với cùng dữ liệu.

Search space v2 đã quét: 80–500 cây bước 20, learning rate 0,02–0,12, depth
2–6, min child weight 2–16, subsample 0,60–0,95, colsample 0,60–1,00,
gamma 0–4, alpha `1e-4`–2, lambda 0,05–15, class-0 weight 1–2,8,
`min_frequency` 5/10/15, history smoothing 5/10/20 và threshold 0,20–0,80.

## 11. Franchise history point-in-time

Đã bổ sung bốn feature collection:

- số phim trước đó trong collection;
- tỷ lệ thành công lịch sử đã smoothing;
- mean log-budget trước đó;
- số năm từ phần phim trước.

Trong 1.646 phim, 709 có collection và 518 thuộc collection có ít nhất hai phim
trong mẫu. History chỉ dùng phim phát hành sớm hơn và fit trong training fold.

| Mô hình | Macro-F1 | Delta so với A+B |
| --- | ---: | ---: |
| A+B fixed | 0,710977 | — |
| A+B + franchise history | **0,719483** | **+0,008505** |

Đây là benchmark chính thức. Outer-fold Macro-F1 lần lượt 0,7603; 0,7166; 0,7188;
0,7133; 0,6874. Threshold inner OOF lần lượt 0,56; 0,54; 0,53; 0,51; 0,57.

## 12. Ma trận “đã thử / chưa thử”

| Hướng | Trạng thái | Quyết định |
| --- | --- | --- |
| Budget/runtime/calendar/basic categorical | Đã dùng | Giữ |
| Genre multi-hot | Đã dùng | Giữ |
| Collection/franchise point-in-time | Đã dùng, có gain | Giữ và mở rộng coverage |
| Cast/crew/company counts, certification, release events | Đã dùng | Giữ trong operational |
| Company/country/keyword multi-hot + ratios | Đã thử v2, không gain | Không lặp nguyên trạng |
| Director/cast/writer/company/team history | Đã thử sparse, không ổn định | Chỉ thử lại với dữ liệu rộng/encoding mới |
| Raw identity one-hot | Đã thử, unseen cao | Không lặp nguyên trạng |
| TF-IDF overview/tagline + keywords | Đã thử, 0,5772 | Không lặp nguyên trạng |
| Frequency encoding | Đã thử trong strict FE | Chỉ thử lại khi định nghĩa/coverage đổi |
| Generic smoothed target encoding | Chưa có bằng chứng tái lập hoàn chỉnh | Có thể thử fold-local |
| Inflation-adjusted/relative budget | Chưa triển khai đầy đủ | Hướng mới |
| Release competition | Chưa triển khai | Hướng mới, cần timing |
| Wider historical entity data | Chưa có | Ưu tiên cao |
| Temporal/rolling-origin evaluation | Chưa có | Cần làm sau khi khóa model |
| More diverse sampling | Chưa có | Ưu tiên cao, tạo benchmark version mới |
| Probability calibration chuyên biệt | Chưa có | Ưu tiên thấp hơn feature/data |
| SHAP/stability analysis | Chưa có | Dùng diễn giải, không tự tăng score |

## 13. Quy tắc cho lần tuning tiếp theo

1. Ghi trước feature block, nguồn dữ liệu và lý do nó tồn tại trước phát hành.
2. So sánh trực tiếp với A+B cố định; không retune control.
3. Chỉ chọn block bằng inner CV trên từng outer-training partition.
4. Chỉ khi feature set đã khóa mới tuning XGBoost và threshold trong inner CV.
5. Outer validation chỉ báo cáo cuối; không dùng để thêm/bớt feature hay trial.
6. Ghi rõ delta so với 0,710977 và benchmark 0,719483, độ ổn định giữa fold,
   F1/recall lớp 0 và confusion matrix.
7. Nếu giả thuyết, dữ liệu và representation giống một mục đã bác bỏ ở trên thì
   dừng trước khi chạy và giải thích vì sao không có thông tin mới.
8. Nếu đổi tập dữ liệu/sampling, lập benchmark version mới; không so điểm như thể
   cùng một protocol.
9. Không ghi đè model/bảng hiện hành. Mọi thí nghiệm mới dùng tên/version riêng.

## 14. Vị trí bằng chứng lịch sử

- Commit `b68b927`: k-NN ban đầu.
- Commit `f0cc839`: tối ưu k-NN.
- Commit `e5f8a96`: Logistic Regression và Random Forest.
- Commit `de1e537`: boosted benchmark; có lỗi outer eval-set đã audit.
- Commit `5809ff6`: enrichment experiments.
- Commit `76c6041`: snapshot chứa phần lớn docs, bảng và dữ liệu tái lập cũ.
- Commit `131d40f`: rút gọn project về XGBoost pre-release.

Git history là nguồn tham khảo, không tự động biến kết quả cũ thành benchmark hợp
lệ. Khi trích lại một số liệu, phải kèm scope và protocol của nó.

## 15. Operational Budget Context V1

Ngày chạy: 2026-08-05. Đây là thí nghiệm feature engineering đầu tiên sau khi khóa
benchmark 0,719483. Không tuning hyperparameter và không dùng outer-validation.

### Giả thuyết và representation

Block thêm bảy feature fold-local vào A+B + franchise history:

- percentile budget so với toàn bộ budget lịch sử;
- chênh lệch log-budget so với median lịch sử toàn kỳ và ba năm;
- log số quan sát budget lịch sử toàn kỳ và ba năm;
- hai cờ availability.

Reference chỉ gồm phim trong training partition có `release_date` nhỏ hơn ngày
của phim đang transform. Validation không cập nhật state; target không tham gia
budget feature; ngày bằng nhau cũng bị loại. Sáu unit tests chống leakage,
fallback và reproducibility đều đạt.

Coverage outer-validation rất cao: 99,39–100% có ít nhất một budget lịch sử;
95,74–98,78% có ít nhất 30 quan sát. Median history toàn kỳ theo fold là
595–692,5 phim và median history ba năm là 156–161 phim.

### Inner screening đã khóa trước khi chạy

Điều kiện để được phép xem outer result:

- mean inner Macro-F1 delta ít nhất +0,003;
- thắng control ở ít nhất 3/5 outer-training partitions;
- mean recall lớp 0 delta không thấp hơn -0,01.

| Outer-training partition | Control inner Macro-F1 | Candidate | Delta | Recall lớp 0 delta |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0,710459 | 0,705854 | -0,004606 | -0,044619 |
| 2 | 0,706543 | 0,711671 | +0,005128 | +0,068063 |
| 3 | 0,707954 | 0,705549 | -0,002406 | +0,036649 |
| 4 | 0,705245 | 0,705944 | +0,000699 | +0,007853 |
| 5 | 0,715913 | 0,723400 | +0,007487 | -0,055118 |

Mean Macro-F1 delta chỉ **+0,001261**, dù candidate thắng 3/5 partition. Mean
recall lớp 0 delta là +0,002566. Vì không đạt mức tăng +0,003, block bị
**rejected by inner gate** và outer evaluation không được chạy. Không có outer
OOF score, confusion matrix hoặc tuyên bố so sánh mới với 0,719483.

Kết luận: coverage không phải nút thắt, nhưng global percentile/median và cửa sổ
ba năm ở representation này không tạo tín hiệu inner đủ mạnh. Không lặp lại đúng
bảy feature và công thức này; thử nghiệm tiếp theo phải thay đổi giả thuyết, ví dụ
budget tương đối theo nhóm có coverage cao thay vì chỉ theo toàn thị trường.

Artifacts: `operational_budget_v1_feature_schema.csv`,
`operational_budget_v1_coverage.csv`, `operational_budget_v1_inner_screening.csv`
và `operational_budget_v1_run_metadata.json`. Runtime screening: 43,52 giây.

## 16. Entity History Enrichment V1

Ngày thu thập và chạy: 2026-08-06. Giả thuyết mới so với block H cũ là mở rộng
kho lịch sử TMDb bên ngoài 1.646 target movies để giảm unseen và tăng số quan
sát quá khứ cho director, production company và top-5 cast. Distributor được
audit riêng và bị chặn; không dùng production company làm proxy.

### Scope và protocol

- Control cố định: `A+B + franchise history`.
- Outer assignments, seed và XGBoost parameters được đối chiếu bằng SHA-256 với
  artifact benchmark trước khi chạy.
- Mỗi block được sàng lọc độc lập bằng 4-fold inner OOF trên cả năm
  outer-training partition.
- Chỉ history có ngày phát hành trước query date; outcome history dùng maturity
  lag 365 ngày; target ID luôn bị loại; validation không cập nhật state.
- Mỗi nhóm có 18 feature về count, coverage, experience, recency, smoothed
  success, historical log-budget, recent form và time-weighted form.
- Không đưa lại bảy feature Operational Budget Context V1.

### Dữ liệu và coverage

Snapshot `tmdb-2026-08-05-v1` có 134.254 phim, 8.022 director edges, 98.788
production-company edges và 162.144 top-cast edges. Coverage có ít nhất một
history trung bình lần lượt là 94,17%, 99,45% và 99,82%; target leakage, khóa
trùng và ngày ngoài phạm vi đều bằng 0. Ba movie-detail ID trả HTTP 404 được ghi
nhận là lỗi không thể phục hồi.

### Inner screening

| Block | Inner Macro-F1 | Mean delta | Wins | Recall lớp 0 delta | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Top-5 cast | 0,707323 | -0,001900 | 2/5 | -0,025714 | Fail |
| Director | 0,706383 | -0,002840 | 1/5 | +0,035113 | Fail |
| Production company | 0,706266 | -0,002957 | 1/5 | -0,012120 | Fail |

Gate yêu cầu mean delta ít nhất +0,003, thắng ít nhất 3/5 partition và mean
recall lớp 0 delta không thấp hơn -0,01. Không block nào đạt, nên không có tổ
hợp được tạo và outer-validation không chạy. Benchmark `0.719483` không bị ghi
đè. Runtime inner screening: 1.463,39 giây.

Quyết định: bác bỏ representation V1 hiện tại. Không lặp lại đúng 18 feature mỗi
nhóm với smoothing 10, recent-3, half-life 3 năm và maturity lag 365 ngày nếu
không có giả thuyết hoặc representation mới.

Artifacts: `entity_history_v1_feature_schema.csv`,
`entity_history_v1_inner_screening.csv`, `entity_history_v1_inner_gate_summary.csv`,
`entity_history_v1_run_metadata.json`, các bảng audit `entity_history_*` và
`docs/entity_history_findings.md`.
