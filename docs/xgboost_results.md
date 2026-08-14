# XGBoost Results Within the Pre-Release Scope

## Reference out-of-sample result

The reference configuration was evaluated with five outer and four inner
stratified folds on 1,646 movies. Pooled outer-OOF results are:

| Metric | Value |
| --- | ---: |
| Macro-F1 | **0.719483** |
| F1, class 0 | 0.605128 |
| F1, class 1 | 0.833837 |
| Recall, class 0 | 0.618449 |
| Recall, class 1 | 0.826347 |
| Balanced accuracy | 0.722398 |
| Accuracy | 0.766100 |

Outer folds were not used to select features, preprocessing, boosting rounds,
or the classification threshold. Per-movie predictions remain local; the
repository publishes aggregate tables only.

## Public report artifacts

- `xgboost_fold_metrics.csv`: performance across the five outer folds;
- `xgboost_confusion_matrix.csv`: pooled confusion matrix;
- `xgboost_pooled_metrics.csv`: pooled out-of-sample metrics;
- `xgboost_feature_importance.csv`: feature importance from the final fitted model.

The corresponding PNG figures are stored in `reports/figures/`. The generator
is `src/report_xgboost_results.py`, called as part of the public figure workflow.

## Packaged model

The model in `models/` is fit on all 1,646 movies after the reference
configuration was locked. It contains 144 boosting rounds and a classification
threshold of 0.51. Because this fit has seen the full cohort, its predictions
are not an independent generalization estimate; the pooled outer-OOF result is
the official benchmark.

## Interpretation and limitations

- The model identifies the successful class more accurately than the unsuccessful class; class-specific metrics should be read alongside Macro-F1.
- Franchise history improved Macro-F1 from 0.710977 to 0.719483 relative to the fixed A+B control, but the difference is modest and is not causal evidence.
- Results are affected by the maximum-100-popular-movies-per-year sampling strategy, missing financial metadata, and the snapshot nature of TMDb.
- Macro-F1 0.719483 indicates useful predictive signal in the defined metadata scope but does not support high-stakes financial decisions on its own.
