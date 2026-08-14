# Project Status

## Final scope

- **Topic:** Predicting movie financial success from information available before release.
- **Source:** TMDb Official API.
- **Target:** `revenue >= 2 × budget`.
- **Model:** XGBoost within the defined pre-release operational scope.
- **Evaluation:** 5 outer × 4 inner nested stratified cross-validation.
- **Primary metric:** Macro-F1.

## Dataset

- 2,597 TMDb movies from 2000–2025.
- 1,646 observations satisfy the modeling-data requirements.
- Class 0 (unsuccessful): 477 movies.
- Class 1 (successful): 1,169 movies.
- Raw, interim, and processed row-level files remain local and are excluded from Git.

## Reference benchmark

| Metric | Value |
| --- | ---: |
| Pooled outer-OOF Macro-F1 | **0.719483** |
| F1, class 0 | 0.605128 |
| Recall, class 0 | 0.618449 |
| Balanced accuracy | 0.722398 |
| Accuracy | 0.766100 |

The official model bundle contains 144 boosting rounds, a classification
threshold of 0.51, 51 engineered input features, and 160 transformed features.
The bundle is fit on the full modeling cohort after the reference configuration
was locked; its fit is not used as an independent generalization estimate.

## Completed components

- TMDb collection, validation, preprocessing, enrichment, EDA, evaluation, and training workflows.
- TMDb-only EDA notebook and reproducible report-figure generator.
- Official XGBoost bundle, native model, manifest, and checksums.
- Public aggregate tables and figures without row-level movie data.
- Repository contract tests that do not require private data.
- Documentation of feature scope, leakage controls, limitations, and rejected experiments.

## Known limitations

- The collection samples at most 100 popular movies per year and is not random.
- TMDb is a current snapshot rather than a complete point-in-time archive for every field.
- Missing financial metadata reduces the modeling cohort.
- The target threshold is a project-defined classification rule, not an accounting-profit measure.
- Class 0 has lower predictive performance than class 1.
- EDA associations are descriptive and do not establish causality.

## Current status

The reference configuration is frozen at Macro-F1 `0.719483`. Further modeling
work must preserve the current target, split protocol, leakage controls, and
benchmark artifacts and must be documented in [`docs/tuning_registry.md`](docs/tuning_registry.md).
