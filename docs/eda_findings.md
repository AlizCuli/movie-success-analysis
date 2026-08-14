# TMDb Exploratory Data Analysis Findings

## Data quality and sample coverage

- The cleaned dataset contains 2,597 movies; the modeling cohort contains 1,646 movies with valid budget, revenue, runtime, release date, and target values.
- Budget is missing for 855 movies (32.92%), revenue for 839 (32.31%), and runtime for 72 (2.77%). See [`core_missingness.csv`](../reports/tables/core_missingness.csv).
- The modeling cohort contains 477 unsuccessful movies (28.98%) and 1,169 successful movies (71.02%).

## Descriptive relationships

- `budget` and `revenue` are right-skewed, with medians below their means. `log1p` transformations are used for descriptive scale comparisons and for the budget representation used by the model.
- Individual pre-release variables have weak-to-moderate Spearman associations with the target. The full matrix is available in [`pre_release_spearman.csv`](../reports/tables/pre_release_spearman.csv).
- The largest basic associations with `is_successful` are `is_collection` (ρ = 0.283), `theatrical_country_count` (ρ = 0.245), and `release_event_count` (ρ = 0.192).
- Observed success rates differ across primary genres and collection status. Group sizes are shown in the figure and in [`success_by_primary_genre_collection.csv`](../reports/tables/success_by_primary_genre_collection.csv).
- Observed success rates range from 28.6% in the 0–5-market theatrical-release group to 77.7% in the more-than-30-market group. The public aggregate evidence is [`success_by_theatrical_release_breadth.csv`](../reports/tables/success_by_theatrical_release_breadth.csv).
- Yearly rates vary within the sample; the collection strategy is limited to at most 100 popular movies per year and is not representative of the full market.

## Reproducible outputs

The EDA generator is `src/eda_movies.py`. The complete nine-figure set is
recreated by `src/generate_report_figures.py`.

- `reports/figures/dataset_overview.png`
- `reports/figures/pre_release_spearman_heatmap.png`
- `reports/figures/success_by_genre_collection.png`

Corresponding aggregate tables are stored in `reports/tables/`.

## Interpretation limits

- Correlations and group rates describe associations in the sampled snapshot and do not establish causality.
- Monetary values are nominal and are not inflation-adjusted.
- The current TMDb snapshot does not prove the historical publication date of every metadata field.
- `revenue` is used to construct the target and describe the realized outcome; it is not a predictive feature.
