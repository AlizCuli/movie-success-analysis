# Preprocessing Decisions

## Input scope

The input is the structured TMDb CSV at
`data/interim/tmdb_movies_2000_2025.csv`. The source file is never edited in
place.

## Missing and invalid values

- Empty strings and `\N` are converted to missing values.
- `budget <= 0`, `revenue <= 0`, and `runtime <= 0` are treated as unavailable or invalid and converted to missing values.
- Ratings outside 0–10 are converted to missing for raw-data validity; rating fields are excluded from the final EDA and model.
- The processed CSV is not filled with a mean or median.
- Predictive missing values are handled by the model pipeline, whose imputer is fit only on the relevant training partition.

`movies_cleaned.csv` retains all 2,597 movies. `movies_modeling.csv` retains
movies with valid budget, revenue, runtime, release date, and target values;
other predictors may remain missing and are handled inside the modeling
pipeline.

## EDA features

- Time: `release_year`, `release_month`, and `release_quarter`.
- Descriptive financial scale: `profit`, `revenue_to_budget`, `roi`, `log_budget`, and `log_revenue`.
- Categories and counts: `genre_count`, `primary_genre`, `production_country_count`, `primary_country`, and `production_company_count`.
- Quality flags: `budget_available`, `revenue_available`, `budget_outlier`, and `revenue_outlier`.

Post-release financial fields are used only to construct or describe the target;
they are not model predictors.

## Outliers

`budget_outlier` and `revenue_outlier` use the IQR rule outside
`[Q1 − 1.5 × IQR, Q3 + 1.5 × IQR]`. Outliers are flagged rather than removed
because they may represent genuinely different production scales.

## Target definition

For movies with valid budget and revenue:

```text
is_successful = 1 if revenue >= 2 * budget
is_successful = 0 if revenue < 2 * budget
```

This is the operational financial-success criterion defined for the study. It
is not an accounting-profit measure because marketing, exhibitor revenue
sharing, distribution costs, and other income are not modeled.

## Target-leakage controls

The columns `revenue`, `log_revenue`, `profit`, `roi`, and `revenue_to_budget`
are excluded from predictive features. `is_successful` is the target only.
Popularity, ratings, and vote counts are also excluded because their
availability before release is not guaranteed.
