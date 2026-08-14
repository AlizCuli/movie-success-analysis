# Packaged Model Input Schema

`src/predict_xgboost.py` accepts a structured CSV of movie metadata. It does not
accept raw TMDb JSON directly. Each row represents one movie.

## Required fields

### Preprocessed movie fields

`log_budget`, `runtime`, `release_year`, `release_month`, `genre_count`,
`primary_genre`, `original_language`, `production_country_count`,
`primary_country`, `production_company_count`, `genres`,
`production_countries`, `production_companies`, and `release_date`.

### TMDb enrichment fields

`is_collection`, `collection_id`, `primary_company_id`, `company_count`,
`spoken_language_count`, `cast_count`, `crew_count`, `certification`,
`theatrical_country_count`, `release_event_count`, `overview_word_count`,
`has_tagline`, and `keyword_count`.

`tmdb_id` is optional and is copied to the output for identification. The input
must not contain the target or post-release variables as predictive fields.

## Missing values

Missing values may remain blank or be represented as `NaN`. The packaged
pipeline contains the numeric imputer and categorical encoder; these components
were fit without using statistics from the row being predicted.

## Franchise-history limitation

The bundle includes the reference history state from the training snapshot. At
inference time, the four franchise-history features use only reference movies
released before the input movie. The history state is not updated with other
rows in the same input file. Adding new history requires retraining under the
point-in-time protocol.
