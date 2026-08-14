# Pre-Release Operational Scope

## Scope definition

The project uses TMDb fields that can represent information available before
release: budget, runtime, release timing, genres, language, countries,
production companies, collection membership, certification, cast and crew
counts, release metadata, overview length, tagline, and keywords. This is an
**operational** pre-release scope because the current TMDb snapshot is not a
historical archive proving when every field became public.

`revenue` is used only to construct the target. Popularity, ratings, vote
counts, profit, ROI, and all revenue-derived variables are excluded from the
predictors.

## Final feature groups

- **Basic metadata:** `log_budget`, runtime, release year/month/season, genres, language, country, and production company.
- **TMDb enrichment:** collection status, company and language counts, cast and crew counts, certification, theatrical-country and release-event counts, overview length, tagline flag, and keyword count.
- **Time-aware franchise history:** prior movie count, smoothed historical success rate, historical mean `log_budget`, and years since the previous release.

## Leakage-safe protocol

1. Keep the 1,646-film cohort and the five outer `StratifiedKFold` partitions with seed 42 fixed.
2. Use four inner stratified folds with seed 43 for boosting-round and threshold selection.
3. Fit imputation, encoding, categorical vocabulary, and history construction only on the relevant training partition.
4. For franchise history, use only movies released earlier than the query movie; movies released on the same date do not create history for one another.
5. Reserve outer folds for final evaluation and do not reuse them for feature or parameter selection.

## Reference benchmark

The XGBoost configuration with franchise history achieved pooled outer-OOF
Macro-F1 **0.719483**, class-0 F1 **0.605128**, class-0 recall **0.618449**, and
balanced accuracy **0.722398**.
