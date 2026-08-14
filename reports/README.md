# Report Assets

This directory contains public, aggregate report tables and reproducible figures for the Movie Success Analysis project. Row-level TMDb data, tokens, and per-movie predictions remain local and are excluded from Git.

## Recreate the Figure Set

After recreating the local TMDb artifacts, run this command from the repository root:

```bash
python src/generate_report_figures.py
```

The generator validates its required aggregate artifacts, then recreates the nine versioned PNG figures. It needs the following local files, which are intentionally not distributed in the public repository:

- `data/processed/movies_cleaned.csv`
- `data/processed/movies_modeling.csv`
- `data/raw/tmdb_movie_enrichment.jsonl`

## Figure Inventory

- `dataset_overview.png`: missingness in core TMDb fields and class balance in the modeling cohort.
- `pre_release_spearman_heatmap.png`: Spearman associations among the descriptive pre-release variables.
- `success_by_genre_collection.png`: observed success rates by primary genre and collection status.
- `tmdb_raw_feature_map.png`: the 15 TMDb metadata fields used before feature engineering.
- `success_rate_by_theatrical_release_breadth.png`: observed success rate by theatrical release breadth.
- `success_rate_by_release_breadth_and_collection.png`: release-breadth and collection interaction.
- `xgboost_performance_summary.png`: official nested outer-OOF metric summary.
- `xgboost_confusion_matrix.png`: pooled outer-OOF confusion matrix.
- `xgboost_fold_macro_f1.png`: Macro-F1 across outer validation folds.

## Public aggregate evidence

The theatrical-release figures are supported by two row-free tables generated
by `src/create_theatrical_release_figures.py`:

- [`success_by_theatrical_release_breadth.csv`](tables/success_by_theatrical_release_breadth.csv)
- [`success_by_release_breadth_and_collection.csv`](tables/success_by_release_breadth_and_collection.csv)

The tables under `reports/tables/` are public aggregates that support these
figures and the official benchmark. They are not substitutes for the
intentionally excluded row-level data.
