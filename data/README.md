# Local Data

The public repository does not distribute raw TMDb responses or row-level
derived data. Reproduction requires a valid TMDb API Read Access Token.

## Directory structure

- `raw/`: TMDb JSON/JSONL responses, enrichment snapshots, and checkpoints;
- `external/`: reserved for auxiliary sources outside the final scope;
- `interim/`: structured tables derived directly from TMDb responses;
- `processed/`: cleaned tables used for EDA and modeling.

## Main local artifacts

```text
data/raw/tmdb_movies_2000_2025.json
data/raw/tmdb_movie_enrichment.jsonl
data/interim/tmdb_movies_2000_2025.csv
data/processed/movies_cleaned.csv
data/processed/movies_modeling.csv
```

All content in these directories except `.gitkeep` is excluded by `.gitignore`.
The API token belongs only in the local `.env` file. TMDb data must not be
redistributed through this repository.
