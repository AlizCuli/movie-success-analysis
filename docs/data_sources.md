# Data Sources and Collection Scope

## Primary source

The final analysis uses the **TMDb Official API** as its sole data source. The
research snapshot was collected on 2026-07-17 (UTC), with up to 100 popular
movies per year from 2000 through 2025.

Primary endpoints:

- `/discover/movie`: selects movies by year and popularity order;
- `/movie/{movie_id}`: retrieves budget, revenue, runtime, genres, countries, and companies;
- `/movie/{movie_id}/credits`, `/release_dates`, and `/keywords`: supplies additional metadata within the defined operational scope.

Popularity, ratings, vote counts, and other post-release signals may remain in
raw API responses for source fidelity, but they are excluded from the main EDA
and from model predictors. `revenue` is used only to construct the target.

Source: [The Movie Database (TMDb)](https://www.themoviedb.org/).

> This product uses the TMDB API but is not endorsed or certified by TMDB.

## Sampling and validation

- Date range: 2000-01-01 through 2025-12-31.
- Maximum five discovery pages per year, equivalent to 100 movies per year.
- `sort_by=popularity.desc` is used for sample selection, not as a predictor.
- Adult and video titles are excluded.
- Movies are retained only when the actual release date falls within the target period.
- Duplicate records are removed by `tmdb_id`.

## Distribution policy

Raw, interim, processed, checkpoint, and per-movie prediction files remain local
and are excluded by `.gitignore`. The public repository provides source code,
directory structure, the packaged model, aggregate tables, and figures. Data
reproduction requires a valid TMDb API Read Access Token.

TMDb is updated continuously, so reruns at a later date may produce a different
snapshot and slightly different aggregate results.
