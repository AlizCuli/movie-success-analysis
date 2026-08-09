"""Feature engineering chỉ dùng phạm vi pre-release operational."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELING_PATH = PROJECT_ROOT / "data" / "processed" / "movies_modeling.csv"
ENRICHMENT_PATH = PROJECT_ROOT / "data" / "raw" / "tmdb_movie_enrichment.jsonl"

FORBIDDEN_PREDICTORS = {
    "revenue",
    "log_revenue",
    "profit",
    "roi",
    "revenue_to_budget",
    "is_successful",
    "popularity",
    "log_popularity",
    "vote_average",
    "vote_count",
    "log_vote_count",
    "imdb_rating",
    "imdb_vote_count",
    "log_imdb_vote_count",
    "tmdb_id",
    "imdb_id",
    "title",
    "original_title",
}

BASE_NUMERIC = [
    "log_budget",
    "runtime",
    "release_year",
    "genre_count",
    "production_country_count",
    "production_company_count",
]
BASE_CATEGORICAL = [
    "release_month",
    "primary_genre",
    "original_language",
    "primary_country",
    "primary_company",
    "release_season",
    "release_decade",
]
METADATA_NUMERIC = [
    "is_collection",
    "company_count",
    "spoken_language_count",
    "cast_count",
    "crew_count",
    "theatrical_country_count",
    "release_event_count",
    "overview_word_count",
    "has_tagline",
    "keyword_count",
]
METADATA_CATEGORICAL = ["collection_id", "primary_company_id", "certification"]


def clean_pipe_values(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()


def read_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("Dữ liệu enrichment đang trống.")
    if frame["tmdb_id"].duplicated().any():
        raise ValueError("Dữ liệu enrichment có tmdb_id trùng.")
    return frame


def load_pre_release_modeling_data() -> tuple[pd.DataFrame, pd.Series]:
    modeling = pd.read_csv(MODELING_PATH)
    enrichment = read_jsonl(ENRICHMENT_PATH)
    overlapping = [
        column for column in enrichment.columns
        if column in modeling.columns and column != "tmdb_id"
    ]
    merged = modeling.merge(
        enrichment.drop(columns=overlapping),
        on="tmdb_id",
        how="left",
        validate="1:1",
    )
    if merged.empty or len(merged) != len(modeling) or merged["tmdb_id"].duplicated().any():
        raise ValueError("Tập modeling/enrichment phải có tmdb_id duy nhất và ghép 1:1.")
    if merged["retrieved_at"].isna().any():
        raise ValueError("Có phim modeling chưa được enrichment.")
    target = pd.to_numeric(merged["is_successful"], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("Target không hợp lệ.")
    return merged, target.astype(int)


def add_pre_release_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame["release_decade"] = (
        np.floor(pd.to_numeric(frame["release_year"], errors="coerce") / 10) * 10
    ).astype("Int64").astype(str).replace("<NA>", "__MISSING__")
    month = pd.to_numeric(frame["release_month"], errors="coerce")
    frame["release_season"] = pd.cut(
        month,
        bins=[0, 3, 6, 9, 12],
        labels=["Q1", "Q2", "Q3", "Q4"],
        include_lowest=True,
    ).astype(str).replace("nan", "__MISSING__")
    frame["primary_company"] = frame["production_companies"].map(
        lambda value: clean_pipe_values(value)[0]
        if clean_pipe_values(value)
        else "__MISSING__"
    )
    return frame


class PreReleaseFeatureBuilder:
    """Học vocabulary genre từ training fold và tạo đúng feature A+B."""

    def __init__(self) -> None:
        self.genre_vocabulary_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.feature_columns_: list[str] = []

    def fit(self, data: pd.DataFrame, target: pd.Series) -> "PreReleaseFeatureBuilder":
        del target  # Target không tham gia tạo feature A+B.
        source = add_pre_release_features(data)
        self.genre_vocabulary_ = sorted(
            {genre for value in source["genres"] for genre in clean_pipe_values(value)}
        )
        genre_columns = [f"genre_flag_{safe_name(genre)}" for genre in self.genre_vocabulary_]
        self.numeric_columns_ = BASE_NUMERIC + genre_columns + [
            "is_english",
            "is_us_production",
        ] + METADATA_NUMERIC
        self.categorical_columns_ = BASE_CATEGORICAL + METADATA_CATEGORICAL
        self.feature_columns_ = self.numeric_columns_ + self.categorical_columns_
        forbidden = sorted(set(self.feature_columns_) & FORBIDDEN_PREDICTORS)
        if forbidden:
            raise ValueError(f"Predictor bị cấm: {forbidden}")
        return self

    def _transform(self, data: pd.DataFrame) -> pd.DataFrame:
        source = add_pre_release_features(data)
        output = pd.DataFrame(index=source.index)
        for column in BASE_NUMERIC + METADATA_NUMERIC:
            output[column] = pd.to_numeric(source[column], errors="coerce")
        genre_sets = [set(clean_pipe_values(value)) for value in source["genres"]]
        for genre in self.genre_vocabulary_:
            output[f"genre_flag_{safe_name(genre)}"] = [
                int(genre in values) for values in genre_sets
            ]
        output["is_english"] = source["original_language"].fillna("").astype(str).eq("en").astype(int)
        country_sets = [set(clean_pipe_values(value)) for value in source["production_countries"]]
        output["is_us_production"] = [
            int("United States of America" in values) for values in country_sets
        ]
        for column in self.categorical_columns_:
            output[column] = source[column].fillna("__MISSING__").astype(str)
        output = output[self.feature_columns_]
        if np.isinf(output[self.numeric_columns_].to_numpy(dtype=float)).any():
            raise ValueError("Feature numeric có Inf.")
        return output.reset_index(drop=True)

    def fit_transform(self, data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        return self.fit(data, target)._transform(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._transform(data)
