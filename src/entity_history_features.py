"""Fold-local point-in-time feature blocks for entity history enrichment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.entity_history_schema import (
    CAST_PATH,
    COMPANIES_PATH,
    DIRECTORS_PATH,
    load_config,
    read_jsonl,
    target_entity_relations,
)
from src.operational_franchise_features import OperationalFranchiseBuilder
from src.pre_release_features import FORBIDDEN_PREDICTORS


HISTORY_PATHS = {
    "director": DIRECTORS_PATH,
    "production_company": COMPANIES_PATH,
    "cast": CAST_PATH,
}

OPERATIONAL_BUDGET_V1_BLACKLIST = {
    "budget_prior_percentile",
    "log_budget_minus_prior_median",
    "log_prior_budget_count",
    "log_budget_minus_prior_3y_median",
    "log_prior_budget_3y_count",
    "budget_history_available",
    "budget_history_3y_available",
}

ENTITY_METRICS = (
    "prior_movie_count",
    "valid_financial_count",
    "years_experience",
    "years_since_last",
    "smoothed_success_rate",
    "median_log_budget",
    "recent_success_rate",
    "recency_weighted_success_rate",
)


def block_columns(group: str) -> list[str]:
    prefix = "company" if group == "production_company" else group
    return [
        f"{prefix}_entity_count",
        f"{prefix}_entities_with_history",
        f"{prefix}_history_available",
        f"{prefix}_history_coverage_rate",
        f"{prefix}_prior_movie_count_mean",
        f"{prefix}_prior_movie_count_max",
        f"{prefix}_valid_financial_count_mean",
        f"{prefix}_valid_financial_count_max",
        f"{prefix}_years_experience_mean",
        f"{prefix}_years_experience_max",
        f"{prefix}_years_since_last_mean",
        f"{prefix}_years_since_last_min",
        f"{prefix}_smoothed_success_rate_mean",
        f"{prefix}_smoothed_success_rate_max",
        f"{prefix}_median_log_budget_mean",
        f"{prefix}_median_log_budget_max",
        f"{prefix}_recent_success_rate_mean",
        f"{prefix}_recency_weighted_success_rate_mean",
    ]


@dataclass(frozen=True)
class HistorySettings:
    maturity_lag_days: int = 365
    smoothing: float = 10.0
    recent_movie_count: int = 3
    half_life_years: float = 3.0
    top_cast_k: int = 5

    @classmethod
    def from_config(cls) -> "HistorySettings":
        config = load_config()
        return cls(
            maturity_lag_days=int(config["maturity_lag_days"]),
            smoothing=float(config["success_smoothing"]),
            recent_movie_count=int(config["recent_movie_count"]),
            half_life_years=float(config["recency_half_life_years"]),
            top_cast_k=int(config["top_cast_k"]),
        )


@lru_cache(maxsize=3)
def _load_external_history(path: Path) -> pd.DataFrame:
    records = read_jsonl(path)
    if not records:
        raise FileNotFoundError(f"Entity history is missing or empty: {path}")
    frame = pd.DataFrame(records)
    required = {
        "movie_tmdb_id", "entity_id", "release_date", "budget", "revenue",
        "financial_pair_valid",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Entity history schema is missing: {missing}")
    frame["movie_tmdb_id"] = pd.to_numeric(frame["movie_tmdb_id"], errors="raise").astype(int)
    frame["entity_id"] = pd.to_numeric(frame["entity_id"], errors="raise").astype(int)
    frame["release_date"] = pd.to_datetime(frame["release_date"], errors="coerce")
    frame["budget"] = pd.to_numeric(frame["budget"], errors="coerce")
    frame["revenue"] = pd.to_numeric(frame["revenue"], errors="coerce")
    frame = frame[frame["release_date"].notna()].copy()
    frame["success"] = np.where(
        frame["budget"].gt(0) & frame["revenue"].gt(0),
        frame["revenue"].ge(2.0 * frame["budget"]).astype(float),
        np.nan,
    )
    return frame.drop_duplicates(["movie_tmdb_id", "entity_id"]).reset_index(drop=True)


class EntityHistoryBlockBuilder:
    """Create one entity block from external history and fold-local target rows."""

    def __init__(self, group: str, settings: HistorySettings | None = None) -> None:
        if group not in HISTORY_PATHS:
            raise ValueError(f"Unsupported or blocked entity group: {group}")
        self.group = group
        self.settings = settings or HistorySettings.from_config()
        self.reference_: pd.DataFrame | None = None
        self.reference_by_entity_: dict[int, pd.DataFrame] = {}
        self.global_mature_dates_: np.ndarray = np.asarray([], dtype="datetime64[ns]")
        self.global_mature_success_prefix_: np.ndarray = np.asarray([], dtype=float)
        self.relation_by_movie_: dict[int, list[tuple[int, int | None]]] = {}
        self.numeric_columns_ = block_columns(group)

    def _relations(self) -> pd.DataFrame:
        relations = target_entity_relations(self.settings.top_cast_k)[self.group]
        frame = pd.DataFrame(relations)
        frame["target_tmdb_id"] = frame["target_tmdb_id"].astype(int)
        frame["entity_id"] = frame["entity_id"].astype(int)
        return frame

    def fit(self, data: pd.DataFrame, target: pd.Series) -> "EntityHistoryBlockBuilder":
        relations = self._relations()
        all_relation_by_movie: dict[int, list[tuple[int, int | None]]] = {}
        for movie_id, group in relations.groupby("target_tmdb_id"):
            values = []
            for _, row in group.sort_values("billing_order", na_position="last").iterrows():
                order = None if pd.isna(row["billing_order"]) else int(row["billing_order"])
                values.append((int(row["entity_id"]), order))
            all_relation_by_movie[int(movie_id)] = values
        self.relation_by_movie_ = all_relation_by_movie

        train = data.reset_index(drop=True).copy()
        train_target = target.reset_index(drop=True).astype(int)
        train["release_date"] = pd.to_datetime(train["release_date"], errors="coerce")
        train["budget"] = pd.to_numeric(train["budget"], errors="coerce")
        train["revenue"] = pd.to_numeric(train["revenue"], errors="coerce")
        if train["release_date"].isna().any():
            raise ValueError("Entity history requires valid training release dates.")

        training_rows: list[dict[str, Any]] = []
        for position, movie in train.iterrows():
            movie_id = int(movie["tmdb_id"])
            for entity_id, _ in self.relation_by_movie_.get(movie_id, []):
                training_rows.append({
                    "movie_tmdb_id": movie_id,
                    "entity_id": entity_id,
                    "release_date": movie["release_date"],
                    "budget": movie["budget"] if movie["budget"] > 0 else np.nan,
                    "revenue": movie["revenue"] if movie["revenue"] > 0 else np.nan,
                    "financial_pair_valid": int(movie["budget"] > 0 and movie["revenue"] > 0),
                    "success": int(train_target.iloc[position]),
                    "reference_source": "fold_training",
                })
        external = _load_external_history(HISTORY_PATHS[self.group]).copy()
        external["reference_source"] = "external_history"
        reference = pd.concat([external, pd.DataFrame(training_rows)], ignore_index=True)
        reference = reference.drop_duplicates(["movie_tmdb_id", "entity_id"], keep="first")
        reference["release_date"] = pd.to_datetime(reference["release_date"], errors="coerce")
        reference["budget"] = pd.to_numeric(reference["budget"], errors="coerce")
        reference["revenue"] = pd.to_numeric(reference["revenue"], errors="coerce")
        reference["success"] = pd.to_numeric(reference["success"], errors="coerce")
        self.reference_ = reference[reference["release_date"].notna()].reset_index(drop=True)
        self.reference_by_entity_ = {
            int(entity_id): group.sort_values(["release_date", "movie_tmdb_id"]).reset_index(drop=True)
            for entity_id, group in self.reference_.groupby("entity_id", sort=False)
        }
        global_mature = (
            self.reference_[self.reference_["success"].notna()]
            .sort_values(["release_date", "movie_tmdb_id"])
            .drop_duplicates("movie_tmdb_id")
        )
        self.global_mature_dates_ = global_mature["release_date"].to_numpy(dtype="datetime64[ns]")
        self.global_mature_success_prefix_ = np.cumsum(
            global_mature["success"].to_numpy(dtype=float)
        )
        return self

    def _global_prior(self, query_date: pd.Timestamp, query_id: int) -> float:
        cutoff = query_date - timedelta(days=self.settings.maturity_lag_days)
        del query_id  # A movie at query_date cannot satisfy the 365-day maturity cutoff.
        count = int(
            np.searchsorted(
                self.global_mature_dates_,
                np.datetime64(cutoff.to_datetime64()),
                side="right",
            )
        )
        if count == 0:
            return 0.5
        return float(self.global_mature_success_prefix_[count - 1] / count)

    def _entity_stats(
        self,
        entity_id: int,
        query_id: int,
        query_date: pd.Timestamp,
        global_prior: float,
    ) -> dict[str, float]:
        entity_reference = self.reference_by_entity_.get(entity_id)
        if entity_reference is None:
            prior = pd.DataFrame()
        else:
            prior = entity_reference[
                entity_reference["release_date"].lt(query_date)
                & entity_reference["movie_tmdb_id"].ne(query_id)
            ]
        if prior.empty:
            return {
                "prior_movie_count": 0.0,
                "valid_financial_count": 0.0,
                "years_experience": np.nan,
                "years_since_last": np.nan,
                "smoothed_success_rate": global_prior,
                "median_log_budget": np.nan,
                "recent_success_rate": global_prior,
                "recency_weighted_success_rate": global_prior,
            }

        cutoff = query_date - timedelta(days=self.settings.maturity_lag_days)
        mature = prior[prior["release_date"].le(cutoff) & prior["success"].notna()].copy()
        count = float(len(prior))
        valid_count = float(len(mature))
        success_sum = float(mature["success"].sum()) if len(mature) else 0.0
        smoothed = (success_sum + self.settings.smoothing * global_prior) / (
            valid_count + self.settings.smoothing
        )
        budgets = prior.loc[prior["budget"].gt(0), "budget"]
        median_log_budget = float(np.log1p(budgets).median()) if len(budgets) else np.nan
        recent = mature.tail(self.settings.recent_movie_count)
        recent_rate = float(recent["success"].mean()) if len(recent) else global_prior
        if len(mature):
            age_years = (query_date - mature["release_date"]).dt.days / 365.25
            weight = np.exp(-np.log(2.0) * age_years / self.settings.half_life_years)
            recency_rate = float(np.average(mature["success"], weights=weight))
        else:
            recency_rate = global_prior
        return {
            "prior_movie_count": count,
            "valid_financial_count": valid_count,
            "years_experience": float((query_date - prior["release_date"].min()).days / 365.25),
            "years_since_last": float((query_date - prior["release_date"].max()).days / 365.25),
            "smoothed_success_rate": smoothed,
            "median_log_budget": median_log_budget,
            "recent_success_rate": recent_rate,
            "recency_weighted_success_rate": recency_rate,
        }

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.reference_ is None:
            raise RuntimeError("EntityHistoryBlockBuilder must be fit before transform.")
        rows: list[dict[str, float]] = []
        source = data.reset_index(drop=True).copy()
        source["release_date"] = pd.to_datetime(source["release_date"], errors="coerce")
        if source["release_date"].isna().any():
            raise ValueError("Entity history query has invalid release dates.")
        prefix = "company" if self.group == "production_company" else self.group

        for _, movie in source.iterrows():
            movie_id = int(movie["tmdb_id"])
            query_date = movie["release_date"]
            entities = self.relation_by_movie_.get(movie_id, [])
            global_prior = self._global_prior(query_date, movie_id)
            stats = [
                self._entity_stats(entity_id, movie_id, query_date, global_prior)
                for entity_id, _ in entities
            ]
            if self.group == "cast":
                weights = np.asarray(
                    [1.0 / ((order if order is not None else self.settings.top_cast_k) + 1.0) for _, order in entities],
                    dtype=float,
                )
            else:
                weights = np.ones(len(stats), dtype=float)
            with_history = np.asarray([item["prior_movie_count"] > 0 for item in stats], dtype=bool)

            def values(metric: str) -> np.ndarray:
                return np.asarray([item[metric] for item in stats], dtype=float)

            def weighted_mean(metric: str) -> float:
                metric_values = values(metric)
                valid = np.isfinite(metric_values)
                if not valid.any():
                    return np.nan
                return float(np.average(metric_values[valid], weights=weights[valid]))

            def safe_max(metric: str) -> float:
                metric_values = values(metric)
                return float(np.nanmax(metric_values)) if np.isfinite(metric_values).any() else np.nan

            def safe_min(metric: str) -> float:
                metric_values = values(metric)
                return float(np.nanmin(metric_values)) if np.isfinite(metric_values).any() else np.nan

            rows.append({
                f"{prefix}_entity_count": float(len(entities)),
                f"{prefix}_entities_with_history": float(with_history.sum()),
                f"{prefix}_history_available": float(with_history.any()),
                f"{prefix}_history_coverage_rate": float(with_history.mean()) if len(with_history) else 0.0,
                f"{prefix}_prior_movie_count_mean": weighted_mean("prior_movie_count") if stats else 0.0,
                f"{prefix}_prior_movie_count_max": safe_max("prior_movie_count") if stats else 0.0,
                f"{prefix}_valid_financial_count_mean": weighted_mean("valid_financial_count") if stats else 0.0,
                f"{prefix}_valid_financial_count_max": safe_max("valid_financial_count") if stats else 0.0,
                f"{prefix}_years_experience_mean": weighted_mean("years_experience") if stats else np.nan,
                f"{prefix}_years_experience_max": safe_max("years_experience") if stats else np.nan,
                f"{prefix}_years_since_last_mean": weighted_mean("years_since_last") if stats else np.nan,
                f"{prefix}_years_since_last_min": safe_min("years_since_last") if stats else np.nan,
                f"{prefix}_smoothed_success_rate_mean": weighted_mean("smoothed_success_rate") if stats else global_prior,
                f"{prefix}_smoothed_success_rate_max": safe_max("smoothed_success_rate") if stats else global_prior,
                f"{prefix}_median_log_budget_mean": weighted_mean("median_log_budget") if stats else np.nan,
                f"{prefix}_median_log_budget_max": safe_max("median_log_budget") if stats else np.nan,
                f"{prefix}_recent_success_rate_mean": weighted_mean("recent_success_rate") if stats else global_prior,
                f"{prefix}_recency_weighted_success_rate_mean": weighted_mean("recency_weighted_success_rate") if stats else global_prior,
            })
        output = pd.DataFrame(rows, columns=self.numeric_columns_)
        if np.isinf(output.to_numpy(dtype=float)).any():
            raise ValueError(f"{self.group} history block contains Inf.")
        return output

    def fit_transform(self, data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        return self.fit(data, target).transform(data)


class OperationalEntityHistoryBuilder:
    """Keep A+B+franchise fixed and append selected independent history blocks."""

    def __init__(self, groups: tuple[str, ...]) -> None:
        if not groups:
            raise ValueError("At least one entity history group is required.")
        if "distributor" in groups:
            raise ValueError("Distributor history is blocked until an approved source exists.")
        self.groups = tuple(groups)
        self.control_: OperationalFranchiseBuilder | None = None
        self.blocks_: dict[str, EntityHistoryBlockBuilder] = {}
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.feature_columns_: list[str] = []

    def fit_transform(self, data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        raw = data.reset_index(drop=True)
        y = target.reset_index(drop=True)
        self.control_ = OperationalFranchiseBuilder(smoothing=10.0)
        output = self.control_.fit_transform(raw, y)
        self.blocks_ = {}
        extra_columns: list[str] = []
        for group in self.groups:
            block = EntityHistoryBlockBuilder(group)
            output = pd.concat([output, block.fit_transform(raw, y)], axis=1)
            self.blocks_[group] = block
            extra_columns.extend(block.numeric_columns_)
        self.numeric_columns_ = self.control_.numeric_columns_ + extra_columns
        self.categorical_columns_ = self.control_.categorical_columns_.copy()
        self.feature_columns_ = self.numeric_columns_ + self.categorical_columns_
        self._validate_schema()
        return output[self.feature_columns_]

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.control_ is None or set(self.blocks_) != set(self.groups):
            raise RuntimeError("OperationalEntityHistoryBuilder must be fit before transform.")
        raw = data.reset_index(drop=True)
        output = self.control_.transform(raw)
        for group in self.groups:
            output = pd.concat([output, self.blocks_[group].transform(raw)], axis=1)
        self._validate_schema()
        return output[self.feature_columns_]

    def _validate_schema(self) -> None:
        forbidden = set(self.feature_columns_) & FORBIDDEN_PREDICTORS
        forbidden_budget = set(self.feature_columns_) & OPERATIONAL_BUDGET_V1_BLACKLIST
        if forbidden or forbidden_budget:
            raise ValueError(f"Forbidden feature detected: {sorted(forbidden | forbidden_budget)}")
