"""Point-in-time franchise features for the operational XGBoost experiment."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.pre_release_features import PreReleaseFeatureBuilder


FRANCHISE_NUMERIC = [
    "collection_prior_movie_count",
    "collection_prior_success_rate",
    "collection_prior_mean_log_budget",
    "collection_years_since_previous",
]


class FranchiseHistoryBuilder:
    """Compute collection history using only earlier movies in the training partition."""

    def __init__(self, smoothing: float = 10.0) -> None:
        self.smoothing = float(smoothing)
        self.reference_: pd.DataFrame | None = None

    def fit(self, data: pd.DataFrame, target: pd.Series) -> "FranchiseHistoryBuilder":
        reference = pd.DataFrame(
            {
                "collection_id": pd.to_numeric(data["collection_id"], errors="coerce"),
                "release_date": pd.to_datetime(data["release_date"], errors="coerce"),
                "success": target.reset_index(drop=True).astype(int),
                "log_budget": pd.to_numeric(data["log_budget"], errors="coerce"),
            }
        )
        if reference["release_date"].isna().any():
            raise ValueError("Franchise history requires a valid release_date.")
        self.reference_ = reference.reset_index(drop=True)
        return self

    def _calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.reference_ is None:
            raise RuntimeError("FranchiseHistoryBuilder must be fit before transform.")
        queries = pd.DataFrame(
            {
                "collection_id": pd.to_numeric(data["collection_id"], errors="coerce"),
                "release_date": pd.to_datetime(data["release_date"], errors="coerce"),
                "query_id": np.arange(len(data)),
            }
        )
        if queries["release_date"].isna().any():
            raise ValueError("Franchise history query has an invalid release_date.")

        reference_groups = list(
            self.reference_.sort_values("release_date").groupby("release_date", sort=True)
        )
        group_index = 0
        global_count = 0.0
        global_success = 0.0
        collection_stats: dict[int, dict[str, object]] = defaultdict(
            lambda: {"count": 0.0, "success": 0.0, "budget_sum": 0.0, "budget_count": 0.0, "last_date": None}
        )
        rows: list[dict[str, float]] = []

        def update(movie: pd.Series) -> None:
            nonlocal global_count, global_success
            collection = movie["collection_id"]
            global_count += 1.0
            global_success += float(movie["success"])
            if pd.isna(collection):
                return
            stats = collection_stats[int(collection)]
            stats["count"] = float(stats["count"]) + 1.0
            stats["success"] = float(stats["success"]) + float(movie["success"])
            budget = movie["log_budget"]
            if pd.notna(budget):
                stats["budget_sum"] = float(stats["budget_sum"]) + float(budget)
                stats["budget_count"] = float(stats["budget_count"]) + 1.0
            stats["last_date"] = movie["release_date"]

        for date, group in queries.sort_values("release_date").groupby("release_date", sort=True):
            while group_index < len(reference_groups) and reference_groups[group_index][0] < date:
                for _, movie in reference_groups[group_index][1].iterrows():
                    update(movie)
                group_index += 1
            global_prior = global_success / global_count if global_count else 0.5
            for _, movie in group.iterrows():
                collection = movie["collection_id"]
                if pd.isna(collection):
                    count, success, budget_mean, gap_years = 0.0, global_prior, np.nan, np.nan
                else:
                    stats = collection_stats[int(collection)]
                    count = float(stats["count"])
                    success = (float(stats["success"]) + self.smoothing * global_prior) / (count + self.smoothing)
                    budget_count = float(stats["budget_count"])
                    budget_mean = float(stats["budget_sum"]) / budget_count if budget_count else np.nan
                    previous_date = stats["last_date"]
                    gap_years = (date - previous_date).days / 365.25 if previous_date is not None else np.nan
                rows.append(
                    {
                        "query_id": int(movie["query_id"]),
                        "collection_prior_movie_count": count,
                        "collection_prior_success_rate": success,
                        "collection_prior_mean_log_budget": budget_mean,
                        "collection_years_since_previous": gap_years,
                    }
                )
        return pd.DataFrame(rows).sort_values("query_id").drop(columns="query_id").reset_index(drop=True)

    def fit_transform(self, data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        return self.fit(data, target)._calculate(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._calculate(data)


class OperationalFranchiseBuilder:
    """A+B operational features augmented by date-safe collection history."""

    def __init__(self, smoothing: float = 10.0) -> None:
        self.smoothing = float(smoothing)
        self.base_: PreReleaseFeatureBuilder | None = None
        self.history_: FranchiseHistoryBuilder | None = None
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.feature_columns_: list[str] = []

    def fit_transform(self, data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        self.base_ = PreReleaseFeatureBuilder()
        base_features = self.base_.fit_transform(data.reset_index(drop=True), target.reset_index(drop=True))
        self.history_ = FranchiseHistoryBuilder(self.smoothing)
        history = self.history_.fit_transform(data.reset_index(drop=True), target.reset_index(drop=True))
        self.numeric_columns_ = self.base_.numeric_columns_ + FRANCHISE_NUMERIC
        self.categorical_columns_ = self.base_.categorical_columns_.copy()
        self.feature_columns_ = self.numeric_columns_ + self.categorical_columns_
        return self._combine(base_features, history)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.base_ is None or self.history_ is None:
            raise RuntimeError("OperationalFranchiseBuilder must be fit before transform.")
        return self._combine(self.base_.transform(data.reset_index(drop=True)), self.history_.transform(data.reset_index(drop=True)))

    def _combine(self, base_features: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
        output = pd.concat([base_features.reset_index(drop=True), history.reset_index(drop=True)], axis=1)
        for column in self.numeric_columns_:
            output[column] = pd.to_numeric(output[column], errors="coerce")
        for column in self.categorical_columns_:
            output[column] = output[column].fillna("__MISSING__").astype(str)
        output = output[self.feature_columns_]
        if np.isinf(output[self.numeric_columns_].to_numpy(dtype=float)).any():
            raise ValueError("Franchise features contain Inf.")
        return output
