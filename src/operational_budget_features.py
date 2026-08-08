"""Fold-local budget context features for the operational XGBoost experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.operational_franchise_features import OperationalFranchiseBuilder


BUDGET_CONTEXT_NUMERIC = [
    "budget_prior_percentile",
    "log_budget_minus_prior_median",
    "log_prior_budget_count",
    "log_budget_minus_prior_3y_median",
    "log_prior_budget_3y_count",
    "budget_history_available",
    "budget_history_3y_available",
]


class BudgetContextHistoryBuilder:
    """Compare a movie budget only with earlier movies from the fitted partition."""

    def __init__(self, window_years: int = 3) -> None:
        self.window_years = int(window_years)
        self.reference_: pd.DataFrame | None = None

    def fit(
        self,
        data: pd.DataFrame,
        target: pd.Series | None = None,
    ) -> "BudgetContextHistoryBuilder":
        del target  # Budget context tuyệt đối không sử dụng target.
        reference = pd.DataFrame(
            {
                "release_date": pd.to_datetime(data["release_date"], errors="coerce"),
                "log_budget": pd.to_numeric(data["log_budget"], errors="coerce"),
            }
        )
        reference = reference.dropna(subset=["release_date", "log_budget"])
        self.reference_ = reference.sort_values("release_date").reset_index(drop=True)
        return self

    @staticmethod
    def _neutral_row() -> dict[str, float]:
        return {
            "budget_prior_percentile": 0.5,
            "log_budget_minus_prior_median": 0.0,
            "log_prior_budget_count": 0.0,
            "log_budget_minus_prior_3y_median": 0.0,
            "log_prior_budget_3y_count": 0.0,
            "budget_history_available": 0.0,
            "budget_history_3y_available": 0.0,
        }

    def _calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.reference_ is None:
            raise RuntimeError("BudgetContextHistoryBuilder must be fit before transform.")

        query_dates = pd.to_datetime(data["release_date"], errors="coerce")
        query_budgets = pd.to_numeric(data["log_budget"], errors="coerce")
        reference_dates = self.reference_["release_date"]
        reference_budgets = self.reference_["log_budget"]
        rows: list[dict[str, float]] = []

        for query_date, query_budget in zip(query_dates, query_budgets, strict=True):
            if pd.isna(query_date):
                rows.append(self._neutral_row())
                continue

            earlier_mask = reference_dates < query_date
            earlier_budgets = reference_budgets.loc[earlier_mask].to_numpy(dtype=float)
            window_start = query_date - pd.DateOffset(years=self.window_years)
            window_mask = earlier_mask & (reference_dates >= window_start)
            window_budgets = reference_budgets.loc[window_mask].to_numpy(dtype=float)

            row = self._neutral_row()
            prior_count = len(earlier_budgets)
            window_count = len(window_budgets)
            row["log_prior_budget_count"] = float(np.log1p(prior_count))
            row["log_prior_budget_3y_count"] = float(np.log1p(window_count))
            row["budget_history_available"] = float(prior_count > 0)
            row["budget_history_3y_available"] = float(window_count > 0)

            if pd.notna(query_budget) and prior_count:
                less_or_equal = int(np.count_nonzero(earlier_budgets <= float(query_budget)))
                row["budget_prior_percentile"] = (less_or_equal + 0.5) / (prior_count + 1.0)
                row["log_budget_minus_prior_median"] = float(
                    query_budget - np.median(earlier_budgets)
                )
            if pd.notna(query_budget) and window_count:
                row["log_budget_minus_prior_3y_median"] = float(
                    query_budget - np.median(window_budgets)
                )
            rows.append(row)

        output = pd.DataFrame(rows, columns=BUDGET_CONTEXT_NUMERIC)
        if not np.isfinite(output.to_numpy(dtype=float)).all():
            raise ValueError("Budget context contains NaN/Inf.")
        return output

    def fit_transform(
        self,
        data: pd.DataFrame,
        target: pd.Series | None = None,
    ) -> pd.DataFrame:
        return self.fit(data, target)._calculate(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._calculate(data)


class OperationalBudgetContextBuilder:
    """A+B + franchise history augmented by global point-in-time budget context."""

    def __init__(self, franchise_smoothing: float = 10.0, window_years: int = 3) -> None:
        self.franchise_smoothing = float(franchise_smoothing)
        self.window_years = int(window_years)
        self.base_: OperationalFranchiseBuilder | None = None
        self.budget_: BudgetContextHistoryBuilder | None = None
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.feature_columns_: list[str] = []

    def fit_transform(self, data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        clean_data = data.reset_index(drop=True)
        clean_target = target.reset_index(drop=True)
        self.base_ = OperationalFranchiseBuilder(smoothing=self.franchise_smoothing)
        base_features = self.base_.fit_transform(clean_data, clean_target)
        self.budget_ = BudgetContextHistoryBuilder(window_years=self.window_years)
        budget_features = self.budget_.fit_transform(clean_data)
        self.numeric_columns_ = self.base_.numeric_columns_ + BUDGET_CONTEXT_NUMERIC
        self.categorical_columns_ = self.base_.categorical_columns_.copy()
        self.feature_columns_ = self.numeric_columns_ + self.categorical_columns_
        return self._combine(base_features, budget_features)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.base_ is None or self.budget_ is None:
            raise RuntimeError("OperationalBudgetContextBuilder must be fit before transform.")
        clean_data = data.reset_index(drop=True)
        return self._combine(
            self.base_.transform(clean_data),
            self.budget_.transform(clean_data),
        )

    def _combine(
        self,
        base_features: pd.DataFrame,
        budget_features: pd.DataFrame,
    ) -> pd.DataFrame:
        output = pd.concat(
            [base_features.reset_index(drop=True), budget_features.reset_index(drop=True)],
            axis=1,
        )
        output = output[self.feature_columns_]
        if np.isinf(output[self.numeric_columns_].to_numpy(dtype=float)).any():
            raise ValueError("Operational budget features contain Inf.")
        return output
