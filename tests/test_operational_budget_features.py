"""Leakage and reproducibility checks using the project's real movie records."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.operational_budget_features import (
    BUDGET_CONTEXT_NUMERIC,
    BudgetContextHistoryBuilder,
    OperationalBudgetContextBuilder,
)
from src.pre_release_features import FORBIDDEN_PREDICTORS, load_pre_release_modeling_data


class BudgetContextHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data, _ = load_pre_release_modeling_data()
        cls.data = cls.data.sort_values(["release_date", "tmdb_id"]).reset_index(drop=True)
        cls.target = cls.data["is_successful"].astype(int).reset_index(drop=True)

    def test_strictly_earlier_reference_only(self) -> None:
        builder = BudgetContextHistoryBuilder().fit(self.data)
        query = self.data.iloc[[len(self.data) // 2]].copy()
        result = builder.transform(query).iloc[0]
        query_date = pd.to_datetime(query.iloc[0]["release_date"])
        expected = self.data[
            pd.to_datetime(self.data["release_date"], errors="coerce") < query_date
        ]
        self.assertEqual(
            int(round(np.expm1(result["log_prior_budget_count"]))),
            len(expected),
        )

    def test_validation_transform_does_not_update_state(self) -> None:
        train = self.data.iloc[:1000]
        valid = self.data.iloc[1000:1010]
        builder = BudgetContextHistoryBuilder().fit(train)
        before = builder.reference_.copy(deep=True)
        first = builder.transform(valid)
        second = builder.transform(valid)
        pd.testing.assert_frame_equal(before, builder.reference_)
        pd.testing.assert_frame_equal(first, second)

    def test_target_does_not_change_budget_features(self) -> None:
        sample = self.data.iloc[:500]
        target = self.target.iloc[:500]
        first = BudgetContextHistoryBuilder().fit_transform(sample, target)
        second = BudgetContextHistoryBuilder().fit_transform(
            sample,
            target.iloc[::-1].reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(first, second)

    def test_missing_inputs_use_neutral_fallback(self) -> None:
        builder = BudgetContextHistoryBuilder().fit(self.data.iloc[:100])
        query = self.data.iloc[[100]].copy()
        query["release_date"] = pd.NA
        query["log_budget"] = np.nan
        result = builder.transform(query).iloc[0]
        self.assertEqual(result["budget_prior_percentile"], 0.5)
        self.assertEqual(result["log_prior_budget_count"], 0.0)
        self.assertEqual(result["budget_history_available"], 0.0)
        self.assertTrue(np.isfinite(result.to_numpy(dtype=float)).all())

    def test_equal_date_rows_are_excluded(self) -> None:
        dates = pd.to_datetime(self.data["release_date"], errors="coerce")
        duplicate_dates = dates[dates.duplicated(keep=False)]
        self.assertFalse(duplicate_dates.empty)
        query_date = duplicate_dates.iloc[0]
        query = self.data.loc[dates.eq(query_date)].iloc[[0]]
        result = BudgetContextHistoryBuilder().fit(self.data).transform(query).iloc[0]
        expected = int((dates < query_date).sum())
        self.assertEqual(
            int(round(np.expm1(result["log_prior_budget_count"]))),
            expected,
        )

    def test_combined_schema_has_no_forbidden_predictor(self) -> None:
        sample = self.data.iloc[:300]
        target = self.target.iloc[:300]
        builder = OperationalBudgetContextBuilder()
        transformed = builder.fit_transform(sample, target)
        self.assertTrue(set(BUDGET_CONTEXT_NUMERIC).issubset(transformed.columns))
        self.assertFalse(set(builder.feature_columns_) & FORBIDDEN_PREDICTORS)
        self.assertFalse(np.isinf(transformed[builder.numeric_columns_].to_numpy(dtype=float)).any())
        self.assertTrue(np.isfinite(transformed[BUDGET_CONTEXT_NUMERIC]).all().all())


if __name__ == "__main__":
    unittest.main()
