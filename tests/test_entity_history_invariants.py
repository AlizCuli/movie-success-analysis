"""Integration checks against the real local entity-history snapshot."""

from __future__ import annotations

import unittest
from datetime import timedelta

import numpy as np
import pandas as pd

from src.entity_history_features import EntityHistoryBlockBuilder, OperationalEntityHistoryBuilder
from src.entity_history_schema import (
    CAST_PATH,
    COMPANIES_PATH,
    DIRECTORS_PATH,
    load_modeling,
    read_jsonl,
)
from src.pre_release_features import FORBIDDEN_PREDICTORS, load_pre_release_modeling_data


class EntityHistorySnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (DIRECTORS_PATH, COMPANIES_PATH, CAST_PATH)
        if not all(path.exists() and path.stat().st_size > 0 for path in required):
            raise unittest.SkipTest("Real entity-history snapshot has not been finalized yet.")
        cls.modeling = load_modeling()

    def test_external_history_excludes_all_target_ids(self) -> None:
        target_ids = set(self.modeling["tmdb_id"].astype(int))
        for path in (DIRECTORS_PATH, COMPANIES_PATH, CAST_PATH):
            history_ids = {int(row["movie_tmdb_id"]) for row in read_jsonl(path)}
            self.assertFalse(target_ids & history_ids)

    def test_external_history_keys_are_unique(self) -> None:
        for path in (DIRECTORS_PATH, COMPANIES_PATH, CAST_PATH):
            frame = pd.DataFrame(read_jsonl(path))
            self.assertFalse(frame.duplicated(["movie_tmdb_id", "entity_id", "role"]).any())

    def test_validation_transform_does_not_update_state(self) -> None:
        raw, target = load_pre_release_modeling_data()
        train = raw.iloc[:500].reset_index(drop=True)
        valid = raw.iloc[500:510].reset_index(drop=True)
        builder = EntityHistoryBlockBuilder("director").fit(train, target.iloc[:500])
        before = builder.reference_.copy(deep=True)
        first = builder.transform(valid)
        second = builder.transform(valid)
        pd.testing.assert_frame_equal(before, builder.reference_)
        pd.testing.assert_frame_equal(first, second)

    def test_entity_stats_exclude_equal_and_future_dates(self) -> None:
        raw, target = load_pre_release_modeling_data()
        train = raw.iloc[:700].reset_index(drop=True)
        builder = EntityHistoryBlockBuilder("director").fit(train, target.iloc[:700])
        query = train.sort_values("release_date").iloc[len(train) // 2]
        movie_id = int(query["tmdb_id"])
        query_date = pd.to_datetime(query["release_date"])
        entity_id = builder.relation_by_movie_[movie_id][0][0]
        stats = builder._entity_stats(
            entity_id,
            movie_id,
            query_date,
            builder._global_prior(query_date, movie_id),
        )
        reference = builder.reference_by_entity_[entity_id]
        expected = reference[
            reference["release_date"].lt(query_date)
            & reference["movie_tmdb_id"].ne(movie_id)
        ]
        self.assertEqual(stats["prior_movie_count"], float(len(expected)))

    def test_financial_history_obeys_365_day_maturity_lag(self) -> None:
        raw, target = load_pre_release_modeling_data()
        train = raw.iloc[:700].reset_index(drop=True)
        builder = EntityHistoryBlockBuilder("director").fit(train, target.iloc[:700])
        query = train.sort_values("release_date").iloc[-1]
        movie_id = int(query["tmdb_id"])
        query_date = pd.to_datetime(query["release_date"])
        entity_id = builder.relation_by_movie_[movie_id][0][0]
        stats = builder._entity_stats(
            entity_id,
            movie_id,
            query_date,
            builder._global_prior(query_date, movie_id),
        )
        reference = builder.reference_by_entity_[entity_id]
        expected = reference[
            reference["release_date"].le(query_date - timedelta(days=365))
            & reference["movie_tmdb_id"].ne(movie_id)
            & reference["success"].notna()
        ]
        self.assertEqual(stats["valid_financial_count"], float(len(expected)))

    def test_combined_schema_has_no_forbidden_or_budget_v1_features(self) -> None:
        raw, target = load_pre_release_modeling_data()
        builder = OperationalEntityHistoryBuilder(("director",))
        transformed = builder.fit_transform(raw.iloc[:300], target.iloc[:300])
        self.assertFalse(set(builder.feature_columns_) & FORBIDDEN_PREDICTORS)
        self.assertFalse(any(column.startswith("budget_prior_") for column in builder.feature_columns_))
        self.assertFalse(np.isinf(transformed[builder.numeric_columns_].to_numpy(dtype=float)).any())


if __name__ == "__main__":
    unittest.main()
