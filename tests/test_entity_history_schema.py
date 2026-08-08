"""Schema tests using the project's real TMDb records."""

from __future__ import annotations

import unittest

from src.entity_history_schema import (
    load_base_movies,
    load_config,
    load_modeling,
    normalized_movie_record,
    target_entity_relations,
)


class EntityHistorySchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        cls.modeling = load_modeling()
        cls.relations = target_entity_relations(int(cls.config["top_cast_k"]))
        cls.base_movies, _ = load_base_movies()

    def test_fixed_target_set_is_unchanged(self) -> None:
        self.assertEqual(len(self.modeling), 1646)
        self.assertTrue(self.modeling["tmdb_id"].is_unique)

    def test_entity_relations_use_stable_integer_ids(self) -> None:
        for group in ("director", "production_company", "cast"):
            self.assertTrue(self.relations[group])
            self.assertTrue(all(isinstance(row["entity_id"], int) for row in self.relations[group]))
            self.assertTrue(all(isinstance(row["target_tmdb_id"], int) for row in self.relations[group]))

    def test_top_cast_is_fixed_and_billing_ordered(self) -> None:
        by_movie: dict[int, list[int]] = {}
        for row in self.relations["cast"]:
            by_movie.setdefault(row["target_tmdb_id"], []).append(row["billing_order"])
        self.assertTrue(all(len(orders) <= self.config["top_cast_k"] for orders in by_movie.values()))
        self.assertTrue(all(orders == sorted(orders) for orders in by_movie.values()))

    def test_all_directors_are_preserved(self) -> None:
        counts: dict[int, int] = {}
        for row in self.relations["director"]:
            counts[row["target_tmdb_id"]] = counts.get(row["target_tmdb_id"], 0) + 1
        self.assertEqual(len(counts), 1646)
        self.assertTrue(any(count > 1 for count in counts.values()))

    def test_normalized_movie_has_no_audience_feedback_fields(self) -> None:
        record = normalized_movie_record(
            self.base_movies[0],
            source_endpoint="local-test-read",
            retrieved_at_utc="",
            snapshot_version=self.config["snapshot_version"],
        )
        forbidden = set(self.config["forbidden_predictors"])
        self.assertFalse(forbidden & set(record))
        self.assertIn("budget", record)
        self.assertIn("revenue", record)
        self.assertIn("release_date", record)


if __name__ == "__main__":
    unittest.main()
