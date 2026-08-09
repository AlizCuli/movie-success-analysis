"""Kiểm tra các invariant của repository cuối mà không cần dữ liệu local."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from src.pre_release_features import FORBIDDEN_PREDICTORS
from src.predict_xgboost import REQUIRED_INPUT_COLUMNS
from src.preprocess_movies import INPUT_PATH


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MANIFEST_PATH = MODELS / "xgboost_pre_release_operational_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RepositoryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_preprocessing_reads_tmdb_csv_directly(self) -> None:
        self.assertEqual(INPUT_PATH.name, "tmdb_movies_2000_2025.csv")

    def test_prediction_schema_is_pre_release_only(self) -> None:
        forbidden = set(REQUIRED_INPUT_COLUMNS) & FORBIDDEN_PREDICTORS
        self.assertEqual(sorted(forbidden), [])
        self.assertIn("release_date", REQUIRED_INPUT_COLUMNS)
        self.assertIn("collection_id", REQUIRED_INPUT_COLUMNS)

    def test_model_contains_no_forbidden_predictor(self) -> None:
        raw_features = set(self.manifest["raw_feature_columns"])
        self.assertEqual(sorted(raw_features & FORBIDDEN_PREDICTORS), [])
        self.assertEqual(self.manifest["forbidden_predictors_found"], [])

    def test_benchmark_is_preserved(self) -> None:
        benchmark = self.manifest["benchmark_outer_oof"]
        self.assertEqual(benchmark["rows"], 1646)
        self.assertAlmostEqual(benchmark["macro_f1"], 0.7194825315671237, places=12)

    def test_model_hashes_match_manifest(self) -> None:
        for file_name, expected_hash in self.manifest["artifact_sha256"].items():
            self.assertEqual(sha256(MODELS / file_name), expected_hash)

    def test_env_template_uses_read_access_token_name(self) -> None:
        content = (ROOT / ".env.example").read_text(encoding="utf-8").strip()
        self.assertEqual(content, "TMDB_API_TOKEN=your_read_access_token_here")


if __name__ == "__main__":
    unittest.main()
