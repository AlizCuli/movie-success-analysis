"""Nạp gói XGBoost chính thức và dự đoán từ metadata pre-release."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "models" / "xgboost_pre_release_operational_bundle.joblib"

REQUIRED_INPUT_COLUMNS = [
    "log_budget",
    "runtime",
    "release_year",
    "release_month",
    "genre_count",
    "primary_genre",
    "original_language",
    "production_country_count",
    "primary_country",
    "production_company_count",
    "genres",
    "production_countries",
    "production_companies",
    "release_date",
    "is_collection",
    "collection_id",
    "primary_company_id",
    "company_count",
    "spoken_language_count",
    "cast_count",
    "crew_count",
    "certification",
    "theatrical_country_count",
    "release_event_count",
    "overview_word_count",
    "has_tagline",
    "keyword_count",
]


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_bundle(path: Path = DEFAULT_BUNDLE) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy model bundle: {path}")
    return joblib.load(path)


def predict_movies(data: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_INPUT_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError("CSV đầu vào thiếu cột bắt buộc: " + ", ".join(missing))

    builder = bundle["feature_builder"]
    transformer = bundle["preprocessor"]
    fitted_model = bundle["model"]
    threshold = float(bundle["threshold"])

    features = builder.transform(data.reset_index(drop=True))
    matrix = transformer.transform(features)
    probability = fitted_model.predict_proba(matrix)[:, 1]
    if not np.isfinite(probability).all():
        raise ValueError("Xác suất dự đoán chứa NaN hoặc Inf.")

    result = pd.DataFrame(
        {
            "probability_success": probability,
            "threshold": threshold,
            "prediction": (probability >= threshold).astype(int),
        }
    )
    if "tmdb_id" in data.columns:
        result.insert(0, "tmdb_id", data["tmdb_id"].reset_index(drop=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, nargs="?", help="CSV metadata pre-release đã cấu trúc hóa")
    parser.add_argument("output_csv", type=Path, nargs="?", help="Nơi lưu kết quả dự đoán")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--show-schema",
        action="store_true",
        help="In danh sách cột đầu vào bắt buộc rồi thoát",
    )
    args = parser.parse_args()

    if args.show_schema:
        print("Các cột bắt buộc:")
        for column in REQUIRED_INPUT_COLUMNS:
            print(f"- {column}")
        print("Cột tùy chọn: tmdb_id")
        return
    if args.input_csv is None or args.output_csv is None:
        parser.error("input_csv và output_csv là bắt buộc, trừ khi dùng --show-schema")

    data = pd.read_csv(args.input_csv)
    predictions = predict_movies(data, load_bundle(args.bundle))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_csv, index=False)
    print(f"Đã dự đoán {len(predictions)} phim: {args.output_csv}")


if __name__ == "__main__":
    main()
