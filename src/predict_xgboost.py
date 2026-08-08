"""Nạp gói XGBoost chính thức và dự đoán từ metadata pre-release."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "models" / "xgboost_pre_release_operational_bundle.joblib"


def load_bundle(path: Path = DEFAULT_BUNDLE) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy model bundle: {path}")
    return joblib.load(path)


def predict_movies(data: pd.DataFrame, bundle: dict) -> pd.DataFrame:
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
    parser.add_argument("input_csv", type=Path, help="CSV metadata pre-release đã cấu trúc hóa")
    parser.add_argument("output_csv", type=Path, help="Nơi lưu kết quả dự đoán")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()

    data = pd.read_csv(args.input_csv)
    predictions = predict_movies(data, load_bundle(args.bundle))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_csv, index=False)
    print(f"Đã dự đoán {len(predictions)} phim: {args.output_csv}")


if __name__ == "__main__":
    main()
